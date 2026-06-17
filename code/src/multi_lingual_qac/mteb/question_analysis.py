"""Question-level analysis of MTEB retrieval predictions.

Given per-query rankings saved by ``run_mteb_evaluation(..., prediction_dir=...)``
(MTEB ``prediction_folder`` JSON) plus the dataset's queries/corpus/qrels, this
breaks Recall@K / MRR@K down by:

  1. query language
  2. question mode (technical vs semantic) and strategy (random / random_missing / ...)
  3. query origin (original vs synthetic-translation)
  4. cross-lingual targets (same- vs cross-language relevant docs)
  5. query-language x target-language pair matrix (best model)

Question ``mode``/``strategy`` come from the queries config when present, or from a CSV
passed via ``query_metadata_csv`` (joined by question text, with a (corpus_id, language)
fallback). It is dataset-agnostic: each breakdown is skipped gracefully when its data is
absent, so it keeps working for other question sets / datasets.
"""
from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from src.multi_lingual_qac.mteb.evaluation import (
    DEFAULT_MTEB_DATASET_REPO,
    DEFAULT_MTEB_VARIANT,
    _dataset_config_name,
    _infer_language,
    _normalize_dataset_variant,
    _query_language_column,
    _slugify,
)

PREDICTION_GLOB = "*_predictions.json"
DEFAULT_K = 10

# Question-generation attributes (real fields, sourced from the QAC metadata).
MODE_ORDER = ["technical", "semantic"]
STRATEGY_ORDER = ["random", "random_missing", "random_existing", "all", "forced_zh"]
_STRATEGY_NUM = {"0": "forced_zh", "1": "random", "2": "random_missing", "3": "random_existing", "4": "all"}


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def _id_column(columns: list[str], specific: str) -> str:
    if "_id" in columns:
        return "_id"
    if specific in columns:
        return specific
    return columns[0] if columns else specific


def _corpus_language_column(columns: list[str]) -> str | None:
    for name in ("corpus_language", "language"):
        if name in columns:
            return name
    return None


def _strategy_display(value: str) -> str:
    """Canonical display name for a strategy (random_any/numeric -> 'random', etc.)."""
    v = str(value).strip()
    if v.endswith(".0"):
        v = v[:-2]
    if v == "random_any":
        return "random"
    return _STRATEGY_NUM.get(v, v)


def _query_metadata_columns(columns: list[str]) -> tuple[str | None, str | None]:
    """Column names holding the question mode and strategy, if present."""
    mode_col = "mode" if "mode" in columns else None
    strat_col = "strategy_name" if "strategy_name" in columns else ("strategy" if "strategy" in columns else None)
    return mode_col, strat_col


def _load_query_metadata_csv(csv_path: str | Path) -> tuple[dict, dict]:
    """Build text->(mode, strategy) and (corpus_id, lang)->(mode, strategy) maps from a CSV.

    Used to attach question mode/strategy when the dataset's queries config does not carry
    them (e.g. an already-published dataset whose export predates those columns).
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    text_col = "question" if "question" in cols else ("text" if "text" in cols else None)
    mode_col, strat_col = _query_metadata_columns(cols)
    cid_col = "corpus_id" if "corpus_id" in cols else None
    lang_col = next((c for c in ("question_language", "query_language", "language") if c in cols), None)
    by_text: dict[str, tuple] = {}
    by_cid_lang: dict[tuple, tuple] = {}
    for _, row in df.iterrows():
        mode_val = str(row[mode_col]).strip().lower() if mode_col and pd.notna(row[mode_col]) else ""
        strat_val = _strategy_display(row[strat_col]) if strat_col and pd.notna(row[strat_col]) else ""
        meta = (mode_val or None, strat_val or None)
        if text_col and pd.notna(row[text_col]):
            by_text[str(row[text_col]).strip()] = meta
        if cid_col and lang_col and pd.notna(row[cid_col]) and pd.notna(row[lang_col]):
            by_cid_lang[(str(row[cid_col]).strip(), str(row[lang_col]).strip().lower())] = meta
    return by_text, by_cid_lang


def _load_predictions(model_dir: Path) -> dict[str, dict[str, float]] | None:
    files = glob.glob(str(model_dir / PREDICTION_GLOB))
    if not files:
        return None
    payload = json.loads(Path(files[0]).read_text(encoding="utf-8"))
    subsets = [key for key in payload if key != "mteb_model_meta"]
    if not subsets:
        return None
    merged: dict[str, dict[str, float]] = {}
    for subset in subsets:
        for split_preds in payload[subset].values():
            merged.update(split_preds)
    return merged


def _discover_models(
    predictions_dir: Path, model_names: list[str] | None
) -> list[tuple[str, str]]:
    """Return [(label, slug)] pairs that actually have a predictions file."""
    if not predictions_dir.is_dir():
        return []
    pairs: list[tuple[str, str]] = []
    if model_names:
        for name in model_names:
            slug = _slugify(name)
            if (predictions_dir / slug).is_dir():
                pairs.append((name, slug))
    else:
        for child in sorted(p for p in predictions_dir.iterdir() if p.is_dir()):
            pairs.append((child.name, child.name))
    return [(label, slug) for label, slug in pairs if glob.glob(str(predictions_dir / slug / PREDICTION_GLOB))]


def _per_query_metrics(preds, rel, query_lang, corpus_lang, k):
    out = {}
    for qid, rel_set in rel.items():
        if qid not in preds or not rel_set:
            continue
        ranking = [doc for doc, _ in sorted(preds[qid].items(), key=lambda kv: -kv[1])]
        top = set(ranking[:k])
        rr = 0.0
        for rank, doc in enumerate(ranking[:k], start=1):  # MRR@k: only the top k count
            if doc in rel_set:
                rr = 1.0 / rank
                break
        ql = query_lang.get(qid)
        same = {doc for doc in rel_set if corpus_lang.get(doc) == ql} if corpus_lang else set()
        cross = (rel_set - same) if corpus_lang else set()
        out[qid] = {
            "recall": len(rel_set & top) / len(rel_set),
            "rr": rr,
            "hit": 1.0 if (rel_set & top) else 0.0,
            "same_recall": (len(same & top) / len(same)) if same else None,
            "cross_recall": (len(cross & top) / len(cross)) if cross else None,
            "top": top,
            "rel": rel_set,
        }
    return out


def _short_label(name: str) -> str:
    base = name.split("/")[-1]
    return base.replace("paraphrase-multilingual-", "").replace("multilingual-", "")


def _load_summary_metrics(summary_path: Path) -> dict[str, dict]:
    """Map model-slug -> metrics from a run's summary.json (empty if missing/unreadable)."""
    try:
        payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {_slugify(m.get("model_name", "")): m.get("metrics", {}) for m in payload.get("models", [])}


def _summary_model_names(summary_path: Path) -> list[str]:
    """Real model names from a run's summary.json (for nice labels in standalone mode)."""
    try:
        payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except Exception:
        return []
    return [m["model_name"] for m in payload.get("models", []) if m.get("model_name")]


def _make_plots(
    output_dir: Path,
    *,
    pq_by_model: dict,
    labels: list[str],
    langs: list[str],
    query_lang: dict,
    corpus_lang: dict,
    query_synth: dict,
    query_mode: dict,
    query_strategy: dict,
    best: str | None,
    summary_metrics: dict,
    k: int,
) -> Path | None:
    """Render PNG summaries into output_dir/plots. Skips gracefully without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[plots skipped] matplotlib unavailable: {exc}")
        return None

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("mode_same_vs_cross.png", "strategy_original_vs_translation.png"):
        (plots_dir / stale).unlink(missing_ok=True)  # superseded by renamed plots
    short = [_short_label(lb) for lb in labels]
    n = len(labels)

    def grouped_bar(fname, group_labels, values_by_label, ylabel, title, ymax=1.0):
        if not group_labels:
            return
        fig, ax = plt.subplots(figsize=(1.7 * max(len(group_labels), 3) + 1.5, 4.3))
        width = 0.8 / max(n, 1)
        xs = list(range(len(group_labels)))
        for i in range(n):
            offs = [x + (i - (n - 1) / 2) * width for x in xs]
            ax.bar(offs, [v if v is not None else 0.0 for v in values_by_label[i]], width=width, label=short[i])
            for off, v in zip(offs, values_by_label[i]):
                if v is not None:
                    ax.text(off, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(xs)
        ax.set_xticklabels(group_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, ymax)
        ax.legend(fontsize=7, ncol=min(n, 4), loc="upper center", bbox_to_anchor=(0.5, -0.07))
        fig.tight_layout()
        fig.savefig(plots_dir / fname, dpi=130, bbox_inches="tight")
        plt.close(fig)

    def mean_where(label, predicate, key="recall"):
        vals = [pq[key] for qid, pq in pq_by_model[label].items()
                if predicate(qid, pq) and pq.get(key) is not None]
        return _mean(vals) if vals else None

    # 1) overall metrics
    if summary_metrics:
        keys = [("recall_at_10", "Recall@10"), ("ndcg_at_10", "nDCG@10"), ("map_at_10", "MAP@10")]
        groups = [lbl for _, lbl in keys]
        values = [[summary_metrics.get(_slugify(lb), {}).get(mk) for mk, _ in keys] for lb in labels]
        grouped_bar("overall_metrics.png", groups, values, "score", f"Overall retrieval metrics (k={k})")
    else:
        groups = [f"Recall@{k}", f"MRR@{k}", f"hit@{k}"]
        values = [[_mean(p["recall"] for p in pq_by_model[lb].values()),
                   _mean(p["rr"] for p in pq_by_model[lb].values()),
                   _mean(p["hit"] for p in pq_by_model[lb].values())] for lb in labels]
        grouped_bar("overall_metrics.png", groups, values, "score", f"Overall (k={k})")

    # 2) recall by query language
    if langs:
        values = [[mean_where(lb, lambda q, pq, lng=lng: query_lang.get(q) == lng) for lng in langs]
                  for lb in labels]
        grouped_bar("recall_by_language.png", langs, values, f"Recall@{k}", f"Recall@{k} by query language")

    # 3) question mode (technical vs semantic) + strategy
    if query_mode:
        present = [m for m in MODE_ORDER if any(query_mode.get(q) == m for q in query_mode)]
        present += sorted({m for m in query_mode.values()} - set(MODE_ORDER))
        values = [[mean_where(lb, lambda q, pq, m=m: query_mode.get(q) == m) for m in present] for lb in labels]
        grouped_bar("mode.png", present, values, f"Recall@{k}", f"Recall@{k} by question mode")
    if query_strategy:
        present = [s for s in STRATEGY_ORDER if any(query_strategy.get(q) == s for q in query_strategy)]
        present += sorted({s for s in query_strategy.values()} - set(STRATEGY_ORDER))
        values = [[mean_where(lb, lambda q, pq, s=s: query_strategy.get(q) == s) for s in present] for lb in labels]
        grouped_bar("strategy.png", present, values, f"Recall@{k}", f"Recall@{k} by question strategy")

    # 4) cross-lingual targets: same vs cross-language
    def mode_mean(label, key):
        vals = [pq[key] for pq in pq_by_model[label].values() if pq.get(key) is not None]
        return _mean(vals) if vals else None

    if any(mode_mean(lb, "same_recall") is not None or mode_mean(lb, "cross_recall") is not None for lb in labels):
        values = [[mode_mean(lb, "same_recall"), mode_mean(lb, "cross_recall")] for lb in labels]
        grouped_bar("cross_lingual_targets.png", ["same-language", "cross-language"], values,
                    f"Recall@{k}", f"Cross-lingual targets: same vs cross (Recall@{k})")

    # 5) query origin: original vs synthetic-translation
    if query_synth:
        values = [[mean_where(lb, lambda q, pq, want=want: q in query_synth and query_synth[q] is want)
                   for want in (False, True)] for lb in labels]
        grouped_bar("query_origin.png", ["original", "synthetic-translation"], values,
                    f"Recall@{k}", f"Query origin: original vs synthetic (Recall@{k})")

    # 5) language-pair heatmap for the best model
    if best and corpus_lang and langs:
        pair_hits: dict = defaultdict(list)
        for qid, pq in pq_by_model[best].items():
            ql = query_lang.get(qid)
            for doc in pq["rel"]:
                pair_hits[(ql, corpus_lang.get(doc, "?"))].append(1.0 if doc in pq["top"] else 0.0)
        cols = sorted({dl for (_, dl) in pair_hits})
        if cols:
            mat = [[(_mean(pair_hits[(ql, dl)]) if pair_hits.get((ql, dl)) else float("nan")) for dl in cols]
                   for ql in langs]
            fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 2.5, 1.0 * len(langs) + 2))
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(cols)), labels=cols)
            ax.set_yticks(range(len(langs)), labels=langs)
            ax.set_xlabel("relevant-doc language")
            ax.set_ylabel("query language")
            ax.set_title(f"{_short_label(best)}: Recall@{k} by query x doc language")
            for i in range(len(langs)):
                for j in range(len(cols)):
                    v = mat[i][j]
                    nn = len(pair_hits.get((langs[i], cols[j]), []))
                    txt = "-" if v != v else f"{v:.2f}\n(n={nn})"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                            color="white" if (v == v and v < 0.6) else "black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(plots_dir / f"language_pair_heatmap_{_slugify(best)}.png", dpi=130, bbox_inches="tight")
            plt.close(fig)

    # 6) same-language irrelevant share by language (diagnostic; needs summary metrics)
    if summary_metrics and langs:
        bias_langs = [lng for lng in langs
                      if any(summary_metrics.get(_slugify(lb), {}).get(f"same_language_irrelevant_share_at_100_lang_{lng}") is not None
                             for lb in labels)]
        if bias_langs:
            values = [[summary_metrics.get(_slugify(lb), {}).get(f"same_language_irrelevant_share_at_100_lang_{lng}")
                       for lng in bias_langs] for lb in labels]
            grouped_bar("same_language_bias_by_language.png", bias_langs, values, "same-lang share",
                        "Same-language irrelevant share @100 (lower = less language bias)")

    print(f"Plots written to {plots_dir}")
    return plots_dir


def run_question_analysis(
    predictions_dir: str | Path,
    *,
    output_dir: str | Path,
    dataset_repo: str = DEFAULT_MTEB_DATASET_REPO,
    dataset_variant: str = DEFAULT_MTEB_VARIANT,
    revision: str = "main",
    k: int = DEFAULT_K,
    model_names: list[str] | None = None,
    make_plots: bool = True,
    query_metadata_csv: str | Path | None = None,
) -> Path:
    """Write a question-level analysis report from saved per-query predictions.

    Returns the path to the markdown report.
    """
    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variant = _normalize_dataset_variant(dataset_variant)

    if model_names is None:  # recover real names from the run's summary.json for nicer labels
        model_names = _summary_model_names(output_dir.parent / "summary.json") or None

    models = _discover_models(predictions_dir, model_names)
    if not models:
        raise ValueError(
            f"No per-query predictions found under {predictions_dir}. "
            "Run the benchmark with prediction saving enabled first."
        )

    # --- dataset metadata (variant-aware config names) ---
    queries = load_dataset(
        dataset_repo, _dataset_config_name(dataset_repo, revision, variant, "queries"),
        split="train", revision=revision,
    )
    corpus = load_dataset(
        dataset_repo, _dataset_config_name(dataset_repo, revision, variant, "corpus"),
        split="train", revision=revision,
    )
    qrels = load_dataset(
        dataset_repo, _dataset_config_name(dataset_repo, revision, variant, "qrels"),
        split="train", revision=revision,
    )

    q_id_col = _id_column(list(queries.column_names), "query_id")
    q_lang_col = _query_language_column(list(queries.column_names))
    synth_col = "is_synthetic_translation" if "is_synthetic_translation" in queries.column_names else None
    mode_col, strat_col = _query_metadata_columns(list(queries.column_names))

    # If the queries config lacks mode/strategy, optionally attach them from a metadata CSV.
    csv_by_text, csv_by_cid_lang = {}, {}
    if (mode_col is None or strat_col is None) and query_metadata_csv:
        try:
            csv_by_text, csv_by_cid_lang = _load_query_metadata_csv(query_metadata_csv)
        except Exception as exc:
            print(f"[query-metadata skipped] {exc}")

    query_lang, query_synth, query_mode, query_strategy = {}, {}, {}, {}
    for row in queries:
        qid = str(row[q_id_col])
        lang = str(row.get(q_lang_col) or "").strip().lower() if q_lang_col else ""
        lang = lang or _infer_language(qid)  # fall back to language encoded in the id
        if lang:
            query_lang[qid] = lang
        if synth_col is not None:
            query_synth[qid] = str(row.get(synth_col)).strip().lower() in {"true", "1", "yes"}
        mode_val = str(row.get(mode_col) or "").strip().lower() if mode_col else ""
        strat_val = _strategy_display(row.get(strat_col)) if strat_col and row.get(strat_col) not in (None, "") else ""
        if (not mode_val or not strat_val) and (csv_by_text or csv_by_cid_lang):
            meta = (csv_by_text.get(str(row.get("text") or "").strip())
                    or csv_by_cid_lang.get((str(row.get("corpus_id") or "").strip(), lang)))
            if meta:
                mode_val = mode_val or (meta[0] or "")
                strat_val = strat_val or (meta[1] or "")
        if mode_val:
            query_mode[qid] = mode_val
        if strat_val:
            query_strategy[qid] = strat_val

    c_id_col = _id_column(list(corpus.column_names), "corpus_id")
    c_lang_col = _corpus_language_column(list(corpus.column_names))
    corpus_lang = {}
    for row in corpus:
        cid = str(row[c_id_col])
        lang = str(row.get(c_lang_col) or "").strip().lower() if c_lang_col else ""
        lang = lang or _infer_language(cid)  # fall back to language encoded in the id
        if lang:
            corpus_lang[cid] = lang

    qr_cols = list(qrels.column_names)
    qid_col_qr = "query-id" if "query-id" in qr_cols else qr_cols[0]
    cid_col_qr = "corpus-id" if "corpus-id" in qr_cols else (qr_cols[1] if len(qr_cols) > 1 else qr_cols[0])
    score_col = "score" if "score" in qr_cols else (qr_cols[2] if len(qr_cols) > 2 else None)
    rel = defaultdict(set)
    for row in qrels:
        if score_col is None or float(row[score_col]) > 0:  # no score column => binary relevance
            rel[str(row[qid_col_qr])].add(str(row[cid_col_qr]))

    # --- per-query metrics per model ---
    pq_by_model: dict[str, dict] = {}
    for label, slug in models:
        preds = _load_predictions(predictions_dir / slug)
        if preds is None:
            continue
        pq_by_model[label] = _per_query_metrics(preds, rel, query_lang, corpus_lang, k)
    labels = list(pq_by_model)
    if not labels:
        raise ValueError(
            f"Found prediction folders under {predictions_dir} but none contained usable "
            "per-query rankings."
        )

    langs = sorted({v for v in query_lang.values() if v}) if query_lang else []
    best = max(labels, key=lambda lb: _mean(pq["recall"] for pq in pq_by_model[lb].values()))
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)

    emit(f"# Question-level analysis ({dataset_repo}, `{variant}`, Recall@{k} / MRR@{k})\n")

    # --- dataset structure ---
    qids = [q for q in rel if (not query_lang or q in query_lang)]
    emit("## Dataset structure")
    emit(f"- Queries with relevance judgements: {len(qids)}")
    if synth_col is not None:
        n_synth = sum(1 for q in qids if query_synth.get(q))
        emit(f"- Original: {len(qids) - n_synth}  |  synthetic-translation: {n_synth}")
    if query_mode:
        mode_counts = defaultdict(int)
        for q in qids:
            if query_mode.get(q):
                mode_counts[query_mode[q]] += 1
        emit("- Questions by mode: " + ", ".join(f"{m}={mode_counts[m]}" for m in MODE_ORDER if mode_counts.get(m)))
    if query_strategy:
        strat_counts = defaultdict(int)
        for q in qids:
            if query_strategy.get(q):
                strat_counts[query_strategy[q]] += 1
        strat_keys = [s for s in STRATEGY_ORDER if strat_counts.get(s)] + sorted(s for s in strat_counts if s not in STRATEGY_ORDER)
        emit("- Questions by strategy: " + ", ".join(f"{s}={strat_counts[s]}" for s in strat_keys))
    if query_lang:
        by_lang = defaultdict(int)
        for q in qids:
            by_lang[query_lang.get(q, "?")] += 1
        emit("- Queries by language: " + ", ".join(f"{lng}={by_lang[lng]}" for lng in langs if by_lang.get(lng)))
    pairs = sum(len(rel[q]) for q in qids)
    emit(f"- Relevant (query, doc) pairs: {pairs} (avg {pairs / max(len(qids), 1):.2f}/query)")
    emit("- Models analysed: " + ", ".join(labels))
    emit("")

    def grouped_table(title, group_of, key, order=None):
        emit(f"## {title}")
        groups = set()
        for label in labels:
            for qid, pq in pq_by_model[label].items():
                g = group_of(qid)
                if g is not None and pq.get(key) is not None:
                    groups.add(g)
        if order:
            ordered = [g for g in order if g in groups] + sorted(g for g in groups if g not in order)
        else:
            ordered = sorted(groups)
        emit("| Group | n | " + " | ".join(labels) + " |")
        emit("|" + "---|" * (len(labels) + 2))
        for g in ordered:
            ref = labels[0]
            n = sum(1 for qid, pq in pq_by_model[ref].items() if group_of(qid) == g and pq.get(key) is not None)
            cells = []
            for label in labels:
                vals = [pq[key] for qid, pq in pq_by_model[label].items()
                        if group_of(qid) == g and pq.get(key) is not None]
                cells.append(f"{_mean(vals):.3f}" if vals else " - ")
            emit(f"| {g} | {n} | " + " | ".join(cells) + " |")
        emit("")

    if query_lang and len(langs) > 1:
        grouped_table(f"1) Recall@{k} by query language", lambda q: query_lang.get(q), "recall")
        grouped_table(f"   MRR@{k} by query language", lambda q: query_lang.get(q), "rr")

    if query_mode:
        grouped_table(f"2) Recall@{k} by question mode (technical vs semantic)",
                      lambda q: query_mode.get(q), "recall", order=MODE_ORDER)

    if query_strategy:
        grouped_table(f"3) Recall@{k} by question strategy",
                      lambda q: query_strategy.get(q), "recall", order=STRATEGY_ORDER)

    if synth_col is not None:
        grouped_table(
            f"4) Recall@{k} by query origin (original vs synthetic-translation)",
            lambda q: ("synthetic-translation" if query_synth.get(q) else "original"),
            "recall", order=["original", "synthetic-translation"],
        )

    if corpus_lang:
        emit(f"## 5) Cross-lingual targets: same- vs cross-language (mean Recall@{k})")
        emit("| Target | " + " | ".join(labels) + " |")
        emit("|" + "---|" * (len(labels) + 1))
        for mode, key in [("same-language target", "same_recall"), ("cross-language target", "cross_recall")]:
            cells = []
            for label in labels:
                vals = [pq[key] for pq in pq_by_model[label].values() if pq.get(key) is not None]
                cells.append(f"{_mean(vals):.3f}" if vals else " - ")
            emit(f"| {mode} | " + " | ".join(cells) + " |")
        emit("")

    if query_lang and corpus_lang and len(langs) > 1:
        emit(f"## 6) Language-pair Recall@{k} matrix — {best} (best model)")
        emit("Rows = query language, Cols = relevant-doc language; cell = fraction of those")
        emit("relevant docs retrieved in the top %d (n = #relevant pairs)." % k)
        emit("")
        pair_hits = defaultdict(list)
        for qid, pq in pq_by_model[best].items():
            ql = query_lang.get(qid)
            for doc in pq["rel"]:
                pair_hits[(ql, corpus_lang.get(doc, "?"))].append(1.0 if doc in pq["top"] else 0.0)
        cols = sorted({dl for (_, dl) in pair_hits})
        emit("| q\\d | " + " | ".join(cols) + " |")
        emit("|" + "---|" * (len(cols) + 1))
        for ql in langs:
            cells = []
            for dl in cols:
                vals = pair_hits.get((ql, dl), [])
                cells.append(f"{_mean(vals):.2f} ({len(vals)})" if vals else " - ")
            emit(f"| **{ql}** | " + " | ".join(cells) + " |")
        emit("")

    report_path = output_dir / "question_level_analysis.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- per-query CSV (for custom pivots on other questions) ---
    csv_path = output_dir / "question_level_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "query_id", "query_language", "mode", "strategy",
                         "is_synthetic_translation",
                         f"recall_at_{k}", f"rr_at_{k}", f"hit_at_{k}", "n_relevant"])
        for label in labels:
            for qid, pq in pq_by_model[label].items():
                writer.writerow([
                    label, qid, query_lang.get(qid, ""),
                    query_mode.get(qid, ""), query_strategy.get(qid, ""),
                    query_synth.get(qid, "") if synth_col is not None else "",
                    round(pq["recall"], 5), round(pq["rr"], 5), int(pq["hit"]), len(pq["rel"]),
                ])
    print(f"Question-level analysis written to {report_path} and {csv_path}")

    if make_plots:
        summary_metrics = _load_summary_metrics(Path(output_dir).parent / "summary.json")
        try:
            _make_plots(
                output_dir, pq_by_model=pq_by_model, labels=labels, langs=langs,
                query_lang=query_lang, corpus_lang=corpus_lang, query_synth=query_synth,
                query_mode=query_mode, query_strategy=query_strategy,
                best=best, summary_metrics=summary_metrics, k=k,
            )
        except Exception as exc:  # pragma: no cover - plotting must never break the analysis
            print(f"[plots skipped] {exc}")
    return report_path
