"""
"How often does a confusable *wrong* compound beat the right one?" — per language.

This is the analysis the alias graph was built for. For each query (about a
concept, in some language) the benchmark provides gold documents (the right
compound) and hard-negative documents (chemically-similar look-alike compounds,
each labelled with its neighbour concept + relation). Given a retriever's per-query
rankings, we ask: does a look-alike document rank above *every* gold document?
Aggregated per query language, that is the confusion rate.

It runs off the per-query predictions saved by the standard benchmark
(`run_mteb_evaluation(..., prediction_dir=...)`), so it reuses the same encodings
and lives inside the same `reports/runs/<id>/` folder as the standard metrics.
Inputs come from the dataset's `qrels` (gold = score 1, look-alike = score 0) and
`hard_negatives` (corpus-id -> neighbour) configs.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from datasets import load_dataset

from src.multi_lingual_qac.mteb.evaluation import _dataset_config_name, _normalize_dataset_variant
from src.multi_lingual_qac.mteb.question_analysis import _discover_models, _load_predictions

DEFAULT_DATASET = "owner/multi-lingual-qac-alias-graph"
_INF = float("inf")


def _load_config(dataset: str, config: str, variant: str, revision: str):
    """Load one config from a local hf_export dir or from the Hugging Face Hub."""
    path = Path(dataset)
    if path.is_dir():
        return load_dataset("parquet", data_files=str(path / config / f"{config}.parquet"), split="train")
    return load_dataset(
        dataset, _dataset_config_name(dataset, revision, variant, config),
        split="train", revision=revision,
    )


def run_confusion_from_predictions(
    predictions_dir: Path,
    output_dir: Path,
    *,
    dataset_repo: str = DEFAULT_DATASET,
    dataset_variant: str = "multilingual",
    model_names: Optional[Sequence[str]] = None,
    revision: str = "main",
    make_plots: bool = True,
) -> Path:
    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variant = _normalize_dataset_variant(dataset_variant)

    models = _discover_models(predictions_dir, list(model_names) if model_names else None)
    if not models:
        raise ValueError(
            f"No per-query predictions under {predictions_dir}. Run the benchmark with "
            "prediction saving enabled (--analyze-confusion implies --save-predictions)."
        )

    queries = _load_config(dataset_repo, "queries", variant, revision)
    qrels = _load_config(dataset_repo, "qrels", variant, revision)
    try:
        hard_neg = _load_config(dataset_repo, "hard_negatives", variant, revision)
    except Exception:
        hard_neg = []  # dataset has no hard-negative labels (e.g. no look-alikes)

    qid_col = "_id" if "_id" in queries.column_names else "query_id"
    q_lang = {str(r[qid_col]): str(r.get("query_language", "")).strip().lower() for r in queries}
    q_concept = {str(r[qid_col]): str(r.get("concept_name", "")) for r in queries}

    gold: Dict[str, set] = defaultdict(set)
    hardneg: Dict[str, set] = defaultdict(set)
    for r in qrels:
        (gold if float(r["score"]) > 0 else hardneg)[str(r["query-id"])].add(str(r["corpus-id"]))
    neighbor: Dict[tuple, dict] = {
        (str(r["query-id"]), str(r["corpus-id"])): {
            "name": r.get("neighbor_name", "") or r.get("neighbor_chebi_id", ""),
            "relation": r.get("relation", ""),
        }
        for r in hard_neg
    }

    per_query: List[dict] = []
    for label, slug in models:
        preds = _load_predictions(predictions_dir / slug)
        if preds is None:
            continue
        for qid, gset in gold.items():
            hset = hardneg.get(qid)
            if not gset or not hset or qid not in preds:
                continue
            ranking = [d for d, _ in sorted(preds[qid].items(), key=lambda kv: -kv[1])]
            rank = {d: i + 1 for i, d in enumerate(ranking)}
            best_gold = min((rank.get(d, _INF) for d in gset), default=_INF)
            hn_ranked = sorted(hset, key=lambda d: rank.get(d, _INF))
            best_hn_doc = hn_ranked[0]
            best_hn = rank.get(best_hn_doc, _INF)
            if best_gold == _INF and best_hn == _INF:
                continue  # neither retrieved — no signal
            win = best_hn < best_gold
            lab = neighbor.get((qid, best_hn_doc), {}) if win else {}
            per_query.append({
                "model": label, "query_id": qid, "query_language": q_lang.get(qid, ""),
                "concept_name": q_concept.get(qid, ""),
                "best_gold_rank": best_gold if best_gold != _INF else "",
                "best_hardneg_rank": best_hn if best_hn != _INF else "",
                "win": int(win),
                "top_neighbor_name": lab.get("name", ""), "top_relation": lab.get("relation", ""),
            })

    _write(output_dir, per_query, [m[0] for m in models], make_plots)
    return output_dir


def _write(output_dir: Path, per_query: List[dict], models: List[str], make_plots: bool) -> None:
    if per_query:
        with (output_dir / "per_query.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_query[0].keys()))
            w.writeheader()
            w.writerows(per_query)

    # Aggregate per (model, language) + ALL.
    agg: Dict[tuple, List[dict]] = defaultdict(list)
    for r in per_query:
        agg[(r["model"], r["query_language"])].append(r)
        agg[(r["model"], "ALL")].append(r)
    rows = []
    for (model, lang), items in sorted(agg.items()):
        n = len(items)
        wins = sum(r["win"] for r in items)
        ranks = lambda key: [r[key] for r in items if isinstance(r[key], (int, float))]
        gr, hr = ranks("best_gold_rank"), ranks("best_hardneg_rank")
        rows.append({
            "model": model, "query_language": lang, "n_queries": n, "n_wins": wins,
            "confusion_rate": round(wins / n, 4) if n else 0.0,
            "mean_best_gold_rank": round(sum(gr) / len(gr), 2) if gr else "",
            "mean_best_hardneg_rank": round(sum(hr) / len(hr), 2) if hr else "",
        })
    cols = ["model", "query_language", "n_queries", "n_wins", "confusion_rate",
            "mean_best_gold_rank", "mean_best_hardneg_rank"]
    with (output_dir / "confusion_by_language.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    langs = sorted({r["query_language"] for r in per_query if r["query_language"]})
    by = {(r["model"], r["query_language"]): r for r in rows}
    lines = ["# Confusion analysis — does a wrong (look-alike) compound beat the right one?", "",
             "Confusion rate = fraction of queries where a hard-negative (chemically-similar "
             "wrong compound) ranks above every gold document.", "",
             "| model | " + " | ".join(langs) + " | ALL |",
             "| --- | " + " | ".join(["---:"] * (len(langs) + 1)) + " |"]
    for model in models:
        cells = [(f"{by[(model, l)]['confusion_rate']:.1%} (n={by[(model, l)]['n_queries']})"
                  if (model, l) in by else "—") for l in langs + ["ALL"]]
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
    conf = Counter((r["concept_name"], r["top_neighbor_name"], r["top_relation"])
                   for r in per_query if r["win"] and r["top_neighbor_name"])
    lines += ["", "## Most frequent confusions (winning look-alike, all models)", ""]
    if conf:
        lines += ["| right compound | beaten by (look-alike) | relation | count |",
                  "| --- | --- | --- | ---: |"]
        lines += [f"| {a} | {b} | {c} | {n} |" for (a, b, c), n in conf.most_common(30)]
    else:
        lines.append("_No confusions (no look-alike outranked the gold)._")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if make_plots and rows:
        _plot(output_dir / "plots", rows, models, langs)
    print(f"Confusion analysis written to {output_dir}")


def _plot(plots_dir: Path, rows: List[dict], models: List[str], langs: List[str]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[plots skipped] matplotlib unavailable: {exc}")
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    by = {(r["model"], r["query_language"]): r["confusion_rate"] for r in rows}
    groups = langs + ["ALL"]
    n = max(len(models), 1)
    fig, ax = plt.subplots(figsize=(1.7 * max(len(groups), 3) + 1.5, 4.3))
    width = 0.8 / n
    xs = list(range(len(groups)))
    for i, model in enumerate(models):
        offs = [x + (i - (n - 1) / 2) * width for x in xs]
        vals = [by.get((model, g), 0.0) for g in groups]
        ax.bar(offs, vals, width=width, label=model.split("/")[-1])
        for off, v in zip(offs, vals):
            ax.text(off, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups)
    ax.set_ylabel("confusion rate")
    ax.set_title("Confusion rate by query language (lower = better)")
    ax.set_ylim(0, max(1.0, max((by.values() or [0])) * 1.15))
    ax.legend(fontsize=7, ncol=min(n, 4), loc="upper center", bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_by_language.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot written to {plots_dir / 'confusion_by_language.png'}")
