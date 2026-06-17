"""
Full-corpus retrieval-decay evaluation for progressive code-switching (Stage C).

For each base document the builder produced a cumulative ladder of variant docs
(``base__r0 .. base__rN``) and a single fixed query (about the step-1 term). This
module measures, for each embedding model, where each ladder variant ranks **in
the full shared corpus** as the replacement depth grows — the dose-response curve
answering "how fast does retrieval fall apart?".

Method (full-corpus realism):
  * Load the shared corpus haystack (``owner/multilingual-corpus`` by default),
    *removing every language version of the base publications* so the original
    patent is not a near-duplicate twin of the depth-0 variant. The variant docs
    are the only representatives of those publications.
  * Encode the haystack ONCE per model (cached to disk), then encode the fixed
    queries and all ladder variant docs with the same model loader the MTEB runs
    use (``mteb.get_model`` — correct query/document prompts).
  * For each (base, depth k): ``score_k = cos(query, variant_k)`` and
    ``rank_k = 1 + #{haystack doc : cos(query, doc) > score_k}`` (siblings excluded
    because variants are never in the haystack).
  * Aggregate per depth into recall@{1,10,100}, MRR and mean cosine, and break the
    per-step cosine drop down by the swap mode added at that step.

Outputs land under ``reports/runs/progressive_cs`` (raw records parquet, summary
CSVs, a decay-curve plot, and a key-findings note).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.multi_lingual_qac.mteb.evaluation import (
    ALIAS_GRAPH_MODELS,
    DEFAULT_MTEB_RETRIEVAL_PROMPT,
    MODELS_NEEDING_POSITION_IDS_REPAIR,
    _load_corpus_dataset,
    _repair_position_ids_buffers,
    _slugify,
)

_LANG_SUFFIX = re.compile(r"_(de|en|es|fr|zh)$")


def _patent_of(corpus_id: str) -> str:
    """Strip the language suffix: 'EP-4633662-A1_zh' -> 'EP-4633662-A1'."""
    return _LANG_SUFFIX.sub("", str(corpus_id))


_QUERY_SUFFIX = re.compile(r"__q_(de|en|es|fr|zh)$")


def _strip_query_suffix(query_id: str) -> str:
    """Recover the base id from a query id: 'EP-..._en__q_de' -> 'EP-..._en'."""
    return _QUERY_SUFFIX.sub("", str(query_id))


def _doc_text(title: Optional[str], text: Optional[str]) -> str:
    """Compose document text exactly as the shared corpus / MTEB does:
    ``title + ' ' + text`` where the haystack's ``text`` is ``context|abstract``."""
    return (str(title or "") + " " + str(text or "")).strip()


def _load_pcs_config(dataset_repo: str, config: str, revision: str):
    """Load a progressive-dataset config (``corpus``/``queries``/``qrels``) from a HF
    repo or a local dry-run export dir (``<dir>/<config>/<config>.parquet`` or
    ``<dir>/data/<config>/*.parquet``)."""
    from datasets import load_dataset

    path = Path(dataset_repo)
    if path.is_dir():
        pq = path / config / f"{config}.parquet"
        if not pq.exists():
            hits = sorted((path / "data" / config).glob("*.parquet"))
            if not hits:
                raise FileNotFoundError(f"No {config} parquet under {path}")
            pq = hits[0]
        return load_dataset("parquet", data_files=str(pq), split="train")
    return load_dataset(dataset_repo, config, split="train", revision=revision)


def _task_metadata():
    from mteb.abstasks.task_metadata import TaskMetadata

    return TaskMetadata(
        name="ProgressiveCS", description="Progressive code-switching retrieval decay",
        reference=None, type="Retrieval", category="t2t", modalities=["text"],
        eval_splits=["train"], eval_langs=["eng-Latn"], main_score="ndcg_at_10",
        prompt=DEFAULT_MTEB_RETRIEVAL_PROMPT,  # instruct models (Qwen3/e5-instruct) read this; avoids get_task KeyError
        date=None, domains=None, task_subtypes=None, license=None,
        annotations_creators=None, dialect=None, sample_creation=None,
        bibtex_citation=None, dataset={"path": "local/progressive-cs", "revision": "1.0"},
    )


def _load_encoder(name: str):
    import mteb
    from sentence_transformers import SentenceTransformer

    try:
        model = mteb.get_model(name)
    except Exception:
        model = SentenceTransformer(name, trust_remote_code=True)
    if name in MODELS_NEEDING_POSITION_IDS_REPAIR:
        _repair_position_ids_buffers(model)
    return model


def _encode(model, texts: Sequence[str], prompt_type: str, meta, batch_size: int) -> np.ndarray:
    """Encode and L2-normalize. Uses the MTEB wrapper's dataloader interface
    (prompt-correct) for registered models, falling back to a plain
    SentenceTransformer call for unregistered ones (e.g. SapBERT)."""
    from sentence_transformers import SentenceTransformer

    if isinstance(model, SentenceTransformer):
        emb = model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                           show_progress_bar=True)
    else:
        from mteb.types import PromptType
        from mteb._create_dataloaders import _create_dataloader_from_texts

        pt = PromptType.query if prompt_type == "query" else PromptType.document
        dl = _create_dataloader_from_texts(list(texts), batch_size=batch_size)
        emb = model.encode(dl, task_metadata=meta, hf_split="train", hf_subset="default",
                          prompt_type=pt, batch_size=batch_size, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def _haystack_embeddings(
    model, slug: str, hay_ids: List[str], hay_texts: List[str], meta, batch_size: int,
    cache_dir: Optional[Path],
) -> np.ndarray:
    """Encode the haystack once, caching to ``<cache_dir>/<slug>__haystack.npy`` keyed
    on the exact id list (re-encodes if the haystack changed)."""
    cache_npy = ids_json = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_npy = cache_dir / f"{slug}__haystack.npy"
        ids_json = cache_dir / f"{slug}__haystack_ids.json"
        if cache_npy.exists() and ids_json.exists():
            cached_ids = json.loads(ids_json.read_text(encoding="utf-8"))
            if cached_ids == hay_ids:
                print(f"  [cache] reusing haystack embeddings ({len(hay_ids)} docs) for {slug}")
                return np.load(cache_npy)
    print(f"  encoding haystack: {len(hay_ids)} docs")
    emb = _encode(model, hay_texts, "document", meta, batch_size)
    if cache_npy is not None:
        np.save(cache_npy, emb)
        ids_json.write_text(json.dumps(hay_ids), encoding="utf-8")
    return emb


def run_progressive_eval(
    dataset_repo: str,
    haystack_repo: str,
    output_dir: Path,
    *,
    models: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    batch_size: int = 32,
    emb_cache_dir: Optional[Path] = None,
    revision: str = "main",
) -> Path:
    """Run the decay eval against the published progressive dataset.

    ``dataset_repo`` is the progressive benchmark (HF repo id or local dry-run dir)
    holding the ``corpus`` (ladder variants) + ``queries`` configs; ``haystack_repo``
    is the shared corpus used as the realistic retrieval haystack. Returns the path
    to the per-(model, depth) summary CSV.
    """
    import pandas as pd

    models = list(models) if models else list(ALIAS_GRAPH_MODELS)
    output_dir = Path(output_dir)

    # ---- Load ladder variants (from the published corpus config) --------- #
    by_base_depth: Dict[str, Dict[int, str]] = defaultdict(dict)
    variant_text: Dict[str, str] = {}
    mode_at_depth: Dict[str, Dict[int, str]] = defaultdict(dict)
    base_pubs: set = set()
    for r in _load_pcs_config(dataset_repo, "corpus", revision):
        vid, base, depth = str(r["_id"]), str(r["base_id"]), int(r["depth"])
        by_base_depth[base][depth] = vid
        variant_text[vid] = _doc_text(r.get("title"), r.get("text"))
        mode_at_depth[base][depth] = str(r.get("mode_added") or "")
        pub = str(r.get("source_publication_number") or "").strip()
        if pub:
            base_pubs.add(pub)

    # ---- Queries (one per (base, language); strategy="all" => 5 per base) - #
    queries: List[dict] = [
        {"query_id": str(r["_id"]), "text": str(r["text"]),
         "base_id": str(r.get("base_id") or _strip_query_suffix(str(r["_id"]))),
         "query_language": str(r.get("query_language") or "").strip()}
        for r in _load_pcs_config(dataset_repo, "queries", revision)
    ]
    queries = [q for q in queries if q["base_id"] in by_base_depth]

    # ``limit`` selects base documents (so the haystack/encoding stays bounded);
    # every query of a selected base is then evaluated.
    sel_bases = list(dict.fromkeys(q["base_id"] for q in queries))
    if limit is not None:
        sel_bases = sel_bases[:limit]
    sel_set = set(sel_bases)
    queries = [q for q in queries if q["base_id"] in sel_set]
    if not queries:
        raise ValueError(f"No queries to evaluate (check the dataset at {dataset_repo}).")
    print(f"Progressive eval: {len(queries)} queries over {len(sel_bases)} base docs, "
          f"{len(models)} model(s), dataset={dataset_repo}, haystack={haystack_repo}")

    # ---- Haystack (exclude all language versions of the base publications) #
    ds = _load_corpus_dataset(haystack_repo, revision)
    hay_ids: List[str] = []
    hay_texts: List[str] = []
    for row in ds:
        cid = str(row.get("id") or row.get("_id"))
        if _patent_of(cid) in base_pubs:
            continue
        hay_ids.append(cid)
        hay_texts.append(_doc_text(row.get("title"), row.get("text")))
    print(f"  haystack: {len(hay_ids)} docs (dropped {len(base_pubs)} base publications)")

    meta = _task_metadata()
    records: List[dict] = []

    for name in models:
        slug = _slugify(name)
        print(f"\nModel: {name}")
        try:
            model = _load_encoder(name)
        except Exception as exc:  # gated / missing weights / missing extra -> skip
            print(f"  [skip] could not load `{name}`: {exc}")
            continue
        try:
            E_H = _haystack_embeddings(model, slug, hay_ids, hay_texts, meta, batch_size, emb_cache_dir)
            sel_vids = [by_base_depth[b][k] for b in sel_bases for k in sorted(by_base_depth[b])]
            E_V = _encode(model, [variant_text[v] for v in sel_vids], "document", meta, batch_size)
            v_emb = {vid: E_V[i] for i, vid in enumerate(sel_vids)}
            E_Q = _encode(model, [q["text"] for q in queries], "query", meta, batch_size)
        except Exception as exc:
            print(f"  [skip] encoding failed for `{name}`: {exc}")
            continue

        for qi, qmeta in enumerate(queries):
            q = E_Q[qi]
            b = qmeta["base_id"]
            s_H = E_H @ q
            prev = None
            for k in sorted(by_base_depth[b]):
                score = float(v_emb[by_base_depth[b][k]] @ q)
                rank = int(1 + np.count_nonzero(s_H > score))
                mode = mode_at_depth[b].get(k, "")
                records.append({
                    "model": name, "model_slug": slug, "query_id": qmeta["query_id"],
                    "query_language": qmeta["query_language"], "base_id": b, "depth": k,
                    "mode_added": mode, "score": score, "rank": rank, "rr": 1.0 / rank,
                    "r1": int(rank <= 1), "r10": int(rank <= 10), "r100": int(rank <= 100),
                    "cos_drop_from_prev": (prev - score) if prev is not None else np.nan,
                })
                prev = score

    if not records:
        raise RuntimeError("No records produced — all models failed to load/encode.")

    df = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experimental_plots").mkdir(exist_ok=True)
    (output_dir / "key_findings").mkdir(exist_ok=True)
    df.to_parquet(output_dir / "curve_records.parquet", index=False)

    # ---- Per-(model, depth) summary -------------------------------------- #
    summary = (
        df.groupby(["model", "depth"])
        .agg(n=("query_id", "nunique"), mean_cos=("score", "mean"), mrr=("rr", "mean"),
             recall_at_1=("r1", "mean"), recall_at_10=("r10", "mean"),
             recall_at_100=("r100", "mean"), median_rank=("rank", "median"))
        .reset_index()
    )
    summary_path = output_dir / "curve_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Per-(model, query language, depth): monolingual (query lang == anchor) vs
    # cross-lingual decay — only meaningful with multi-language (strategy=all) queries.
    lang_summary = (
        df.groupby(["model", "query_language", "depth"])
        .agg(n=("query_id", "nunique"), mean_cos=("score", "mean"), mrr=("rr", "mean"),
             recall_at_10=("r10", "mean"), recall_at_100=("r100", "mean"))
        .reset_index()
    )
    lang_summary.to_csv(output_dir / "curve_summary_by_language.csv", index=False)

    mode_breakdown = (
        df[df["depth"] >= 1].groupby(["model", "mode_added"])
        .agg(n=("query_id", "size"), mean_cos_drop=("cos_drop_from_prev", "mean"))
        .reset_index()
    )
    mode_breakdown.to_csv(output_dir / "mode_breakdown.csv", index=False)

    _plot_curves(summary, output_dir / "experimental_plots" / "decay_curve.png")
    _write_findings(df, summary, mode_breakdown, len(hay_ids), output_dir / "key_findings" / "summary.md")

    print(f"\nWrote summary -> {summary_path}")
    print(summary.to_string(index=False))
    return summary_path


def _plot_curves(summary, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [("recall_at_10", "Recall@10"), ("mrr", "MRR"), ("mean_cos", "Mean cosine(query, doc)")]
    for ax, (col, title) in zip(axes, panels):
        for model, g in summary.groupby("model"):
            g = g.sort_values("depth")
            ax.plot(g["depth"], g[col], marker="o", label=model.split("/")[-1])
        ax.set_xlabel("# replacements (code-switch depth)")
        ax.set_ylabel(title)
        ax.set_title(title + " vs depth")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Progressive code-switching: retrieval decay vs dose", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Wrote plot -> {path}")


def _write_findings(df, summary, mode_breakdown, n_haystack: int, path: Path) -> None:
    lines = ["# Progressive code-switching — retrieval decay\n"]
    n_bases = df["base_id"].nunique()
    max_depth = int(df["depth"].max())
    lines.append(
        f"- **{n_bases}** base documents, ladder depth 0..{max_depth}, "
        f"haystack = **{n_haystack}** docs (base publications removed).\n"
    )
    for model, g in summary.groupby("model"):
        g = g.sort_values("depth")
        r10_0 = g[g.depth == 0]["recall_at_10"].iloc[0]
        r10_n = g[g.depth == max_depth]["recall_at_10"].iloc[0]
        cos_0 = g[g.depth == 0]["mean_cos"].iloc[0]
        cos_n = g[g.depth == max_depth]["mean_cos"].iloc[0]
        lines.append(
            f"- `{model}`: recall@10 {r10_0:.2f} → {r10_n:.2f} "
            f"(Δ={r10_0 - r10_n:+.2f}); mean cosine {cos_0:.3f} → {cos_n:.3f}.\n"
        )
    lines.append("\n## Per-step cosine drop by swap mode\n")
    for model, g in mode_breakdown.groupby("model"):
        parts = ", ".join(f"{r.mode_added}={r.mean_cos_drop:.4f}" for r in g.itertuples())
        lines.append(f"- `{model}`: {parts}\n")
    path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote findings -> {path}")
