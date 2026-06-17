"""
Persist the embedding model's retrieval results inside a benchmark run, in a
tidy form ready for *future* metrics (e.g. CLIR@k — cross-lingual recall@k).

The standard run already saves raw per-query rankings as MTEB prediction JSON
(`<run>/predictions/<model>/*.json`), but that is nested `{qid: {doc: score}}`
with no language or relevance info. This reads those predictions (no re-encoding)
and the dataset's queries/corpus/qrels, then writes one flat table
`<run>/retrieval_results/scored_rankings.parquet` with, per ranked (query, doc):

  model, query_id, query_language, chebi_id, rank, corpus_id, corpus_language,
  score, relevance ("gold" | "hard_negative" | "")

From this, CLIR@k is a groupby: recall@k over the gold docs whose
corpus_language != query_language. Works for any retrieval dataset (the
hard_negative label is simply absent when the dataset has no score-0 qrels).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from datasets import Dataset

from src.alias_graph.confusion_analysis import _load_config
from src.multi_lingual_qac.mteb.evaluation import _infer_language, _normalize_dataset_variant
from src.multi_lingual_qac.mteb.question_analysis import _discover_models, _load_predictions

_README = """# Retrieval results (embedding-model rankings)

`scored_rankings.parquet` — one row per ranked (query, document) pair, top-K per query per model.

Columns:
- `model` — embedding model name
- `query_id`, `query_language`, `chebi_id` — the query and its concept
- `rank` (1 = top), `corpus_id`, `corpus_language`, `score` (cosine, higher = better)
- `relevance` — `gold` (right document), `hard_negative` (chemically-similar look-alike), or `` (not judged)

Compute @k metrics by filtering `rank <= k`. CLIR@k (cross-lingual recall@k) = of the gold
documents whose `corpus_language != query_language`, the fraction with `rank <= k`.
"""


def save_retrieval_results(
    predictions_dir: Path,
    output_dir: Path,
    *,
    dataset_repo: str,
    dataset_variant: str = "multilingual",
    model_names: Optional[Sequence[str]] = None,
    revision: str = "main",
    top_k: int = 1000,
) -> Optional[Path]:
    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)
    variant = _normalize_dataset_variant(dataset_variant)

    models = _discover_models(predictions_dir, list(model_names) if model_names else None)
    if not models:
        print(f"[retrieval results skipped] no predictions under {predictions_dir}")
        return None

    queries = _load_config(dataset_repo, "queries", variant, revision)
    corpus = _load_config(dataset_repo, "corpus", variant, revision)
    qrels = _load_config(dataset_repo, "qrels", variant, revision)

    qid_col = "_id" if "_id" in queries.column_names else "query_id"
    q_lang = {str(r[qid_col]): str(r.get("query_language", "")).strip().lower() for r in queries}
    q_concept = {str(r[qid_col]): str(r.get("chebi_id", "")) for r in queries}
    cid_col = "_id" if "_id" in corpus.column_names else "corpus_id"
    c_lang = {
        str(r[cid_col]): (str(r.get("corpus_language") or r.get("language") or "").strip().lower()
                          or _infer_language(str(r[cid_col])))
        for r in corpus
    }
    relevance = {
        (str(r["query-id"]), str(r["corpus-id"])): ("gold" if float(r["score"]) > 0 else "hard_negative")
        for r in qrels
    }

    rows: List[dict] = []
    for label, slug in models:
        preds = _load_predictions(predictions_dir / slug)
        if preds is None:
            continue
        for qid, doc_scores in preds.items():
            ranked = sorted(doc_scores.items(), key=lambda kv: -kv[1])[:top_k]
            ql = q_lang.get(qid, "")
            cid = q_concept.get(qid, "")
            for rank, (doc, score) in enumerate(ranked, start=1):
                rows.append({
                    "model": label, "query_id": qid, "query_language": ql, "chebi_id": cid,
                    # shared-corpus distractors may be absent from the benchmark's corpus
                    # config; ids encode the language (`_en`/`_de`/…), so infer as fallback.
                    "rank": rank, "corpus_id": doc,
                    "corpus_language": c_lang.get(doc) or _infer_language(doc),
                    "score": round(float(score), 6), "relevance": relevance.get((qid, doc), ""),
                })

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "scored_rankings.parquet"
    Dataset.from_list(rows).to_parquet(str(out))
    (output_dir / "README.md").write_text(_README, encoding="utf-8")
    print(f"Saved {len(rows)} ranked (query, doc) rows for {len(models)} model(s) -> {out}")
    return out
