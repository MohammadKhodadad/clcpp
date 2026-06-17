"""
Publish the progressive code-switching benchmark to Hugging Face in MTEB retrieval
format, so embedding models can be evaluated against it.

Three configs (each a `train` split):
  * ``corpus``  — every ladder variant document (``base__r{k}``). Columns:
      ``_id, title, text, corpus_language, base_id, depth, mode_added,
       source_publication_number``. ``text`` is the (code-switched) context|abstract,
      matching the shared-corpus haystack schema so a doc is encoded identically
      whichever corpus it is loaded from.
  * ``queries`` — one fixed question per base document (``_id == base_id``). Columns:
      ``_id, text, query_language, concept_chebi_id, term_used``.
  * ``qrels``   — one row per (query, ladder variant), ``score = 1.0``, carrying the
      ``depth`` so the eval can read the dose for each gold document.

The retrieval *haystack* at eval time is the shared corpus
(``--mteb-corpus-repo``) PLUS these variant docs; this dataset only carries the
variant docs + queries + qrels. ``dry_run`` writes the parquet locally in the
layout the loaders understand (``<dir>/<config>/<config>.parquet``) so the eval can
read it offline.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import List, Optional

from datasets import Dataset

from src.alias_graph.builder import _read_corpus

_CONFIGS = ("corpus", "queries", "qrels")


def _readme(repo_id: str) -> bytes:
    yaml = "---\nconfigs:\n" + "".join(
        f"- config_name: {c}\n  data_files:\n  - split: train\n    path: data/{c}/*.parquet\n"
        for c in _CONFIGS
    ) + "---\n\n"
    body = (
        f"# {repo_id}\n\n"
        "**Progressive code-switching** retrieval-decay benchmark. Each base patent yields a "
        "cumulative ladder of documents (`base__r0` clean → `base__rN`), where each step swaps one "
        "more chemistry term into another language / spelling / ChEBI form. One fixed question per "
        "base (about the step-1 term) is reused for every depth; the `qrels` carry the `depth` so "
        "you can measure how retrieval decays as more terms are code-switched.\n\n"
        "Configs: `corpus` (ladder variant documents, with `depth` + `mode_added`), `queries` "
        "(one per base), `qrels` (query→variant, score 1, with `depth`). The retrieval haystack is "
        "this `corpus` PLUS a shared patent corpus passed at eval time.\n"
    )
    return (yaml + body).encode("utf-8")


def _build_configs(corpus_csv: Path, qac_csv: Path):
    corpus_rows = _read_corpus(corpus_csv)
    qac_rows = _read_corpus(qac_csv)

    corpus_data: List[dict] = []
    for r in corpus_rows:
        depth = int(r["n_replacements"])
        steps = json.loads(r.get("replacements_json") or "[]")
        mode_added = steps[-1]["mode"] if depth >= 1 and steps else ""
        corpus_data.append({
            "_id": r["id"],
            "title": r.get("title", ""),
            "text": r.get("context") or r.get("abstract") or "",
            "corpus_language": str(r.get("anchor_language", "")).strip(),
            "base_id": r["base_id"],
            "depth": depth,
            "mode_added": mode_added,
            "source_publication_number": str(r.get("source_publication_number", "")).strip(),
        })

    # One query per (base, query language); qac has one row per (query, depth).
    queries_data: List[dict] = []
    seen = set()
    for r in qac_rows:
        qid = r.get("query_id") or r["base_id"]
        if qid in seen:
            continue
        seen.add(qid)
        queries_data.append({
            "_id": qid,
            "text": r.get("question", ""),
            "query_language": str(r.get("query_language", "")).strip(),
            "base_id": r["base_id"],
            "strategy": str(r.get("strategy", "")).strip(),
            "concept_chebi_id": str(r.get("concept_chebi_id", "")).strip(),
            "term_used": r.get("term_used", ""),
        })

    # qrels: each query is gold for all of its base's ladder variants (carry depth).
    qrels_data: List[dict] = [
        {"query-id": (r.get("query_id") or r["base_id"]), "corpus-id": r["gold_id"],
         "score": 1.0, "depth": int(r["n_replacements"])}
        for r in qac_rows
    ]
    return {
        "corpus": Dataset.from_list(corpus_data),
        "queries": Dataset.from_list(queries_data),
        "qrels": Dataset.from_list(qrels_data),
    }


def push_progressive_to_hub(
    corpus_csv: Path,
    qac_csv: Path,
    repo_id: str,
    *,
    token: Optional[str] = None,
    private: bool = False,
    dry_run: bool = False,
) -> str:
    """Publish the progressive benchmark (corpus/queries/qrels). ``dry_run`` writes
    parquet locally instead of uploading. Returns the URL or local path."""
    configs = _build_configs(Path(corpus_csv), Path(qac_csv))
    for name, ds in configs.items():
        print(f"  {name}: {len(ds)} rows | columns: {ds.column_names}")
    readme = _readme(repo_id)

    if dry_run:
        out_dir = Path(corpus_csv).parent / "hf_export"
        for name, ds in configs.items():
            (out_dir / name).mkdir(parents=True, exist_ok=True)
            ds.to_parquet(str(out_dir / name / f"{name}.parquet"))
        (out_dir / "README.md").write_bytes(readme)
        print(f"Dry run: wrote progressive dataset -> {out_dir}")
        return str(out_dir)

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN in .env for Hugging Face upload (or use --hf-dry-run).")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    for name, ds in configs.items():
        ds.push_to_hub(repo_id, config_name=name, split="train",
                       data_dir=f"data/{name}", token=token, private=private)
    api.upload_file(path_or_fileobj=io.BytesIO(readme), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="dataset")
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"Pushed progressive dataset to {url}")
    return url
