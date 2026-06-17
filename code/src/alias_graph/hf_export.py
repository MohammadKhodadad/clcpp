"""
Publish the Alias-Graph Retrieval benchmark to the Hugging Face Hub.

Mirrors the MTEB structure of anonymous/multi-lingual-qac-chem-patents
(`corpus` / `queries` / `qrels` / `qac`) and adds the task-specific pieces:
- queries carry the concept identity + the multilingual **name_set** (alias query),
  plus the `source_publication` they were actually generated from,
- `qrels` mark gold docs (score 1) AND the chemically-similar look-alikes (score 0),
- a `source_qrels` config pins each query to the exact publication it was generated
  from and that publication's translations (the gold document, in every language),
  as distinct from `qrels` which marks every document about the concept,
- a `hard_negatives` config names each look-alike's neighbor concept + relation
  (so one can later measure how often a confusable wrong compound outranks the gold),
- a `concepts` config carries the full per-concept alias-graph record.

Everything is scoped to the concepts that actually have a generated query, so the
published dataset is self-consistent. ``dry_run=True`` writes the parquet locally
instead of uploading.
"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset

from src.alias_graph.builder import _read_corpus
from src.multi_lingual_qac.export.hf_upload import DATASET_CARD_ATTRIBUTION

_CONFIGS = ["corpus", "queries", "qrels", "source_qrels", "hard_negatives", "qac", "concepts"]

README_YAML = "---\nconfigs:\n" + "".join(
    f"- config_name: {c}\n  data_files:\n  - split: train\n    path: data/{c}/*.parquet\n"
    for c in _CONFIGS
) + "---\n"


def _load_qac(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _build_configs(
    alias_json: Path, qac_csv: Path, corpus_csv: Path,
    neighbor_names: Optional[Dict[str, str]] = None,
) -> Dict[str, List[dict]]:
    neighbor_names = neighbor_names or {}
    with Path(alias_json).open(encoding="utf-8") as fh:
        concepts = json.load(fh)["concepts"]
    concept_by_cid = {c["chebi_id"]: c for c in concepts}

    qac_rows = _load_qac(qac_csv)
    queried_cids = {r["chebi_id"] for r in qac_rows if r.get("chebi_id")}

    # Corpus docs grouped by publication, limited to pubs the queried concepts reference.
    docs_by_pub: Dict[str, List[dict]] = {}
    for r in _read_corpus(corpus_csv):
        docs_by_pub.setdefault(r["publication_number"], []).append(r)
    needed_pubs: set[str] = set()
    for cid in queried_cids:
        c = concept_by_cid.get(cid)
        if not c:
            continue
        needed_pubs.update(c.get("gold", []))
        needed_pubs.update(hn["pub"] for hn in c.get("hard_negatives", []))

    corpus_data = [
        {
            "_id": r["id"],
            "title": r.get("title", ""),
            "text": r.get("context") or r.get("abstract", ""),
            "corpus_language": str(r.get("language", "")).strip(),
            "publication_number": pub,
        }
        for pub in sorted(needed_pubs)
        for r in docs_by_pub.get(pub, [])
    ]

    queries_data: List[dict] = []
    qrels_data: List[dict] = []
    source_qrels_data: List[dict] = []
    hard_neg_data: List[dict] = []
    qac_data: List[dict] = []
    seen_qids: set[str] = set()

    for i, r in enumerate(qac_rows):
        cid = r.get("chebi_id", "")
        c = concept_by_cid.get(cid, {})
        lang = r.get("question_language", "")
        # The single publication this query was generated from (the gold document).
        src_pub = (r.get("publication_number") or "").strip()
        qid = f"{cid.replace(':', '_')}__{lang}"
        if qid in seen_qids:
            qid = f"{qid}__{i}"
        seen_qids.add(qid)
        name_set_json = json.dumps(c.get("name_set", {}), ensure_ascii=False)

        queries_data.append({
            "_id": qid, "text": r.get("question", ""), "query_language": lang,
            "chebi_id": cid, "concept_name": r.get("concept_name", ""),
            "answer": r.get("answer", ""), "answer_language": r.get("answer_language", ""),
            "question_type": r.get("question_type", ""),
            "source_publication": src_pub,
            "total_score": int(r.get("total_score") or 0),
            "faith_overall": int(r.get("faith_overall") or 0),
            "qual_overall": int(r.get("qual_overall") or 0),
            "name_set_json": name_set_json,
            "codes_json": json.dumps(c.get("codes", []), ensure_ascii=False),
        })

        # Exact (query -> source publication) mapping: the gold document the query
        # was written from, plus its translations (one row per language variant).
        for doc in docs_by_pub.get(src_pub, []):
            source_qrels_data.append({
                "query-id": qid, "corpus-id": doc["id"],
                "publication_number": src_pub,
                "corpus_language": str(doc.get("language", "")).strip(),
            })

        gold_pubs = c.get("gold", [])
        for pub in gold_pubs:
            for doc in docs_by_pub.get(pub, []):
                qrels_data.append({"query-id": qid, "corpus-id": doc["id"], "score": 1.0})
        for hn in c.get("hard_negatives", []):
            pub, neighbor, relation = hn["pub"], hn.get("neighbor", ""), hn.get("relation", "")
            neighbor_name = neighbor_names.get(neighbor) or concept_by_cid.get(neighbor, {}).get("name", "")
            for doc in docs_by_pub.get(pub, []):
                qrels_data.append({"query-id": qid, "corpus-id": doc["id"], "score": 0.0})
                hard_neg_data.append({
                    "query-id": qid, "corpus-id": doc["id"], "publication_number": pub,
                    "neighbor_chebi_id": neighbor, "neighbor_name": neighbor_name,
                    "relation": relation,
                })

        qac_data.append({
            "query_id": qid, "chebi_id": cid, "concept_name": r.get("concept_name", ""),
            "question": r.get("question", ""), "answer": r.get("answer", ""),
            "question_type": r.get("question_type", ""), "query_language": lang,
            "source_publication": src_pub,
            "total_score": int(r.get("total_score") or 0),
            "gold_pubs_json": json.dumps(gold_pubs, ensure_ascii=False),
            "n_gold": c.get("n_gold", len(gold_pubs)),
            "n_hard_neg": c.get("n_hard_neg", len(c.get("hard_negatives", []))),
            "name_set_json": name_set_json,
        })

    concepts_data = [
        {
            "chebi_id": c["chebi_id"], "name": c["name"],
            "name_set_json": json.dumps(c.get("name_set", {}), ensure_ascii=False),
            "codes_json": json.dumps(c.get("codes", []), ensure_ascii=False),
            "gold_json": json.dumps(c.get("gold", []), ensure_ascii=False),
            "hard_negatives_json": json.dumps(c.get("hard_negatives", []), ensure_ascii=False),
            "gold_langs": "|".join(c.get("gold_langs", [])),
            "n_gold": c.get("n_gold", 0), "n_hard_neg": c.get("n_hard_neg", 0),
        }
        for c in concepts if c["chebi_id"] in queried_cids
    ]

    return {
        "corpus": corpus_data, "queries": queries_data, "qrels": qrels_data,
        "source_qrels": source_qrels_data, "hard_negatives": hard_neg_data,
        "qac": qac_data, "concepts": concepts_data,
    }


def _readme(repo_id: str) -> bytes:
    body = (
        README_YAML
        + "# Multi-lingual chemical QAC — Alias-Graph Retrieval benchmark\n\n"
        "Given a chemistry concept (named in several languages), can a retriever find the documents "
        "that genuinely talk about it, across languages, **without being fooled by documents about "
        "chemically similar look-alike concepts**?\n\n"
        "Configs: `corpus` (gold + hard-negative documents), `queries` (technical questions about each "
        "concept, plus the concept's multilingual `name_set` and the `source_publication` each query "
        "was generated from), `qrels` (every document about the concept: gold docs = score 1, "
        "designated look-alike docs = score 0), `source_qrels` (the exact publication each query was "
        "generated from and its translations — the gold document in every language), `hard_negatives` "
        "(each look-alike's neighbor concept and `relation`), `qac` (full triplets), and `concepts` "
        "(the per-concept alias-graph record). Each config has a `train` split.\n"
        + DATASET_CARD_ATTRIBUTION
    )
    return body.encode("utf-8")


def _neighbor_name_map(chebi_cache_dir: Optional[Path]) -> Dict[str, str]:
    """Best-effort chebi_id -> name map (for naming hard-negative look-alikes)."""
    if not chebi_cache_dir:
        return {}
    try:
        from src.alias_graph.chebi import load_chebi_graph
        graph = load_chebi_graph(Path(chebi_cache_dir), "full")
        return {nid: data.get("name", "") for nid, data in graph.nodes(data=True)}
    except Exception as exc:  # cache missing / load error -> names stay empty
        print(f"  (neighbor names unavailable: {exc})")
        return {}


def push_alias_graph_to_hub(
    alias_json: Path,
    qac_csv: Path,
    corpus_csv: Path,
    repo_id: str,
    *,
    token: Optional[str] = None,
    private: bool = False,
    dry_run: bool = False,
    chebi_cache_dir: Optional[Path] = None,
    only_configs: Optional[List[str]] = None,
) -> str:
    """Build the Alias-Graph dataset configs and push them (or, if ``dry_run``,
    write parquet under data/alias_graph/hf_export/ and skip the upload).

    ``only_configs`` restricts the push/write to the named configs (e.g.
    ``["queries", "qac", "source_qrels"]`` to patch an existing dataset in place
    without re-uploading the large, unchanged corpus); the README is always
    refreshed so the config listing stays accurate.
    """
    configs = _build_configs(
        alias_json, qac_csv, corpus_csv, neighbor_names=_neighbor_name_map(chebi_cache_dir)
    )
    if only_configs is not None:
        unknown = set(only_configs) - set(configs)
        if unknown:
            raise ValueError(f"Unknown config(s) for only_configs: {sorted(unknown)}")
        configs = {name: configs[name] for name in only_configs}
    datasets = {name: Dataset.from_list(rows) for name, rows in configs.items()}
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} rows | columns: {ds.column_names}")

    if dry_run:
        out_dir = Path(qac_csv).resolve().parents[1] / "hf_export"
        for name, ds in datasets.items():
            (out_dir / name).mkdir(parents=True, exist_ok=True)
            ds.to_parquet(str(out_dir / name / f"{name}.parquet"))
        (out_dir / "README.md").write_bytes(_readme(repo_id))
        print(f"Dry run: wrote {len(datasets)} configs -> {out_dir}")
        return str(out_dir)

    from huggingface_hub import HfApi

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN in .env for Hugging Face upload.")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    for name, ds in datasets.items():
        ds.push_to_hub(
            repo_id, config_name=name, split="train",
            data_dir=f"data/{name}", token=token, private=private,
        )
    api.upload_file(
        path_or_fileobj=io.BytesIO(_readme(repo_id)),
        path_in_repo="README.md", repo_id=repo_id, repo_type="dataset",
    )
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"Pushed to {url}")
    return url
