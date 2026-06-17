"""
Alias-Graph Retrieval benchmark builder.

For each ChEBI concept that genuinely appears in the corpus (gold documents), we
surround it with hard negatives -- documents that mention a *taxonomic neighbor*
of the concept (chemically similar, but a different concept) and do not mention
the concept itself. A concept is kept only if it has at least ``min_gold`` gold
documents and at least ``min_neg`` hard-negative documents. Each kept concept is
written to its own CSV (gold + hard-negative rows, role-labeled), and a manifest
records the concept's multilingual name set (the retrieval query) plus counts.

Pipeline: read corpus -> load ChEBI graph -> (KG-only scan to find concepts that
appear) -> fetch Wikipedia names for those concepts -> rebuild index + rescan ->
assemble gold/hard-negatives -> write per-concept CSVs + manifest.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

from src.alias_graph.chebi import load_chebi_graph, taxonomic_neighbors
from src.alias_graph.wikidata_names import (
    DEFAULT_LANGS,
    fetch_wikipedia_names,
)
from src.alias_graph.matching import (
    build_name_index,
    is_code_like_name,
    prune_names,
    scan_corpus,
)

# Root of the ChEBI structural (actual-molecule) subtree. Restricting main
# concepts to its descendants keeps real chemical entities and drops role /
# group / atom / application classes whose names are ordinary words.
MOLECULAR_ENTITY = "CHEBI:23367"

# Patent description fields can exceed the default csv field-size limit.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

CORPUS_FIELDS: Tuple[str, ...] = (
    "id", "language", "title", "abstract", "description", "first_claim",
    "context", "publication_number", "country_code", "publication_date",
    "source", "ipc_codes",
)
EXTRA_FIELDS: Tuple[str, ...] = (
    "role", "concept_chebi_id", "concept_name", "matched_chebi_id", "relation",
)
OUTPUT_FIELDS: Tuple[str, ...] = CORPUS_FIELDS + EXTRA_FIELDS


def _slug(name: str, maxlen: int = 60) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_name).strip("-").lower()
    return s[:maxlen] or "concept"


def _read_corpus(path: Path) -> List[dict]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _dedupe_keep_first(names: List[str]) -> List[str]:
    """Case-insensitive dedupe, preserving the first (canonical) casing."""
    out: List[str] = []
    seen: Set[str] = set()
    for n in names:
        key = n.casefold()
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


def _concept_name_set(
    graph: nx.DiGraph,
    cid: str,
    wiki_names: Dict[str, Dict[str, str]],
) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    """
    Return (name_set, codes, brand_names) for a concept. The multilingual name_set
    (the query/answer side) holds only real names; registry/regulatory codes
    (E-numbers, refrigerant numbers, company/CAS codes) are split into ``codes``.
    ChEBI ``BRAND:NAME`` synonyms are already excluded from the graph's ``synonyms``
    (kept out of matching) and surfaced here as ``brand_names`` for provenance.
    The ChEBI primary name always stays in name_set; only synonyms are classified.
    """
    data = graph.nodes[cid]
    name_set: Dict[str, List[str]] = {}
    chebi_names: List[str] = []
    codes: List[str] = []
    if data.get("name"):
        chebi_names.append(data["name"])
    for syn in data.get("synonyms", ()):
        (codes if is_code_like_name(syn) else chebi_names).append(syn)
    if chebi_names:
        name_set["chebi"] = _dedupe_keep_first(chebi_names)
    for lang, title in wiki_names.get(cid, {}).items():
        name_set.setdefault(lang, []).append(title)
    return name_set, _dedupe_keep_first(codes), _dedupe_keep_first(list(data.get("brand_names", [])))


def _neighbor_relations(graph: nx.DiGraph, cid: str) -> Dict[str, str]:
    """Map each taxonomic neighbor id -> relation, parent/child before sibling."""
    nb = taxonomic_neighbors(graph, cid)
    rel: Dict[str, str] = {}
    for nid in nb["sibling"]:
        rel[nid] = "sibling"
    for nid in nb["child"]:
        rel[nid] = "child"
    for nid in nb["parent"]:
        rel[nid] = "parent"
    return rel


def build_alias_graph(
    corpus_csv: Path,
    output_dir: Path,
    chebi_cache_dir: Path,
    *,
    variant: str = "full",
    langs: Sequence[str] = DEFAULT_LANGS,
    use_wikipedia: bool = True,
    wiki_cache_path: Optional[Path] = None,
    min_gold: int = 2,
    min_neg: int = 3,
    max_concepts: Optional[int] = None,
    match_field: str = "context",
    min_name_len: int = 4,
    max_concepts_per_name: int = 3,
    max_df_ratio: float = 0.02,
    molecular_only: bool = True,
    leaf_only: bool = True,
) -> dict:
    """Build the benchmark; returns a summary dict."""
    corpus_csv = Path(corpus_csv)
    output_dir = Path(output_dir)
    chebi_cache_dir = Path(chebi_cache_dir)
    wiki_cache_path = Path(wiki_cache_path) if wiki_cache_path else chebi_cache_dir / "wiki_names_cache.json"

    print(f"Reading corpus: {corpus_csv}")
    rows = _read_corpus(corpus_csv)
    doc_by_id = {r["id"]: r for r in rows}
    print(f"  {len(rows)} documents")

    graph = load_chebi_graph(chebi_cache_dir, variant)

    mol_entity_set = None
    if molecular_only:
        if MOLECULAR_ENTITY in graph:
            mol_entity_set = nx.ancestors(graph, MOLECULAR_ENTITY) | {MOLECULAR_ENTITY}
            print(f"  restricting to {len(mol_entity_set)} molecular-entity concepts")
        else:
            print(f"  warning: {MOLECULAR_ENTITY} absent from {variant} graph; no molecular filter")

    # Pass 1: KG-only scan to discover which concepts occur (so we only ask
    # Wikidata about those) and to measure per-name document frequency.
    print("Scanning corpus for ChEBI names (KG only) ...")
    kg_index = build_name_index(
        graph, {}, min_len=min_name_len, max_concepts_per_name=max_concepts_per_name
    )
    print(f"  name index: {kg_index.n_names()} names")
    concept_to_docs, _, name_doc_freq = scan_corpus(rows, kg_index, field=match_field)
    print(f"  concepts found in corpus: {len(concept_to_docs)}")

    # Names that behave like corpus stopwords (common words masquerading as
    # aliases, e.g. "para", "groupe") are pruned so only specific names match.
    df_ceiling = max_df_ratio * len(rows)
    stop_grams = {g for g, n in name_doc_freq.items() if n > df_ceiling}
    if stop_grams:
        examples = sorted(stop_grams, key=lambda g: -name_doc_freq[g])[:8]
        print(f"  pruning {len(stop_grams)} stopword names (df > {df_ceiling:.0f}); e.g. {examples}")

    wiki_names: Dict[str, Dict[str, str]] = {}
    if use_wikipedia and concept_to_docs:
        wiki_names = fetch_wikipedia_names(
            list(concept_to_docs.keys()), langs=langs, cache_path=wiki_cache_path
        )

    # Pass 2: final index (with Wikipedia names if enabled), stopwords removed.
    index = (
        build_name_index(
            graph, wiki_names, min_len=min_name_len, max_concepts_per_name=max_concepts_per_name
        )
        if wiki_names
        else kg_index
    )
    prune_names(index, stop_grams)
    print("Re-scanning corpus (Wikipedia names folded in, stopwords pruned) ...")
    concept_to_docs, _, _ = scan_corpus(rows, index, field=match_field)
    print(f"  concepts found in corpus: {len(concept_to_docs)}")

    # Documents are stored by publication_number (i.e. the doc id with the
    # `_<lang>` suffix stripped): the per-language versions of one patent are the
    # same gold/negative item, and their text lives once in the corpus CSV. This
    # avoids the 16x text duplication of writing a CSV per concept.
    concept_to_pubs: Dict[str, Set[str]] = {}
    for cid, docs in concept_to_docs.items():
        concept_to_pubs[cid] = {doc_by_id[d]["publication_number"] for d in docs}

    # Candidate main concepts: specific molecular entities (leaves of the is_a
    # graph -- not broad classes) with enough gold publications, most-attested first.
    def _is_candidate(cid: str) -> bool:
        if len(concept_to_pubs[cid]) < min_gold:
            return False
        if mol_entity_set is not None and cid not in mol_entity_set:
            return False
        if leaf_only and graph.in_degree(cid) > 0:
            return False
        return True

    candidates = sorted(
        (cid for cid in concept_to_pubs if _is_candidate(cid)),
        key=lambda c: len(concept_to_pubs[c]),
        reverse=True,
    )
    kind = "leaf molecular" if leaf_only else "molecular"
    print(f"Candidate concepts ({kind}, >= {min_gold} gold pubs): {len(candidates)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    concepts: List[dict] = []

    for cid in candidates:
        if max_concepts is not None and len(concepts) >= max_concepts:
            break
        gold_pubs = concept_to_pubs[cid]
        relations = _neighbor_relations(graph, cid)

        # Hard negatives: publications that mention a neighbor but NOT the concept.
        hard_neg: Dict[str, Tuple[str, str]] = {}  # pub -> (neighbor_id, relation)
        neighbors_in_corpus: Set[str] = set()
        for nid, rel in relations.items():
            nid_pubs = concept_to_pubs.get(nid)
            if not nid_pubs:
                continue
            neighbors_in_corpus.add(nid)
            for pub in nid_pubs - gold_pubs:
                hard_neg.setdefault(pub, (nid, rel))

        if len(hard_neg) < min_neg:
            continue

        concept_name = graph.nodes[cid].get("name", cid)
        name_set, codes, brand_names = _concept_name_set(graph, cid, wiki_names)
        gold_langs = sorted({doc_by_id[d]["language"] for d in concept_to_docs[cid]})

        concepts.append({
            "chebi_id": cid,
            "name": concept_name,
            "name_set": name_set,
            "codes": codes,
            "brand_names": brand_names,
            "query_names": sorted({n for names in name_set.values() for n in names}),
            "gold": sorted(gold_pubs),
            "hard_negatives": [
                {"pub": pub, "neighbor": nid, "relation": rel}
                for pub, (nid, rel) in sorted(hard_neg.items())
            ],
            "n_gold": len(gold_pubs),
            "n_hard_neg": len(hard_neg),
            "gold_langs": gold_langs,
            "n_neighbors_in_corpus": len(neighbors_in_corpus),
        })

    _write_outputs(output_dir, corpus_csv, concepts)

    summary = {
        "corpus": str(corpus_csv),
        "documents": len(rows),
        "variant": variant,
        "use_wikipedia": use_wikipedia,
        "concepts_in_corpus": len(concept_to_docs),
        "candidates": len(candidates),
        "concepts_written": len(concepts),
        "output_dir": str(output_dir),
    }
    print(
        f"Wrote {len(concepts)} concepts -> {output_dir / 'alias_graph.json'}\n"
        f"  summary: {output_dir / 'manifest.csv'}"
    )
    return summary


def _write_outputs(output_dir: Path, corpus_csv: Path, concepts: List[dict]) -> None:
    """One JSON of id-only benchmark data + a tiny CSV summary (no document text)."""
    json_path = output_dir / "alias_graph.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "corpus": str(corpus_csv),
                "id_field": "publication_number",
                "n_concepts": len(concepts),
                "concepts": concepts,
            },
            fh, ensure_ascii=False, indent=2,
        )

    csv_path = output_dir / "manifest.csv"
    cols = ["chebi_id", "name", "n_gold", "n_hard_neg", "gold_langs", "n_neighbors_in_corpus"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for entry in concepts:
            row = {k: entry[k] for k in cols if k != "gold_langs"}
            row["gold_langs"] = "|".join(entry["gold_langs"])
            writer.writerow(row)


def export_concept(
    json_path: Path,
    corpus_csv: Path,
    chebi_id: str,
    output_csv: Optional[Path] = None,
) -> Path:
    """
    Materialize one concept's gold + hard-negative documents (all language
    versions, joined from the corpus) into a CSV for inspection -- the on-demand
    replacement for storing a CSV per concept.
    """
    json_path = Path(json_path)
    cid = chebi_id if chebi_id.upper().startswith("CHEBI:") else f"CHEBI:{chebi_id}"
    with json_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    entry = next((c for c in data["concepts"] if c["chebi_id"].upper() == cid.upper()), None)
    if entry is None:
        raise ValueError(f"{cid} not found in {json_path}")

    by_pub: Dict[str, List[dict]] = defaultdict(list)
    for r in _read_corpus(corpus_csv):
        by_pub[r["publication_number"]].append(r)

    cname = entry["name"]
    out_rows: List[dict] = []
    for pub in entry["gold"]:
        for r in by_pub.get(pub, []):
            out_rows.append({
                **{k: r.get(k, "") for k in CORPUS_FIELDS},
                "role": "gold", "concept_chebi_id": cid, "concept_name": cname,
                "matched_chebi_id": cid, "relation": "self",
            })
    for neg in entry["hard_negatives"]:
        for r in by_pub.get(neg["pub"], []):
            out_rows.append({
                **{k: r.get(k, "") for k in CORPUS_FIELDS},
                "role": "hard_negative", "concept_chebi_id": cid, "concept_name": cname,
                "matched_chebi_id": neg["neighbor"], "relation": neg["relation"],
            })

    output_csv = Path(output_csv) if output_csv else json_path.parent / f"{cid.replace(':', '_')}__{_slug(cname)}.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)
    n_gold = sum(1 for x in out_rows if x["role"] == "gold")
    print(f"Exported {cid} ({cname}): {n_gold} gold + {len(out_rows) - n_gold} hard-neg doc rows -> {output_csv}")
    return output_csv
