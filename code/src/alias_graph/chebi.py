"""
ChEBI ontology loader for the Alias-Graph Retrieval benchmark.

Downloads a ChEBI OBO release from the EBI FTP mirror, stream-parses it
stanza-by-stanza (peak working memory is one term at a time, honoring the
limited-disk / "download -> process -> discard" preference), and builds a
``networkx.DiGraph`` whose ``is_a`` edges point child -> parent. Each node
carries the primary ``name`` plus the set of ``synonyms`` and an optional
``wiki_en`` title scraped from a ``Wikipedia:`` xref.

Variants (https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/):
  * ``full`` -> ``chebi.obo.gz``       (~43 MB; the ONLY variant with synonyms)
  * ``core`` -> ``chebi_core.obo.gz``  (~33 MB; names + is_a + relationships)
  * ``lite`` -> ``chebi_lite.obo.gz``  (~7 MB; names + is_a only)

Synonyms only exist in ``full``, so the benchmark defaults to ``full`` to get the
IUPAC / brand / INN name variants that actually appear in patent text. ``lite`` /
``core`` still load (primary names only) for a lighter, name-only run.

A compact parsed graph is cached as ``chebi_<variant>_graph.json.gz`` so repeat
runs skip both the download and the parse.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import networkx as nx
import requests
from tqdm import tqdm


FTP_BASE = "https://ftp.ebi.ac.uk/pub/databases/chebi/ontology"
USER_AGENT = "multi-lingual-qac/0.1 (+research)"

# Synonym scopes worth keeping for surface-name matching. BROAD/NARROW are
# dropped because they denote more/less specific concepts, not aliases.
_KEEP_SYNONYM_SCOPES = {"EXACT", "RELATED"}

_VARIANT_FILES = {
    "full": "chebi.obo.gz",
    "core": "chebi_core.obo.gz",
    "lite": "chebi_lite.obo.gz",
}

# `synonym: "text" SCOPE [TYPE] [xrefs]` — capture the quoted text, the scope, and
# the optional OBO synonym-type id (e.g. IUPAC:NAME, INN, BRAND:NAME).
_SYNONYM_RE = re.compile(r'^synonym:\s+"((?:[^"\\]|\\.)*)"\s+(\w+)(?:\s+([A-Za-z][\w:]*))?')


def _variant_filename(variant: str) -> str:
    try:
        return _VARIANT_FILES[variant]
    except KeyError as exc:
        raise ValueError(
            f"Unknown ChEBI variant {variant!r}; choose one of {sorted(_VARIANT_FILES)}"
        ) from exc


def _download_obo(variant: str, dest: Path) -> None:
    filename = _variant_filename(variant)
    url = f"{FTP_BASE}/{filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ChEBI {variant} ontology from {url} ...")
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": USER_AGENT}) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) or None
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=filename
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
        tmp.replace(dest)


def _unescape(text: str) -> str:
    return text.replace('\\"', '"').replace("\\\\", "\\")


def _parse_obo(obo_gz: Path) -> Tuple[nx.DiGraph, Dict[str, str]]:
    """Stream-parse an OBO .gz into a child->parent is_a DiGraph + alt_id map."""
    graph = nx.DiGraph()
    alt_to_primary: Dict[str, str] = {}
    is_a_edges: list[Tuple[str, str]] = []

    cur_id: Optional[str] = None
    cur_name: Optional[str] = None
    cur_syns: Set[str] = set()
    cur_brands: Set[str] = set()
    cur_parents: list[str] = []
    cur_alts: list[str] = []
    cur_wiki: Optional[str] = None
    cur_obsolete = False
    in_term = False

    def commit() -> None:
        if not in_term or cur_id is None or cur_obsolete:
            return
        graph.add_node(
            cur_id,
            name=cur_name or "",
            synonyms=sorted(cur_syns),
            brand_names=sorted(cur_brands),
            wiki_en=cur_wiki,
        )
        for alt in cur_alts:
            alt_to_primary[alt] = cur_id
        for parent in cur_parents:
            is_a_edges.append((cur_id, parent))

    with gzip.open(obo_gz, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("[") and line.endswith("]"):
                commit()
                in_term = line == "[Term]"
                cur_id = cur_name = cur_wiki = None
                cur_syns = set()
                cur_brands = set()
                cur_parents = []
                cur_alts = []
                cur_obsolete = False
                continue
            if not in_term:
                continue
            if line.startswith("id: "):
                cur_id = line[4:].strip()
            elif line.startswith("name: "):
                cur_name = line[6:].strip()
            elif line.startswith("alt_id: "):
                cur_alts.append(line[8:].strip())
            elif line.startswith("is_a: "):
                # `is_a: CHEBI:NNNN ! label`
                target = line[6:].split("!", 1)[0].strip()
                if target:
                    cur_parents.append(target)
            elif line.startswith("is_obsolete: true"):
                cur_obsolete = True
            elif line.startswith("synonym: "):
                m = _SYNONYM_RE.match(line)
                if m and m.group(2) in _KEEP_SYNONYM_SCOPES:
                    text = _unescape(m.group(1))
                    # Trade/product names (often common words like "Action", "Balance")
                    # are unreliable for matching — keep them out of `synonyms`.
                    if m.group(3) == "BRAND:NAME":
                        cur_brands.add(text)
                    else:
                        cur_syns.add(text)
            elif line.startswith("xref: Wikipedia:"):
                cur_wiki = line[len("xref: Wikipedia:"):].strip().replace("_", " ")
        commit()

    # Add is_a edges, resolving any alt_id references to their primary id and
    # skipping edges whose endpoints were obsolete (not present as nodes).
    for child, parent in is_a_edges:
        child = alt_to_primary.get(child, child)
        parent = alt_to_primary.get(parent, parent)
        if graph.has_node(child) and graph.has_node(parent):
            graph.add_edge(child, parent)

    return graph, alt_to_primary


def _cache_path(cache_dir: Path, variant: str) -> Path:
    return cache_dir / f"chebi_{variant}_graph.json.gz"


def _write_cache(path: Path, graph: nx.DiGraph, alt_to_primary: Dict[str, str]) -> None:
    payload = {
        "nodes": {
            n: {
                "name": d.get("name", ""),
                "synonyms": d.get("synonyms", []),
                "brand_names": d.get("brand_names", []),
                "wiki_en": d.get("wiki_en"),
            }
            for n, d in graph.nodes(data=True)
        },
        "is_a": [[u, v] for u, v in graph.edges()],
        "alt": alt_to_primary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _read_cache(path: Path) -> Tuple[nx.DiGraph, Dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    graph = nx.DiGraph()
    for node_id, data in payload["nodes"].items():
        graph.add_node(
            node_id,
            name=data.get("name", ""),
            synonyms=data.get("synonyms", []),
            brand_names=data.get("brand_names", []),
            wiki_en=data.get("wiki_en"),
        )
    graph.add_edges_from(payload["is_a"])
    return graph, payload.get("alt", {})


def load_chebi_graph(
    cache_dir: Path,
    variant: str = "full",
    *,
    refresh: bool = False,
) -> nx.DiGraph:
    """
    Return a ChEBI ``is_a`` DiGraph (edges child -> parent).

    Reads the compact cache if present; otherwise downloads (if needed) and
    parses the OBO release, then writes the cache. Node attributes: ``name``
    (str), ``synonyms`` (list[str]), ``wiki_en`` (str | None).
    """
    cache_dir = Path(cache_dir)
    cache = _cache_path(cache_dir, variant)
    if cache.exists() and not refresh:
        graph, _ = _read_cache(cache)
        print(f"Loaded cached ChEBI {variant} graph: {graph.number_of_nodes()} nodes")
        return graph

    obo_gz = cache_dir / _variant_filename(variant)
    if not obo_gz.exists() or refresh:
        _download_obo(variant, obo_gz)

    print(f"Parsing {obo_gz.name} ...")
    graph, alt_to_primary = _parse_obo(obo_gz)
    print(
        f"Parsed ChEBI {variant}: {graph.number_of_nodes()} terms, "
        f"{graph.number_of_edges()} is_a edges"
    )
    _write_cache(cache, graph, alt_to_primary)
    return graph


def taxonomic_neighbors(graph: nx.DiGraph, chebi_id: str) -> Dict[str, Set[str]]:
    """
    1-hop taxonomic neighbors of ``chebi_id`` over the ``is_a`` graph:

      * ``parent``  : direct is_a parents (successors)
      * ``child``   : direct is_a children (predecessors)
      * ``sibling`` : other children of any shared parent

    The concept itself is never returned, and parents/children take precedence
    over siblings when an id would otherwise appear in two buckets.
    """
    if chebi_id not in graph:
        return {"parent": set(), "child": set(), "sibling": set()}

    parents = set(graph.successors(chebi_id))
    children = set(graph.predecessors(chebi_id))
    siblings: Set[str] = set()
    for parent in parents:
        siblings.update(graph.predecessors(parent))
    siblings.discard(chebi_id)
    siblings -= parents
    siblings -= children
    return {"parent": parents, "child": children, "sibling": siblings}
