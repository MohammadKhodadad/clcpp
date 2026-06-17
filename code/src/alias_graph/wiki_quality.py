"""
Quality check for the Wikipedia-derived concept names.

We attach, to each ChEBI concept, a Wikipedia title per language (via the
ChEBI->Wikidata P683->sitelink bridge). Are those titles the words actually used
in real translated patents? This module measures that directly:

  For a concept mentioned in an English document, take the *same patent's*
  translated document (same publication_number) in language L and check whether
  the concept's L-language Wikipedia title appears there.

Example: caffeine's German Wikipedia title is "Coffein". If the German
translation of an English caffeine patent contains "Coffein", the name is good;
if the German text instead says "Koffein", we record a miss with the alternative
surface, which pinpoints the name-quality issue.

Each (concept, parallel-doc, language) check records:
  * wiki_present    -- the L Wikipedia title occurs in the L document
  * concept_present -- the concept occurs in the L document by *any* name
  * matched_instead -- the surface that matched when the Wikipedia title did not
The conditional rate wiki_present / concept_present isolates name quality from
patents whose translation simply does not mention the concept.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.alias_graph.builder import _read_corpus
from src.alias_graph.chebi import load_chebi_graph
from src.alias_graph.matching import (
    build_name_index,
    contains_name,
    prune_names,
    scan_corpus,
)
from src.alias_graph.wikidata_names import (
    DEFAULT_LANGS,
    fetch_wikipedia_names,
)

PIVOT_LANG = "en"


def check_wiki_name_quality(
    corpus_csv: Path,
    chebi_cache_dir: Path,
    output_dir: Path,
    *,
    langs: Sequence[str] = DEFAULT_LANGS,
    variant: str = "full",
    match_field: str = "context",
    min_name_len: int = 4,
    max_concepts_per_name: int = 3,
    max_df_ratio: float = 0.02,
    wiki_cache_path: Optional[Path] = None,
) -> dict:
    """Measure how often Wikipedia names appear in parallel translated docs."""
    corpus_csv = Path(corpus_csv)
    chebi_cache_dir = Path(chebi_cache_dir)
    output_dir = Path(output_dir)
    wiki_cache_path = Path(wiki_cache_path) if wiki_cache_path else chebi_cache_dir / "wiki_names_cache.json"
    target_langs = [lang for lang in langs if lang != PIVOT_LANG]

    print(f"Reading corpus: {corpus_csv}")
    rows = _read_corpus(corpus_csv)
    groups: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for r in rows:
        groups[r["publication_number"]][r["language"]] = r
    print(f"  {len(rows)} documents, {len(groups)} publications")

    graph = load_chebi_graph(chebi_cache_dir, variant)

    # KG-only pass: which concepts occur + per-name document frequency.
    kg_index = build_name_index(
        graph, {}, min_len=min_name_len, max_concepts_per_name=max_concepts_per_name
    )
    concept_to_docs, _, name_doc_freq = scan_corpus(rows, kg_index, field=match_field)
    stop_grams = {g for g, n in name_doc_freq.items() if n > max_df_ratio * len(rows)}

    wiki_names = fetch_wikipedia_names(
        list(concept_to_docs.keys()), langs=langs, cache_path=wiki_cache_path
    )

    # Full pass (Wikipedia names folded in, stopwords pruned) for any-name presence.
    index = build_name_index(
        graph, wiki_names, min_len=min_name_len, max_concepts_per_name=max_concepts_per_name
    )
    prune_names(index, stop_grams)
    _, match_info, _ = scan_corpus(rows, index, field=match_field)

    # Concepts detected in an English document, with their pivot publication.
    detected: set[str] = set()
    en_concept_pubs: Dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r["language"] != PIVOT_LANG:
            continue
        for cid in match_info.get(r["id"], {}):
            detected.add(cid)
            en_concept_pubs[cid].add(r["publication_number"])
    print(f"  concepts detected in English docs: {len(detected)}")

    # One check per (concept, pivot publication, target language).
    checks: List[dict] = []
    for cid, pubs in en_concept_pubs.items():
        cname = graph.nodes[cid].get("name", cid)
        for lang in target_langs:
            title = wiki_names.get(cid, {}).get(lang)
            if not title:
                continue
            for pub in pubs:
                parallel = groups[pub].get(lang)
                if parallel is None:
                    continue
                wiki_present = contains_name(parallel.get(match_field) or "", title)
                pinfo = match_info.get(parallel["id"], {})
                concept_present = cid in pinfo
                matched_instead = (
                    pinfo[cid][0] if (concept_present and not wiki_present) else ""
                )
                checks.append({
                    "chebi_id": cid,
                    "concept_name": cname,
                    "lang": lang,
                    "wiki_title": title,
                    "wiki_present": int(wiki_present),
                    "concept_present": int(concept_present),
                    "matched_instead": matched_instead,
                    "publication_number": pub,
                })

    summary = _summarize(checks, detected, wiki_names, target_langs)
    _write_outputs(output_dir, checks, summary, target_langs)
    _print_summary(summary, target_langs)
    return summary


def _summarize(
    checks: List[dict],
    detected: set[str],
    wiki_names: Dict[str, Dict[str, str]],
    target_langs: Sequence[str],
) -> dict:
    per_lang: Dict[str, dict] = {}
    for lang in target_langs:
        lc = [c for c in checks if c["lang"] == lang]
        n = len(lc)
        wiki_present = sum(c["wiki_present"] for c in lc)
        concept_present = sum(c["concept_present"] for c in lc)
        with_name = sum(1 for cid in detected if wiki_names.get(cid, {}).get(lang))
        per_lang[lang] = {
            "concepts_detected": len(detected),
            "concepts_with_wiki_name": with_name,
            "name_coverage": (with_name / len(detected)) if detected else 0.0,
            "checks": n,
            "wiki_present": wiki_present,
            "concept_present": concept_present,
            "wiki_hit_rate": (wiki_present / n) if n else 0.0,
            "concept_present_rate": (concept_present / n) if n else 0.0,
            "conditional_hit_rate": (wiki_present / concept_present) if concept_present else 0.0,
        }
    return {"per_lang": per_lang}


def _write_outputs(
    output_dir: Path, checks: List[dict], summary: dict, target_langs: Sequence[str]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "chebi_id", "concept_name", "lang", "wiki_title",
        "wiki_present", "concept_present", "matched_instead", "publication_number",
    ]
    with (output_dir / "per_pair.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(checks)

    # Misses: concept present in the translation, but via a name other than the
    # Wikipedia title -> the Wikipedia name is a poor surface form here.
    misses = [c for c in checks if c["concept_present"] and not c["wiki_present"]]
    with (output_dir / "misses.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(misses, key=lambda c: (c["lang"], c["concept_name"])))

    lines = ["# Wikipedia-name quality check", ""]
    lines.append(
        "For a concept mentioned in an English patent, does its Wikipedia title in "
        "language L appear in the same patent's L translation?\n"
    )
    lines.append("| lang | name coverage | checks | wiki hit rate | concept present | conditional (wiki\\|present) |")
    lines.append("|------|---------------|--------|---------------|-----------------|------------------------------|")
    for lang in target_langs:
        s = summary["per_lang"][lang]
        lines.append(
            f"| {lang} | {s['concepts_with_wiki_name']}/{s['concepts_detected']} "
            f"({s['name_coverage']:.0%}) | {s['checks']} | "
            f"{s['wiki_present']}/{s['checks']} ({s['wiki_hit_rate']:.1%}) | "
            f"{s['concept_present_rate']:.1%} | {s['conditional_hit_rate']:.1%} |"
        )
    lines.append("")
    lines.append("- **wiki hit rate**: Wikipedia title found in the L translation.")
    lines.append("- **concept present**: the concept appears in the L translation by *any* name.")
    lines.append("- **conditional**: hit rate among docs where the concept is actually present "
                 "(isolates name quality from untranslated mentions).")
    lines.append("")
    lines.append("## Most common name mismatches (concept present, Wikipedia title absent)")
    lines.append("")
    top = Counter(
        (c["lang"], c["concept_name"], c["wiki_title"], c["matched_instead"])
        for c in misses if c["matched_instead"]
    )
    if top:
        lines.append("| lang | concept | Wikipedia title | matched instead | count |")
        lines.append("|------|---------|-----------------|-----------------|-------|")
        for (lang, cname, title, instead), cnt in top.most_common(40):
            lines.append(f"| {lang} | {cname} | {title} | {instead} | {cnt} |")
    else:
        lines.append("_None._")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(summary: dict, target_langs: Sequence[str]) -> None:
    print("\nWikipedia-name quality (Wikipedia title found in parallel L translation):")
    print(f"  {'lang':>4}  {'coverage':>9}  {'checks':>7}  {'wiki hit':>9}  {'present':>8}  {'conditional':>11}")
    for lang in target_langs:
        s = summary["per_lang"][lang]
        print(
            f"  {lang:>4}  {s['name_coverage']:>8.0%}  {s['checks']:>7}  "
            f"{s['wiki_hit_rate']:>8.1%}  {s['concept_present_rate']:>7.1%}  "
            f"{s['conditional_hit_rate']:>10.1%}"
        )
