"""
Progressive code-switching — a dose-response retrieval-decay experiment.

The single-swap code-switch builder (``code_switch.py``) perturbs ONE chemistry
term per document. This builder instead takes one base document and builds a
**cumulative ladder** of variants:

    clean (0 swaps) -> 1 swap -> 2 -> 3 -> 4 -> 5 swaps

At each step one *more* distinct chemistry term in the document is replaced, using
a randomly chosen mode from {B, C, D, F} (E — the LLM non-chem control — is
excluded here):

  B  in-set swap     — the term's name in another language the patent IS in
  C  out-of-set swap — the term's name in an in-set language the patent is NOT in
  D  noisy           — same language, perturbed spelling (typo/hyphen/case/Greek/oxidation)
  F  ChEBI variant   — another form from the concept's name_set["chebi"] (e.g. CO(2))

A single fixed question (generated later in ``progressive_qa``) is about the
**first** replaced term — the term that changes at step 1. It is reused as the
query for all 6 variant documents, so the benchmark measures how fast retrieval
of the (increasingly code-switched) document falls apart as the dose grows.

All replacement forms come from ``data/alias_graph/alias_graph.json``
(``name_set``); every step is tracked in ``replacements_json`` so a later eval can
break the decay down by mode. The whole build is deterministic given ``seed`` —
only B/C/D/F (no LLM) are used.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from src.alias_graph.builder import CORPUS_FIELDS, _read_corpus
from src.alias_graph.code_switch import (
    IN_SET_LANGS,
    _clean_lang_swap,
    _locate_anchor,
    _perturb,
    _pick_chebi_variant,
    _replace_all,
    _term_regex,
)

# Preference order used only to break ties when two language versions of a
# document yield the same number of swappable terms.
_LANG_PRIORITY: Tuple[str, ...] = ("en", "de", "fr", "es", "zh")

TRACKING_FIELDS: Tuple[str, ...] = (
    "base_id", "n_replacements", "anchor_language", "source_publication_number",
    "question_concept_chebi_id", "question_concept_name", "question_original_term",
    "replacements_json",
)
OUTPUT_FIELDS: Tuple[str, ...] = CORPUS_FIELDS + TRACKING_FIELDS


# --------------------------------------------------------------------------- #
# Per-term mode replacements
# --------------------------------------------------------------------------- #

def _mode_replacements(
    name_set: Dict[str, list], original: str, pub_langs: set, anchor_lang: str,
    modes: Sequence[str], rng: random.Random,
) -> Dict[str, Tuple[str, str]]:
    """For each requested mode, compute a clean ``(replacement, target_lang)`` for
    ``original`` (skipping modes with no clean swap). Deterministic given ``rng``."""
    res: Dict[str, Tuple[str, str]] = {}
    for m in modes:
        if m == "B":
            sw = _clean_lang_swap(name_set, original, [l for l in pub_langs if l != anchor_lang], rng)
            if sw:
                res["B"] = (sw[1], sw[0])
        elif m == "C":
            sw = _clean_lang_swap(name_set, original, [l for l in IN_SET_LANGS if l not in pub_langs], rng)
            if sw:
                res["C"] = (sw[1], sw[0])
        elif m == "D":
            r = _perturb(original, rng)
            if r:
                res["D"] = (r, anchor_lang)
        elif m == "F":
            r = _pick_chebi_variant(name_set.get("chebi", []), original, rng)
            if r:
                res["F"] = (r, "chebi")
    return res


def _swappable_terms(
    row: Dict[str, str], lang: str, pub_langs: set, gold_cids: Sequence[str],
    by_cid: Dict[str, dict], modes: Sequence[str], rng: random.Random,
) -> List[Dict]:
    """Concepts whose term is locatable in this language's document AND have at
    least one applicable B/C/D/F replacement. ``pub_langs`` is the full set of the
    publication's languages (B swaps to another present language, C to an in-set
    language the publication lacks). Each concept carries its precomputed
    ``mode_repls`` so the replacement string is only computed once."""
    out: List[Dict] = []
    for cid in gold_cids:
        concept = by_cid.get(cid)
        if concept is None:
            continue
        name_set = concept["name_set"]
        anchor = _locate_anchor(name_set, {lang: row}, rng)
        if anchor is None:
            continue
        anchor_lang, original = anchor  # anchor_lang == lang
        mode_repls = _mode_replacements(name_set, original, pub_langs, anchor_lang, modes, rng)
        if mode_repls:
            out.append({
                "cid": cid, "name": concept["name"], "original": original,
                "mode_repls": mode_repls,
            })
    return out


def _select_terms(swappable: List[Dict], n_steps: int, rng: random.Random) -> Optional[List[Dict]]:
    """Greedily pick ``n_steps`` terms with non-overlapping surfaces (so cumulative
    replacements stay independent), in a random order. Returns None if fewer than
    ``n_steps`` survive the overlap guard."""
    order = list(swappable)
    rng.shuffle(order)
    chosen: List[Dict] = []
    chosen_surfaces: List[str] = []
    for cand in order:
        surf = cand["original"].casefold()
        if any(surf in s or s in surf for s in chosen_surfaces):
            continue  # surface overlaps an already-chosen term
        chosen.append(cand)
        chosen_surfaces.append(surf)
        if len(chosen) == n_steps:
            return chosen
    return None


# --------------------------------------------------------------------------- #
# Row assembly
# --------------------------------------------------------------------------- #

def _make_row(
    row: Dict[str, str], *, base_id: str, depth: int, anchor_lang: str, pub: str,
    q_cid: str, q_name: str, q_original: str, steps: List[Dict],
) -> Dict[str, str]:
    out = {k: row.get(k, "") for k in CORPUS_FIELDS}
    out["id"] = f"{base_id}__r{depth}"
    out.update({
        "base_id": base_id,
        "n_replacements": str(depth),
        "anchor_language": anchor_lang,
        "source_publication_number": pub,
        "question_concept_chebi_id": q_cid,
        "question_concept_name": q_name,
        "question_original_term": q_original,
        "replacements_json": json.dumps(steps[:depth], ensure_ascii=False),
    })
    return out


def run_progressive_code_switch(
    alias_json: Path,
    corpus_path: Path,
    output_path: Path,
    *,
    n_steps: int = 5,
    modes: Sequence[str] = ("B", "C", "D", "F"),
    limit: Optional[int] = None,
    seed: int = 42,
) -> int:
    """Build the progressive (cumulative-ladder) code-switched corpus. Returns the
    number of variant rows written (``n_bases * (n_steps + 1)``)."""
    modes = [m.upper() for m in modes if m.upper() in ("B", "C", "D", "F")]

    with Path(alias_json).open(encoding="utf-8") as fh:
        concepts = json.load(fh)["concepts"]
    by_cid = {c["chebi_id"]: c for c in concepts}

    # Invert gold membership: publication -> [concept ids that mention it].
    pub2cids: Dict[str, List[str]] = defaultdict(list)
    for c in concepts:
        for pub in c.get("gold", []):
            pub2cids[pub].append(c["chebi_id"])

    by_pub: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for r in _read_corpus(corpus_path):
        by_pub[r["publication_number"]][r["language"]] = r

    rng = random.Random(seed)
    # Deterministic, reproducible publication order: sort, then shuffle with rng.
    candidate_pubs = sorted(p for p, cids in pub2cids.items() if len(cids) >= n_steps and p in by_pub)
    rng.shuffle(candidate_pubs)

    out_rows: List[Dict[str, str]] = []
    mode_counts: Dict[str, int] = defaultdict(int)
    n_bases = 0

    for pub in tqdm(candidate_pubs, desc="Progressive CS", unit="pub"):
        if limit is not None and n_bases >= limit:
            break
        pub_rows = by_pub[pub]
        pub_langs = set(pub_rows)
        gold_cids = pub2cids[pub]

        # Pick the language version that yields the most swappable terms.
        best_lang: Optional[str] = None
        best_swappable: List[Dict] = []
        for lang in sorted(pub_rows, key=lambda l: (_LANG_PRIORITY.index(l) if l in _LANG_PRIORITY else 99, l)):
            swappable = _swappable_terms(pub_rows[lang], lang, pub_langs, gold_cids, by_cid, modes, rng)
            if len(swappable) > len(best_swappable):
                best_swappable, best_lang = swappable, lang
        if best_lang is None or len(best_swappable) < n_steps:
            continue

        chosen = _select_terms(best_swappable, n_steps, rng)
        if chosen is None:
            continue

        # Assign each chosen term a random applicable mode -> (replacement, target_lang).
        anchor_row = pub_rows[best_lang]
        steps: List[Dict] = []
        for i, cand in enumerate(chosen, start=1):
            mode = rng.choice(sorted(cand["mode_repls"]))
            replacement, target_lang = cand["mode_repls"][mode]
            steps.append({
                "step": i, "concept_id": cand["cid"], "original": cand["original"],
                "replacement": replacement, "mode": mode, "target_language": target_lang,
            })

        # Build the cumulative ladder, applying steps 1..k to the anchor doc.
        q = chosen[0]  # term_1 — the question term
        ladder: List[Dict[str, str]] = [
            _make_row(anchor_row, base_id=anchor_row["id"], depth=0, anchor_lang=best_lang,
                      pub=pub, q_cid=q["cid"], q_name=q["name"], q_original=q["original"], steps=steps)
        ]
        cur = anchor_row
        ok = True
        for k, step in enumerate(steps, start=1):
            nxt = _replace_all(cur, step["original"], step["replacement"])
            if nxt is None:  # term vanished after an earlier swap — drop this base
                ok = False
                break
            cur = nxt
            ladder.append(
                _make_row(cur, base_id=anchor_row["id"], depth=k, anchor_lang=best_lang,
                          pub=pub, q_cid=q["cid"], q_name=q["name"], q_original=q["original"], steps=steps)
            )
        if not ok:
            continue

        out_rows.extend(ladder)
        for step in steps:
            mode_counts[step["mode"]] += 1
        n_bases += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nWrote {len(out_rows)} rows ({n_bases} base docs x {n_steps + 1} depths) -> {output_path}")
    print(f"  per-step mode counts: {dict(sorted(mode_counts.items()))}")
    return len(out_rows)
