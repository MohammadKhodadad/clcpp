"""
Idea 2 — Code-Switched Document Variants.

From an original patent document (A) that mentions a ChEBI concept, build noisy /
code-switched copies that swap that one chemistry term:

  B  in-set swap     — the concept's name in another language the patent IS in
  C  out-of-set swap — the concept's name in an in-set language the patent is NOT in
  D  noisy           — same language, perturbed spelling (typo/hyphen/case/Greek/oxidation)
  E  non-chem control— an ordinary non-chemistry noun swapped to another language (LLM)
  F  ChEBI variant   — another form from the concept's name_set["chebi"] (e.g. CO(2))

Replacement forms come from data/alias_graph/alias_graph.json (`name_set`). Every
change is tracked (original_term, replacement_term, languages, variant) so a later
QA step can ask about the original term and the benchmark can check whether the
perturbed document is still retrieved. All occurrences of the chosen term are
replaced across the document's text fields. B/C/D/F are deterministic; only E
uses an LLM.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

from src.alias_graph.builder import CORPUS_FIELDS, _read_corpus
from src.alias_graph.matching import _CJK_RE

IN_SET_LANGS: Tuple[str, ...] = ("en", "de", "fr", "es", "zh")
TEXT_FIELDS: Tuple[str, ...] = ("title", "abstract", "description", "first_claim", "context")
TRACKING_FIELDS: Tuple[str, ...] = (
    "variant", "concept_chebi_id", "concept_name",
    "original_term", "replacement_term", "anchor_language", "target_language",
    "source_id", "source_publication_number",
)
OUTPUT_FIELDS: Tuple[str, ...] = CORPUS_FIELDS + TRACKING_FIELDS


# --------------------------------------------------------------------------- #
# Term location + replacement (operate on raw text)
# --------------------------------------------------------------------------- #

def _term_regex(term: str) -> re.Pattern:
    """Match a term as a standalone token. CJK terms (no word separators) match as
    a plain substring; Latin terms use non-word-char boundaries, case-insensitive."""
    if _CJK_RE.search(term):
        return re.compile(re.escape(term))
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)


def _doc_text(row: Dict[str, str]) -> str:
    return "\n".join(row.get(f) or "" for f in TEXT_FIELDS)


_QUALIFIER_RE = re.compile(r"\s*\([^)]*\)\s*$")


def _clean_name(name: str) -> str:
    """Strip a trailing Wikipedia disambiguation qualifier, e.g. 'Silane (composé)' -> 'Silane'."""
    return _QUALIFIER_RE.sub("", name).strip() or name


def _is_clean_swap(original: str, replacement: str) -> bool:
    """A swap is clean only if it genuinely removes the original surface: the
    replacement is non-empty, differs from the original, and does not still
    contain the original term as a token (e.g. 'sulfate' -> 'Sulfate ion')."""
    if not replacement or replacement.casefold() == original.casefold():
        return False
    return not _term_regex(original).search(replacement)


def _clean_lang_swap(
    name_set: Dict[str, list], original: str, langs: Sequence[str], rng: random.Random
) -> Optional[Tuple[str, str]]:
    """Pick a (target_lang, replacement) from the given languages whose concept
    name yields a clean swap. Returns None if no language qualifies."""
    options = list(langs)
    rng.shuffle(options)
    for t in options:
        for nm in name_set.get(t, []):
            cand = _clean_name(nm)
            if _is_clean_swap(original, cand):
                return t, cand
    return None


def _replace_all(row: Dict[str, str], term: str, replacement: str) -> Optional[Dict[str, str]]:
    """Return a copy of ``row`` with every occurrence of ``term`` in the text fields
    replaced by ``replacement``. Returns None if the term occurs nowhere."""
    rx = _term_regex(term)
    new = dict(row)
    total = 0
    for f in TEXT_FIELDS:
        val = row.get(f) or ""
        if not val:
            continue
        new_val, n = rx.subn(replacement, val)
        if n:
            new[f] = new_val
            total += n
    return new if total else None


def _locate_anchor(
    name_set: Dict[str, list], pub_rows: Dict[str, Dict[str, str]], rng: random.Random
) -> Optional[Tuple[str, str]]:
    """Pick a random (anchor_language, original_term): a language present in the
    publication whose concept term actually occurs in that language's document.

    Prefer the per-language Wikipedia name (a true natural-language term); fall
    back to a ChEBI-bucket name/formula (which can appear in a doc of any
    language, e.g. "CO2") so concepts detected via a ChEBI synonym still anchor.
    """
    natural: List[Tuple[str, str]] = []
    fallback: List[Tuple[str, str]] = []
    # Fallback ChEBI forms: the primary (canonical) name always, plus any
    # *distinctive* synonym (formula/long/hyphenated). This keeps real anchors
    # like "CO2"/"ethanol"/"fluthiacet-methyl" while excluding common-word brand
    # synonyms (e.g. "Action") that aren't really the chemistry term.
    chebi = name_set.get("chebi", [])
    chebi_fallback = (chebi[:1] if chebi else []) + [n for n in chebi[1:] if _distinctive(n)]
    for lang, row in pub_rows.items():
        text = _doc_text(row)
        hit = next((nm for nm in name_set.get(lang, []) if _term_regex(nm).search(text)), None)
        if hit is not None:
            natural.append((lang, hit))
            continue
        hit = next((nm for nm in chebi_fallback if _term_regex(nm).search(text)), None)
        if hit is not None:
            fallback.append((lang, hit))
    pool = natural or fallback
    return rng.choice(pool) if pool else None


def _distinctive(name: str) -> bool:
    """Distinctive enough to be a chemistry term, not a plain common word:
    contains a digit, hyphen, or space, or is reasonably long."""
    return len(name) >= 8 or any(c.isdigit() for c in name) or "-" in name or " " in name


# --------------------------------------------------------------------------- #
# Variant D — spelling perturbations
# --------------------------------------------------------------------------- #

_GREEK_TO_NAME = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "κ": "kappa", "λ": "lambda",
    "μ": "mu", "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau", "φ": "phi",
    "χ": "chi", "ψ": "psi", "ω": "omega",
}
_NAME_TO_GREEK = {v: k for k, v in _GREEK_TO_NAME.items()}
_ROMAN_TO_CHARGE = {
    "(0)": "(0)", "(I)": "(1+)", "(II)": "(2+)", "(III)": "(3+)", "(IV)": "(4+)",
    "(V)": "(5+)", "(VI)": "(6+)", "(VII)": "(7+)",
}
_CHARGE_TO_ROMAN = {v: k for k, v in _ROMAN_TO_CHARGE.items() if k != "(0)"}


def _greek_swap(term: str, rng: random.Random) -> Optional[str]:
    for ch in term:
        if ch in _GREEK_TO_NAME:
            return term.replace(ch, _GREEK_TO_NAME[ch], 1)
    for name in sorted(_NAME_TO_GREEK, key=len, reverse=True):
        m = re.search(name, term, re.IGNORECASE)
        if m:
            return term[: m.start()] + _NAME_TO_GREEK[name] + term[m.end():]
    return None


def _oxidation_swap(term: str, rng: random.Random) -> Optional[str]:
    for roman, charge in _ROMAN_TO_CHARGE.items():
        if roman in term and roman != charge:
            return term.replace(roman, charge, 1)
    for charge, roman in _CHARGE_TO_ROMAN.items():
        if charge in term:
            return term.replace(charge, roman, 1)
    return None


def _hyphen_insert(term: str, rng: random.Random) -> Optional[str]:
    spots = [i for i in range(1, len(term)) if term[i - 1].isalpha() and term[i].isalpha()]
    if not spots:
        return None
    i = rng.choice(spots)
    return term[:i] + "-" + term[i:]


def _typo(term: str, rng: random.Random) -> Optional[str]:
    idx = [i for i, c in enumerate(term) if c.isalpha()]
    if len(idx) < 2:
        return None
    kind = rng.choice(["swap", "drop", "dup"])
    if kind == "swap":
        adj = [i for i in idx if (i + 1) in idx]
        if not adj:
            return None
        i = rng.choice(adj)
        return term[:i] + term[i + 1] + term[i] + term[i + 2:]
    i = rng.choice(idx)
    if kind == "drop":
        return term[:i] + term[i + 1:]
    return term[:i] + term[i] + term[i:]  # duplicate


def _case_noise(term: str, rng: random.Random) -> Optional[str]:
    idx = [i for i, c in enumerate(term) if c.isalpha()]
    if not idx:
        return None
    k = max(1, len(idx) // 3)
    flip = set(rng.sample(idx, min(k, len(idx))))
    out = "".join(
        (c.upper() if c.islower() else c.lower()) if i in flip else c
        for i, c in enumerate(term)
    )
    return out if out != term else None


def _perturb(term: str, rng: random.Random) -> Optional[str]:
    """Apply one applicable spelling perturbation; returns a changed string or None."""
    rules = [_greek_swap, _oxidation_swap, _hyphen_insert, _typo, _case_noise]
    rng.shuffle(rules)
    for rule in rules:
        out = rule(term, rng)
        if out and out != term:
            return out
    return None


# --------------------------------------------------------------------------- #
# Variant F — ChEBI form
# --------------------------------------------------------------------------- #

def _pick_chebi_variant(
    chebi_names: List[str], original: str, rng: random.Random
) -> Optional[str]:
    """Pick a chebi-bucket form that cleanly swaps the original (does not still
    contain it, e.g. avoid 'chloride' -> 'Chloride(1-)'), preferring a
    formula-like one (e.g. CO(2)) to make the swap visibly a different style."""
    pool = [n for n in chebi_names if _is_clean_swap(original, n)]
    if not pool:
        return None
    formulas = [n for n in pool if any(c.isdigit() for c in n)]
    return rng.choice(formulas or pool)


# --------------------------------------------------------------------------- #
# Row assembly
# --------------------------------------------------------------------------- #

def _make_row(
    row: Dict[str, str], variant: str, *, cid: str, cname: str,
    original_term: str, replacement_term: str, anchor_lang: str, target_lang: str,
    source_id: str, pub: str,
) -> Dict[str, str]:
    out = {k: row.get(k, "") for k in CORPUS_FIELDS}
    out["id"] = f"{source_id}__{variant}"
    out.update({
        "variant": variant,
        "concept_chebi_id": cid,
        "concept_name": cname,
        "original_term": original_term,
        "replacement_term": replacement_term,
        "anchor_language": anchor_lang,
        "target_language": target_lang,
        "source_id": source_id,
        "source_publication_number": pub,
    })
    return out


def run_code_switch(
    alias_json: Path,
    corpus_path: Path,
    output_path: Path,
    *,
    variants: Sequence[str] = ("A", "B", "C", "D", "F"),
    limit: Optional[int] = None,
    model: str = "gpt-5-mini",
    seed: int = 42,
) -> int:
    """Build the code-switched variant corpus; returns number of rows written."""
    variants = [v.upper() for v in variants]
    want = set(variants)

    with Path(alias_json).open(encoding="utf-8") as fh:
        entries = json.load(fh)["concepts"]
    if limit is not None:
        entries = entries[:limit]

    by_pub: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for r in _read_corpus(corpus_path):
        by_pub[r["publication_number"]][r["language"]] = r

    rng = random.Random(seed)
    swapper = None
    if "E" in want:
        from src.alias_graph.code_switch_llm import get_nonchem_swapper
        swapper = get_nonchem_swapper(model)

    out_rows: List[Dict[str, str]] = []
    counts: Dict[str, int] = defaultdict(int)

    for entry in tqdm(entries, desc="Code-switch", unit="concept"):
        cid, cname, name_set = entry["chebi_id"], entry["name"], entry["name_set"]
        gold = [p for p in entry["gold"] if p in by_pub]
        if not gold:
            continue
        pub = rng.choice(gold)
        pub_rows = by_pub[pub]
        pub_langs = set(pub_rows)
        anchor = _locate_anchor(name_set, pub_rows, rng)
        if anchor is None:
            continue
        L_a, original = anchor
        anchor_row = pub_rows[L_a]
        source_id = anchor_row["id"]

        def emit(variant: str, new_row: Optional[Dict[str, str]], repl: str, tgt: str) -> None:
            if new_row is None:
                return
            out_rows.append(_make_row(
                new_row, variant, cid=cid, cname=cname, original_term=original,
                replacement_term=repl, anchor_lang=L_a, target_lang=tgt,
                source_id=source_id, pub=pub,
            ))
            counts[variant] += 1

        if "A" in want:
            emit("A", anchor_row, "", "")
        if "B" in want:
            swap = _clean_lang_swap(
                name_set, original, [l for l in pub_langs if l != L_a], rng
            )
            if swap:
                t, repl = swap
                emit("B", _replace_all(anchor_row, original, repl), repl, t)
        if "C" in want:
            swap = _clean_lang_swap(
                name_set, original, [l for l in IN_SET_LANGS if l not in pub_langs], rng
            )
            if swap:
                t, repl = swap
                emit("C", _replace_all(anchor_row, original, repl), repl, t)
        if "D" in want:
            repl = _perturb(original, rng)
            if repl:
                emit("D", _replace_all(anchor_row, original, repl), repl, L_a)
        if "F" in want:
            repl = _pick_chebi_variant(name_set.get("chebi", []), original, rng)
            if repl:
                emit("F", _replace_all(anchor_row, original, repl), repl, "chebi")
        if "E" in want and swapper is not None:
            avoid = {n for v in name_set.values() for n in v} | set(entry.get("codes", []))
            tgt = rng.choice([l for l in IN_SET_LANGS if l != L_a])
            res = swapper(_doc_text(anchor_row), sorted(avoid), tgt)
            if res:
                o, r = res
                row = _replace_all(anchor_row, o, r)
                if row:
                    out_rows.append(_make_row(
                        row, "E", cid=cid, cname=cname, original_term=o,
                        replacement_term=r, anchor_lang=L_a, target_lang=tgt,
                        source_id=source_id, pub=pub,
                    ))
                    counts["E"] += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nWrote {len(out_rows)} variant rows -> {output_path}")
    print(f"  per variant: {dict(sorted(counts.items()))}")
    return len(out_rows)
