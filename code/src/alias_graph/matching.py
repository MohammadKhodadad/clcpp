"""
Concept <-> document matching for the Alias-Graph Retrieval benchmark.

A concept's name set (ChEBI primary name + ChEBI synonyms + Wikipedia titles in
the target languages) is folded into a single normalized name index. The corpus
is then scanned once: every document's text is tokenized the same way the names
were, and 1..N token windows are looked up in the index. This inverted, single
pass is what makes "which concepts does each document mention" tractable over
~200k ChEBI concepts without a per-name regex sweep.

Normalization keeps the internal structure of chemical names (digits, hyphens,
parentheses) while stripping only surrounding punctuation, so
``2-(acetyloxy)benzoic acid`` in a name lines up with the same string in a
patent. Latin-script names match via token n-grams; CJK names (e.g. Chinese
Wikipedia titles) match via direct substring search, which only fires once the
corpus actually contains CJK text.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import networkx as nx


# Longest token-window we look up. Names longer than this are dropped from the
# index: very long systematic names rarely appear verbatim and only add cost.
MAX_NGRAM = 8

_STRIP_RE = re.compile(r"^\W+|\W+$", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯豈-﫿]")

# Hyper-generic names that would otherwise produce meaningless gold sets. Kept
# small and deliberate; the per-name ambiguity cap handles the rest.
_BLACKLIST = {
    "acid", "acids", "base", "bases", "salt", "salts", "water", "ion", "ions",
    "gas", "oil", "oils", "metal", "metals", "group", "groups", "compound",
    "compounds", "element", "elements", "molecule", "molecules", "mixture",
    "solution", "solvent", "polymer", "polymers", "ester", "esters", "amine",
    "amines", "alcohol", "alcohols", "sugar", "sugars", "dye", "dyes", "fuel",
    "agent", "agents", "drug", "drugs", "food", "light", "heat", "carbon",
    "agua", "sel", "eau", "wasser", "salz", "saeure", "acide", "acido", "acida",
}

# Registry / regulatory codes that ChEBI lists as "synonyms" but which are not
# names (E-numbers, refrigerant numbers, company/database codes, CAS/EC numbers).
# Used to keep them out of name sets and the matching index. Chemical formulas
# (CO2) are intentionally NOT treated as codes here.
_CODE_PATTERNS = [
    re.compile(r"^E[ \-]?\d{3,4}[a-z]?$", re.IGNORECASE),       # E290, E-290, E 290, E160a
    re.compile(r"^R[ \-]?\d{2,4}[a-z]?$", re.IGNORECASE),       # R-744, R134a
    re.compile(r"^\d{1,7}-\d{2}-\d$"),                           # CAS, e.g. 50-00-0
    re.compile(r"^\d{3}-\d{3}-\d$"),                             # EC / EINECS
    re.compile(r"^[A-Z]{2,5}[ \-]?\d{2,6}(-\d{1,5})?[a-z]?$"),   # CGA 248757, KIH-9201, BAY 12-9566
]


def is_code_like_name(name: str) -> bool:
    """True if ``name`` is a registry/regulatory code rather than a real name."""
    stripped = (name or "").strip()
    return any(p.match(stripped) for p in _CODE_PATTERNS)


def _normalize(text: str) -> List[str]:
    """NFKC + casefold + whitespace-tokenize, stripping surrounding punctuation
    but preserving token-internal structure. Returns the list of tokens."""
    text = unicodedata.normalize("NFKC", text).casefold()
    tokens: List[str] = []
    for raw in _WS_RE.split(text):
        tok = _STRIP_RE.sub("", raw)
        if tok:
            tokens.append(tok)
    return tokens


def _norm_name(surface: str) -> Tuple[str, int]:
    """Return (normalized joined name, token_count)."""
    toks = _normalize(surface)
    return " ".join(toks), len(toks)


def contains_name(text: str, name: str) -> bool:
    """
    True if ``name`` occurs in ``text`` under the same normalization the scanner
    uses: Latin names match on a contiguous token window (word boundaries
    preserved); CJK names match as a substring of the joined CJK text. Empty or
    over-long (> MAX_NGRAM tokens) names never match.
    """
    name_norm, n_tok = _norm_name(name)
    if not name_norm:
        return False
    if _CJK_RE.search(name_norm):
        return name_norm.replace(" ", "") in "".join(_normalize(text))
    if n_tok > MAX_NGRAM:
        return False
    tokens = _normalize(text)
    if n_tok == 1:
        return name_norm in set(tokens)
    target = tokens
    for i in range(0, len(target) - n_tok + 1):
        if " ".join(target[i : i + n_tok]) == name_norm:
            return True
    return False


@dataclass
class NameIndex:
    latin: Dict[str, Set[str]] = field(default_factory=dict)   # norm_name -> chebi ids
    cjk: Dict[str, Set[str]] = field(default_factory=dict)     # norm_name -> chebi ids
    langs: Dict[str, Set[str]] = field(default_factory=dict)   # norm_name -> source langs
    surface: Dict[str, str] = field(default_factory=dict)      # norm_name -> first surface
    max_ngram: int = MAX_NGRAM

    def n_names(self) -> int:
        return len(self.latin) + len(self.cjk)


def _iter_concept_names(
    graph: nx.DiGraph,
    wiki_names: Dict[str, Dict[str, str]],
) -> Iterable[Tuple[str, str, str, bool]]:
    """Yield (chebi_id, surface_name, source_lang, is_primary) for every alias."""
    for cid, data in graph.nodes(data=True):
        name = data.get("name")
        if name:
            yield cid, name, "chebi", True
        for syn in data.get("synonyms", ()):
            if is_code_like_name(syn):
                continue  # registry/regulatory code, not a matchable name
            yield cid, syn, "chebi", False
    for cid, by_lang in wiki_names.items():
        if cid in graph:
            for lang, title in by_lang.items():
                yield cid, title, lang, False


def build_name_index(
    graph: nx.DiGraph,
    wiki_names: Dict[str, Dict[str, str]],
    *,
    min_len: int = 4,
    max_concepts_per_name: int = 3,
) -> NameIndex:
    """
    Build the normalized alias -> concept index.

    Drops names that are too short, blacklisted, purely numeric, longer than
    ``MAX_NGRAM`` tokens, or ambiguous (mapping to more than
    ``max_concepts_per_name`` distinct concepts).
    """
    # A normalized name that is the *primary* name of some concept(s); used to
    # stop a colloquial synonym (e.g. "alcohol" on ethanol) or a Wikipedia title
    # from attaching to a different concept whose proper name it already is.
    primary_owners: Dict[str, Set[str]] = {}
    for cid, data in graph.nodes(data=True):
        name = data.get("name")
        if name:
            norm, _ = _norm_name(name)
            if norm:
                primary_owners.setdefault(norm, set()).add(cid)

    idx = NameIndex()
    for cid, surface, lang, is_primary in _iter_concept_names(graph, wiki_names):
        norm, n_tok = _norm_name(surface)
        if not norm or n_tok > MAX_NGRAM:
            continue
        if not is_primary:
            owners = primary_owners.get(norm)
            if owners and cid not in owners:
                continue  # this name is another concept's primary name
        is_cjk = bool(_CJK_RE.search(norm))
        if is_cjk:
            if len(norm.replace(" ", "")) < 2:
                continue
        else:
            if len(norm) < min_len or norm in _BLACKLIST or norm.isdigit():
                continue
        bucket = idx.cjk if is_cjk else idx.latin
        bucket.setdefault(norm, set()).add(cid)
        idx.langs.setdefault(norm, set()).add(lang)
        idx.surface.setdefault(norm, surface)

    # Resolve names shared between a concept and one of its is_a ancestors: the
    # name denotes the broader concept (e.g. "Alkohol" belongs to the alcohol
    # class, not ethanol), so drop it from the more specific descendant owners.
    _anc_cache: Dict[str, Set[str]] = {}

    def _ancestors(node: str) -> Set[str]:
        cached = _anc_cache.get(node)
        if cached is None:
            cached = nx.descendants(graph, node)  # is_a edges point child->parent
            _anc_cache[node] = cached
        return cached

    for bucket in (idx.latin, idx.cjk):
        for norm, owners in list(bucket.items()):
            if not (2 <= len(owners) <= 8):
                continue
            drop = {b for b in owners for a in owners if a != b and a in _ancestors(b)}
            if drop:
                owners -= drop
                if not owners:
                    del bucket[norm]
                    idx.langs.pop(norm, None)
                    idx.surface.pop(norm, None)

    # Drop ambiguous names (mapping to too many concepts).
    for bucket in (idx.latin, idx.cjk):
        ambiguous = [n for n, ids in bucket.items() if len(ids) > max_concepts_per_name]
        for n in ambiguous:
            del bucket[n]
            idx.langs.pop(n, None)
            idx.surface.pop(n, None)

    return idx


def prune_names(index: NameIndex, names: Set[str]) -> int:
    """Remove the given normalized names from the index (in place). Returns count."""
    removed = 0
    for bucket in (index.latin, index.cjk):
        for n in names:
            if bucket.pop(n, None) is not None:
                removed += 1
                index.langs.pop(n, None)
                index.surface.pop(n, None)
    return removed


def _pick_lang(idx: NameIndex, norm: str, doc_lang: str) -> str:
    langs = idx.langs.get(norm, set())
    if doc_lang in langs:
        return doc_lang
    if langs == {"chebi"}:
        return "chebi"
    # Prefer an explicit Wikipedia language over the generic "chebi" tag.
    non_chebi = sorted(langs - {"chebi"})
    return non_chebi[0] if non_chebi else "chebi"


def scan_corpus(
    rows: Sequence[dict],
    index: NameIndex,
    *,
    field: str = "context",
) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, Tuple[str, str]]], Counter]:
    """
    Scan corpus rows for concept mentions.

    Returns:
      * ``concept_to_docs``: ``{chebi_id: {doc_id, ...}}``
      * ``match_info``:      ``{doc_id: {chebi_id: (matched_surface, matched_lang)}}``
      * ``name_doc_freq``:   ``Counter{normalized_name: n_docs}`` (for stopword pruning)
    """
    concept_to_docs: Dict[str, Set[str]] = {}
    match_info: Dict[str, Dict[str, Tuple[str, str]]] = {}
    name_doc_freq: Counter = Counter()
    has_cjk_index = bool(index.cjk)

    for row in rows:
        doc_id = row["id"]
        doc_lang = row.get("language", "")
        text = row.get(field) or ""
        tokens = _normalize(text)
        n = len(tokens)
        hits: Dict[str, Tuple[str, str]] = {}
        doc_grams: Set[str] = set()

        max_w = min(index.max_ngram, n)
        for w in range(1, max_w + 1):
            for i in range(0, n - w + 1):
                gram = " ".join(tokens[i : i + w])
                ids = index.latin.get(gram)
                if ids:
                    doc_grams.add(gram)
                    lang = _pick_lang(index, gram, doc_lang)
                    surf = index.surface.get(gram, gram)
                    for cid in ids:
                        hits[cid] = (surf, lang)

        if has_cjk_index and _CJK_RE.search(text):
            joined = "".join(tokens)
            for gram, ids in index.cjk.items():
                if gram.replace(" ", "") in joined:
                    doc_grams.add(gram)
                    lang = _pick_lang(index, gram, doc_lang)
                    surf = index.surface.get(gram, gram)
                    for cid in ids:
                        hits[cid] = (surf, lang)

        name_doc_freq.update(doc_grams)
        if hits:
            match_info[doc_id] = hits
            for cid in hits:
                concept_to_docs.setdefault(cid, set()).add(doc_id)

    return concept_to_docs, match_info, name_doc_freq
