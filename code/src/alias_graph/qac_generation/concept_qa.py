"""
Concept-query QA generation for the Alias-Graph benchmark.

For each ChEBI concept in ``alias_graph.json`` we generate ONE technical search
query whose *answer is the concept itself* (concept = Aspirin -> a query that
describes aspirin from a gold patent without naming it; answer = "Aspirin"). The
query language is chosen with the same four strategies as the multilingual QAC
pipeline, and the two existing verifiers (faithfulness + technical quality) grade
the single (question, answer) pair with the same criteria.

This module deliberately reuses the verifiers, strategies, and helpers from
``multi_lingual_qac.qac_generation.multilingual_qa`` so the grading is identical;
only the generation step (one concept-centric query, with per-language prompts in
``concept_query_generation_prompts/``) is new.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

from src.alias_graph.matching import contains_name
from src.multi_lingual_qac.qac_generation.multilingual_qa import (
    ALL_LANGS,
    DEFAULT_GENERATION_REASONING_EFFORT,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    FAITHFULNESS_FIELDS,
    MODE_TECHNICAL,
    STRATEGY_NAMES,
    STRATEGY_RANDOM_ANY,
    TECHNICAL_QUALITY_FIELDS,
    _build_all_passages_text,
    _compute_faith_overall,
    _compute_quality_overall,
    _compute_total_score,
    _get_client,
    _parse_json_response,
    _pick_context,
    _serialize_context_languages,
    load_multilingual_corpus,
    pick_target_languages,
)

_BASE_DIR = Path(__file__).resolve().parent
_PROMPT_DIR = _BASE_DIR / "concept_query_generation_prompts"
_MAX_GEN_RETRIES = 3  # retry generation on empty/errored response (same document)
_prompt_cache: Dict[str, str] = {}

# Order in which to prefer a language when grounding the answer surface form.
_ANSWER_LANG_ORDER = ["en", "de", "fr", "es", "zh", "chebi"]


_FAITHFULNESS_PROMPT = _BASE_DIR / "faithfulness_prompt" / "faithfulness_prompt.txt"
_QUALITY_PROMPT = _BASE_DIR / "technical_quality_verifier_prompt" / "verifier.txt"


def _load_text(path: Path) -> str:
    key = str(path)
    if key not in _prompt_cache:
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        _prompt_cache[key] = path.read_text(encoding="utf-8").strip()
    return _prompt_cache[key]


def _load_generation_prompt(lang: str) -> str:
    return _load_text(_PROMPT_DIR / f"{lang}.txt")


def grade_faithfulness_single(
    client, all_passages: str, qa: Dict[str, str], *, model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """Faithfulness grade for ONE (question, answer) pair (same rubric, single-object output)."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _load_text(_FAITHFULNESS_PROMPT)},
            {"role": "user", "content": f"{all_passages}\n\nQuestion: {qa['question']}\nAnswer: {qa['answer']}"},
        ],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    if isinstance(data, list):
        data = data[0] if data else {}
    row = {
        "grounding": int(data.get("grounding", 1)),
        "precision": int(data.get("precision", 1)),
        "numerical_fidelity": int(data.get("numerical_fidelity", 1)),
        "reason": str(data.get("reason", "")).strip(),
    }
    row["overall"] = _compute_faith_overall(row)
    return row


def grade_quality_single(
    client, all_passages: str, qa: Dict[str, str], *, model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """Technical-quality grade for ONE question (same rubric, single-object output)."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _load_text(_QUALITY_PROMPT)},
            {"role": "user", "content": f"{all_passages}\n\nQuestion: {qa['question']}\nAnswer: {qa['answer']}"},
        ],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    if isinstance(data, list):
        data = data[0] if data else {}
    row = {
        "search_bar_realism": int(data.get("search_bar_realism", 1)),
        "specificity": int(data.get("specificity", 1)),
        "phrasing_economy": int(data.get("phrasing_economy", 1)),
        "focus": int(data.get("focus", 1)),
        "linguistic_quality": int(data.get("linguistic_quality", 1)),
        "failure_type": str(data.get("failure_type", "none")).strip(),
        "reason": str(data.get("reason", "")).strip(),
    }
    row["overall"] = _compute_quality_overall(row, MODE_TECHNICAL)
    return row


def load_alias_graph(json_path: Path) -> List[Dict[str, Any]]:
    with Path(json_path).open(encoding="utf-8") as fh:
        return json.load(fh)["concepts"]


def _names_for_lang(name_set: Dict[str, Any], lang: str) -> List[str]:
    value = name_set.get(lang)
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


def _all_aliases(name_set: Dict[str, Any]) -> List[str]:
    seen: Dict[str, None] = {}
    for value in name_set.values():
        names = value if isinstance(value, list) else [value]
        for n in names:
            if n:
                seen.setdefault(n, None)
    return list(seen)


def _pick_answer(
    name_set: Dict[str, Any], target_lang: str, passages: str
) -> Tuple[str, str, bool]:
    """
    Choose the answer (the concept's name). Decision: answer = the concept's name
    in the query language. We use the query-language Wikipedia title when the
    concept has one (preferring a grounded variant), otherwise the **canonical
    ChEBI primary name** (English). We never scan the full ChEBI synonym list,
    because it contains brand names and formulas that are common words (e.g.
    "Action", "CO2") and would be wrongly picked up as grounded.
    Returns (answer, lang, grounded).
    """
    if target_lang != "chebi":
        target_names = _names_for_lang(name_set, target_lang)
        for nm in target_names:
            if contains_name(passages, nm):
                return nm, target_lang, True
        if target_names:
            return target_names[0], target_lang, False

    # No Wikipedia title in the query language: use the ChEBI primary name (the
    # first entry of the chebi bucket is the canonical name, not a synonym).
    chebi_names = _names_for_lang(name_set, "chebi")
    if chebi_names:
        primary = chebi_names[0]
        return primary, "chebi", contains_name(passages, primary)

    # Last resort: any Wikipedia title we have.
    for lang in _ANSWER_LANG_ORDER:
        names = _names_for_lang(name_set, lang)
        if names:
            return names[0], lang, contains_name(passages, names[0])
    return "", "", False


def generate_concept_query(
    client,
    all_passages: str,
    concept_name: str,
    aliases: Sequence[str],
    target_lang: str,
    *,
    model: str = DEFAULT_MODEL,
) -> Dict[str, str]:
    """Generate ONE concept-centric technical query in ``target_lang``."""
    prompt = _load_generation_prompt(target_lang)
    alias_block = ", ".join(aliases)
    user = (
        f"CONCEPT (the answer): {concept_name}\n"
        f"ALIASES — never use any of these in the question (any language/spelling): {alias_block}\n\n"
        f"PASSAGES:\n{all_passages}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ],
        reasoning_effort=DEFAULT_GENERATION_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    if isinstance(data, list):
        data = data[0] if data else {}
    return {
        "question": str(data.get("question", "")).strip(),
        "question_type": str(data.get("question_type", "other")).strip(),
    }


def _build_row(
    entry: Dict[str, Any],
    qa: Dict[str, str],
    faith: Dict[str, Any],
    qual: Dict[str, Any],
    *,
    strategy: int,
    corpus_id: str,
    publication_number: str,
    question_language: str,
    answer_language: str,
    context_language: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "chebi_id": entry["chebi_id"],
        "concept_name": entry["name"],
        "mode": MODE_TECHNICAL,
        "strategy": strategy,
        "strategy_name": STRATEGY_NAMES.get(strategy, str(strategy)),
        "corpus_id": corpus_id,
        "publication_number": publication_number,
        "question_language": question_language,
        "context_language": context_language,
        "question": qa["question"],
        "answer": qa["answer"],
        "answer_language": answer_language,
        "question_type": qa.get("question_type", ""),
        "faith_grounding": faith["grounding"],
        "faith_precision": faith["precision"],
        "faith_numerical_fidelity": faith["numerical_fidelity"],
        "faith_overall": faith["overall"],
        "qual_search_bar_realism": qual["search_bar_realism"],
        "qual_specificity": qual["specificity"],
        "qual_phrasing_economy": qual["phrasing_economy"],
        "qual_focus": qual["focus"],
        "qual_linguistic_quality": qual["linguistic_quality"],
        "qual_overall": qual["overall"],
        "qual_failure_type": qual.get("failure_type", "none"),
        "total_score": _compute_total_score(faith, qual, MODE_TECHNICAL),
        "gold_publication_count": entry.get("n_gold", len(entry.get("gold", []))),
    }
    return row


def _process_item(
    client,
    entry: Dict[str, Any],
    doc_rows: List[Dict[str, Any]],
    pub: str,
    target_lang: str,
    name_set: Dict[str, Any],
    aliases: Sequence[str],
    *,
    strategy: int,
    model: str,
) -> Optional[Dict[str, Any]]:
    """Generate + grade ONE (concept, gold publication, target language) item.

    All language variants of *doc_rows* (the same patent in en/de/fr/es/zh) are
    passed to the generator together; the question is written in *target_lang*.
    Returns the output row, or None if no valid question/answer could be formed.
    """
    all_passages = _build_all_passages_text(doc_rows)
    if not all_passages.strip():
        return None
    context_row, _ = _pick_context(doc_rows, target_lang)
    context_languages = _serialize_context_languages(doc_rows)

    answer, answer_lang, _grounded = _pick_answer(name_set, target_lang, all_passages)
    if not answer:
        return None

    # Retry generation on the same document if the model doesn't respond
    # (empty question) or errors, up to _MAX_GEN_RETRIES attempts.
    gen = None
    for attempt in range(1, _MAX_GEN_RETRIES + 1):
        try:
            candidate = generate_concept_query(
                client, all_passages, entry["name"], aliases, target_lang, model=model
            )
        except Exception as exc:
            tqdm.write(
                f"  {entry['chebi_id']} [{target_lang}]: generation error "
                f"(attempt {attempt}/{_MAX_GEN_RETRIES}): {exc}"
            )
            continue
        if candidate["question"]:
            gen = candidate
            break
    if gen is None:
        tqdm.write(
            f"  {entry['chebi_id']} [{target_lang}]: no question after "
            f"{_MAX_GEN_RETRIES} attempts; skipped"
        )
        return None

    qa_pair = {"question": gen["question"], "answer": answer}
    try:
        faith = grade_faithfulness_single(client, all_passages, qa_pair, model=model)
        qual = grade_quality_single(client, all_passages, qa_pair, model=model)
    except Exception as exc:
        tqdm.write(f"  {entry['chebi_id']} [{target_lang}]: grading error: {exc}")
        return None

    row = _build_row(
        entry,
        {"question": gen["question"], "answer": answer, "question_type": gen["question_type"]},
        faith,
        qual,
        strategy=strategy,
        corpus_id=context_row.get("id", ""),
        publication_number=pub,
        question_language=target_lang,
        answer_language=answer_lang,
        context_language=context_languages,
    )
    tqdm.write(
        f"  {entry['chebi_id']} ({entry['name']}) [{target_lang}]: "
        f"ok (total={row['total_score']})"
    )
    return row


def _groundable_concepts(
    pub_concepts: List[Dict[str, Any]], passages: str
) -> List[Dict[str, Any]]:
    """Concepts (gold for this pub) whose name actually appears in the passages."""
    out: List[Dict[str, Any]] = []
    for entry in pub_concepts:
        names = _all_aliases(entry.get("name_set", {}))
        if names and any(contains_name(passages, nm) for nm in names):
            out.append(entry)
    return out


def _build_document_plan(
    entries: List[Dict[str, Any]],
    groups: Dict[str, List[Dict[str, Any]]],
    *,
    per_lang: Optional[int],
    strategy: int,
    rng: random.Random,
) -> List[Tuple[Dict[str, Any], str, str]]:
    """Select unique documents and build a (concept, pub, query language) work list.

    Each selected publication is paired with a single answer-concept (a concept it
    is gold for whose name appears in the passages) and one query per language
    returned by ``strategy``: a single language for strategies 1-3, all five for
    strategy 4 ("all"). Selection is balanced by *source* language: each language L in
    ``ALL_LANGS`` should be present in at least ``per_lang`` selected documents (a
    multilingual document counts toward every language it contains), so the total
    number of documents may be below ``5 * per_lang``. A language with too few
    eligible documents is capped and warned. When ``per_lang`` is None, every
    eligible document is selected (no cap).
    """
    langs = list(ALL_LANGS)

    # Invert concept -> gold into pub -> [concept entries this pub is gold for].
    pub_concepts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        for pub in entry.get("gold", []):
            if pub in groups:
                pub_concepts[pub].append(entry)

    # Per candidate pub: present languages + the concepts groundable in its text.
    present_by_pub: Dict[str, Set[str]] = {}
    groundable_by_pub: Dict[str, List[Dict[str, Any]]] = {}
    candidates: List[str] = []
    for pub, concepts in pub_concepts.items():
        rows = groups[pub]
        present = {r.get("language") for r in rows} & set(langs)
        if not present:
            continue
        passages = _build_all_passages_text(rows)
        if not passages.strip():
            continue
        grounded = _groundable_concepts(concepts, passages)
        if not grounded:
            continue
        present_by_pub[pub] = present
        groundable_by_pub[pub] = grounded
        candidates.append(pub)

    rng.shuffle(candidates)

    if per_lang is None:
        selected = list(candidates)
    else:
        # Assign up to ``per_lang`` DISTINCT documents to each language (each
        # document counts toward exactly one language), so the total is ~``limit``
        # (= 5 * per_lang) rather than collapsing when documents are multilingual.
        # Process scarcer languages first so they claim their few documents before
        # abundant ones (en/fr) can take them; ``candidates`` is already shuffled,
        # so the per-language pick is uniform-random.
        selected_set: Set[str] = set()
        langs_by_scarcity = sorted(
            langs, key=lambda L: sum(1 for pub in candidates if L in present_by_pub[pub])
        )
        for L in langs_by_scarcity:
            picked = 0
            for pub in candidates:
                if picked >= per_lang:
                    break
                if pub in selected_set or L not in present_by_pub[pub]:
                    continue
                selected_set.add(pub)
                picked += 1
        selected = [pub for pub in candidates if pub in selected_set]
        # Warn on each language's true coverage in the final set (a document
        # assigned to one language may also exist in others).
        for L in langs:
            have = sum(1 for pub in selected if L in present_by_pub[pub])
            if have < per_lang:
                print(f"  [balance] {L}: only {have}/{per_lang} eligible documents (selected all available)")

    plan: List[Tuple[Dict[str, Any], str, str]] = []
    for pub in selected:
        grounded = list(groundable_by_pub[pub])
        rng.shuffle(grounded)
        entry = grounded[0]
        # One query per language returned by the strategy: a single language for
        # strategies 1-3, all five for strategy 4 ("all"). Document selection
        # (which/how many docs) is independent of this.
        for target_lang in pick_target_languages(strategy, list(present_by_pub[pub])):
            plan.append((entry, pub, target_lang))
    return plan


def _iter_results(
    plan: List[Tuple[Dict[str, Any], str, str]],
    groups: Dict[str, List[Dict[str, Any]]],
    *,
    strategy: int,
    model: str,
    workers: int,
) -> Iterator[Dict[str, Any]]:
    """Yield one graded query row per planned (concept, pub, query language) item,
    as soon as it is ready, so the caller can persist results incrementally."""
    client = _get_client()

    def work(item: Tuple[Dict[str, Any], str, str]) -> Optional[Dict[str, Any]]:
        entry, pub, target_lang = item
        name_set = entry.get("name_set", {})
        return _process_item(
            client, entry, groups[pub], pub, target_lang, name_set, _all_aliases(name_set),
            strategy=strategy, model=model,
        )

    if workers > 1 and plan:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(work, item) for item in plan]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Concept Q&A", unit="query"):
                row = future.result()
                if row is not None:
                    yield row
    else:
        for item in tqdm(plan, desc="Concept Q&A", unit="query"):
            row = work(item)
            if row is not None:
                yield row


def _output_fieldnames() -> List[str]:
    return [
        "chebi_id", "concept_name", "mode", "strategy", "strategy_name",
        "corpus_id", "publication_number", "question_language", "context_language",
        "question", "answer", "answer_language", "question_type",
        *FAITHFULNESS_FIELDS, *TECHNICAL_QUALITY_FIELDS,
        "qual_failure_type", "total_score", "gold_publication_count",
    ]


def run_concept_qa(
    alias_json: Path,
    corpus_path: Path,
    output_path: Path,
    *,
    strategy: int = STRATEGY_RANDOM_ANY,
    model: str = DEFAULT_MODEL,
    seed: int = 42,
    limit: Optional[int] = None,
    workers: int = 1,
) -> int:
    """Generate concept-centric technical queries (one per selected document) and
    write them to ``output_path``.

    Documents are selected balanced by *source* language: with ``limit`` set, each
    of the five languages should appear in at least ``limit // 5`` selected
    documents (soft cap; a language with too few eligible documents is warned, and
    the document total may be below ``limit`` since a multilingual document counts
    for every language it contains). Without ``limit`` every eligible document is
    used. Each document's query is grounded on all of its language variants. The
    query language is chosen by ``strategy``; strategy 4 ("all") emits one query
    per language for each document. Rows are written to ``output_path`` as they are
    generated (incrementally flushed).
    """
    entries = load_alias_graph(alias_json)
    groups = load_multilingual_corpus(corpus_path)
    random.seed(seed)

    per_lang = max(1, limit // len(ALL_LANGS)) if limit is not None else None
    plan = _build_document_plan(
        entries, groups, per_lang=per_lang, strategy=strategy, rng=random.Random(seed)
    )
    n_docs = len({pub for _, pub, _ in plan})
    target = f"~{per_lang}/language" if per_lang is not None else "all eligible"
    print(
        f"Concept-query QA: {n_docs} documents ({target}) -> {len(plan)} queries, "
        f"strategy={STRATEGY_NAMES.get(strategy, strategy)}, model={model}, workers={workers}"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_output_fieldnames())
        writer.writeheader()
        fh.flush()
        for row in _iter_results(plan, groups, strategy=strategy, model=model, workers=workers):
            writer.writerow(row)
            fh.flush()
            written += 1

    print(f"\nWrote {written} concept-query rows -> {output_path}")
    return written
