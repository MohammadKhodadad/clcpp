"""
Multilingual Q&A generation for documents that already exist in multiple
languages.  No translation step needed — questions are generated directly
in the target language.

Two generation modes:
  - technical  : fact-extraction questions (parameter, material, outcome, …)
  - semantic   : concept/retrieval questions (problem, solution, application)

Four strategies for choosing the question language:
  1  RANDOM_ANY       — pick a random language from {en, de, fr, es}
  2  RANDOM_MISSING   — pick a random language NOT in the document's languages
  3  RANDOM_EXISTING  — pick a random language that IS in the document's languages
  4  ALL              — generate a question for ALL 4 languages
"""

from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_LANGS = ["en", "de", "fr", "es", "zh"]
LANG_NAMES = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "zh": "Chinese"}

MODE_TECHNICAL = "technical"
MODE_SEMANTIC = "semantic"

STRATEGY_RANDOM_ANY = 1
STRATEGY_RANDOM_MISSING = 2
STRATEGY_RANDOM_EXISTING = 3
STRATEGY_ALL = 4

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_GENERATION_REASONING_EFFORT = "medium"

STRATEGY_NAMES = {
    STRATEGY_RANDOM_ANY: "random_any",
    STRATEGY_RANDOM_MISSING: "random_missing",
    STRATEGY_RANDOM_EXISTING: "random_existing",
    STRATEGY_ALL: "all",
}

# Sub-score field names per mode, used for CSV columns and total calculation
FAITHFULNESS_FIELDS = ["faith_grounding", "faith_precision", "faith_numerical_fidelity", "faith_overall"]

TECHNICAL_QUALITY_FIELDS = [
    "qual_search_bar_realism", "qual_specificity", "qual_phrasing_economy",
    "qual_focus", "qual_linguistic_quality", "qual_overall",
]
SEMANTIC_QUALITY_FIELDS = [
    "qual_search_realism", "qual_lexical_distance", "qual_conceptual_framing",
    "qual_retrievability", "qual_linguistic_quality", "qual_overall",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parent


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in .env for Q&A generation.")
    return OpenAI(api_key=api_key)


def _parse_json_response(text: str) -> Any:
    """Parse a JSON object or array from a model response."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def _compute_faith_overall(faith: Dict[str, Any]) -> int:
    """Compute the faithfulness aggregate from verifier sub-scores."""
    return sum(
        int(faith.get(key, 0))
        for key in ("grounding", "precision", "numerical_fidelity")
    )


def _compute_quality_overall(qual: Dict[str, Any], mode: str) -> int:
    """Compute the quality aggregate from verifier sub-scores."""
    if mode == MODE_TECHNICAL:
        keys = (
            "search_bar_realism",
            "specificity",
            "phrasing_economy",
            "focus",
            "linguistic_quality",
        )
    else:
        keys = (
            "search_realism",
            "lexical_distance",
            "conceptual_framing",
            "retrievability",
            "linguistic_quality",
        )

    return sum(int(qual.get(key, 0)) for key in keys)


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def load_multilingual_corpus(
    corpus_path: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load the multilingual corpus CSV and group rows by publication_number.

    Returns {publication_number: [row_dict, ...]}.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with Path(corpus_path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            groups[row["publication_number"]].append(dict(row))
    return dict(groups)


# ---------------------------------------------------------------------------
# Strategy: pick target language(s)
# ---------------------------------------------------------------------------


def pick_target_languages(
    strategy: int,
    available_langs: list[str],
    langs: Optional[list[str]] = None,
) -> list[str]:
    """
    Given the set of languages a document exists in and a strategy number,
    return the list of target language(s) to generate questions in.

    ``langs`` is the language universe to draw from (defaults to ``ALL_LANGS``,
    the 5-language Google Patents set). Pass a different list — e.g.
    ``["en", "de", "fr"]`` for EPO — to restrict generation to a corpus's
    actual languages.
    """
    langs = langs if langs is not None else ALL_LANGS
    available_set = set(available_langs) & set(langs)
    missing = [l for l in langs if l not in available_set]

    if strategy == STRATEGY_RANDOM_ANY:
        return [random.choice(langs)]

    if strategy == STRATEGY_RANDOM_MISSING:
        if not missing:
            return [random.choice(langs)]
        return [random.choice(missing)]

    if strategy == STRATEGY_RANDOM_EXISTING:
        existing = [l for l in langs if l in available_set]
        if not existing:
            return [random.choice(langs)]
        return [random.choice(existing)]

    if strategy == STRATEGY_ALL:
        return list(langs)

    raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


def _pick_context(
    rows: List[Dict[str, Any]],
    target_lang: str,
) -> Tuple[Dict[str, Any], str]:
    """
    Pick the best context row for generating a question in *target_lang*.

    Prefers the row whose language matches target_lang.  Falls back to any
    available row (preferring English if present).

    Returns (chosen_row, context_text).
    """
    by_lang = {r["language"]: r for r in rows}

    if target_lang in by_lang:
        row = by_lang[target_lang]
        return row, row.get("context") or row.get("abstract") or row.get("title", "")

    for fallback in ["en"] + list(by_lang.keys()):
        if fallback in by_lang:
            row = by_lang[fallback]
            return row, row.get("context") or row.get("abstract") or row.get("title", "")

    row = rows[0]
    return row, row.get("context") or row.get("abstract") or row.get("title", "")


def _build_all_passages_text(rows: List[Dict[str, Any]]) -> str:
    """Build a string with all language passages for verifier context."""
    parts: list[str] = []
    for r in rows:
        lang = r.get("language", "?")
        ctx = r.get("context") or r.get("abstract") or r.get("title", "")
        if ctx.strip():
            parts.append(f"[{lang.upper()}] Passage:\n{ctx.strip()}")
    return "\n\n".join(parts)


def _serialize_context_languages(rows: List[Dict[str, Any]]) -> str:
    """Serialize all languages available for a document into one CSV cell."""
    seen: set[str] = set()
    ordered_langs: list[str] = []

    for lang in ALL_LANGS:
        if any(r.get("language") == lang for r in rows):
            seen.add(lang)
            ordered_langs.append(lang)

    for r in rows:
        lang = (r.get("language") or "").strip()
        if lang and lang not in seen:
            seen.add(lang)
            ordered_langs.append(lang)

    return ",".join(ordered_langs)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_prompt_cache: Dict[str, str] = {}


def _load_file(path: Path) -> str:
    """Load and cache a prompt file."""
    key = str(path)
    if key in _prompt_cache:
        return _prompt_cache[key]
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    _prompt_cache[key] = text
    return text


def _load_generation_prompt(mode: str, lang: str) -> str:
    """Load the generation prompt for a given mode and language."""
    if mode == MODE_TECHNICAL:
        return _load_file(_BASE_DIR / "technical_question_generation_prompts" / f"{lang}.txt")
    elif mode == MODE_SEMANTIC:
        return _load_file(_BASE_DIR / "semantic_retrieval_question_generation_prompts" / f"{lang}.txt")
    raise ValueError(f"Unknown mode: {mode}")


def _load_faithfulness_prompt() -> str:
    return _load_file(_BASE_DIR / "faithfulness_prompt" / "faithfulness_prompt.txt")


def _load_quality_prompt(mode: str) -> str:
    if mode == MODE_TECHNICAL:
        return _load_file(_BASE_DIR / "technical_quality_verifier_prompt" / "verifier.txt")
    elif mode == MODE_SEMANTIC:
        return _load_file(_BASE_DIR / "semantic_retrieval_quality_verifier_prompt" / "verifier.txt")
    raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Generation: produce 3 Q&A pairs
# ---------------------------------------------------------------------------


def generate_qa_batch(
    client: OpenAI,
    all_passages: str,
    target_lang: str,
    mode: str,
    *,
    model: str = DEFAULT_MODEL,
) -> List[Dict[str, str]]:
    """
    Generate THREE Q&A pairs in *target_lang* from the passages.

    The prompt already ends with the preamble for passages, so
    all_passages is appended directly as user content.

    Returns a list of 3 dicts. Fields depend on mode:
      technical: {question, answer, question_type}
      semantic:  {question, answer, framing}
    """
    prompt = _load_generation_prompt(mode, target_lang)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": all_passages},
        ],
        reasoning_effort=DEFAULT_GENERATION_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")

    if isinstance(data, dict):
        data = [data]

    results: List[Dict[str, str]] = []
    for item in data[:3]:
        row: Dict[str, str] = {
            "question": str(item.get("question", "")).strip(),
            "answer": str(item.get("answer", "")).strip(),
        }
        if mode == MODE_TECHNICAL:
            row["question_type"] = str(item.get("question_type", "other")).strip()
        else:
            row["framing"] = str(item.get("framing", "other")).strip()
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Faithfulness grader (3 pairs at once)
# ---------------------------------------------------------------------------


def grade_faithfulness(
    client: OpenAI,
    all_passages: str,
    qa_pairs: List[Dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    """
    Grade 3 Q&A pairs for faithfulness against the passages.

    Returns list of 3 dicts with keys:
      grounding, precision, numerical_fidelity, overall, reason
    """
    prompt = _load_faithfulness_prompt()

    candidates = "\n\n".join(
        f"Candidate {i}:\n  Question: {qa['question']}\n  Answer: {qa['answer']}"
        for i, qa in enumerate(qa_pairs)
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"{all_passages}\n\n{candidates}",
            },
        ],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")

    if isinstance(data, dict):
        data = [data]

    # Ensure we have exactly 3, sorted by index
    results = sorted(data[:3], key=lambda x: x.get("index", 0))

    # Normalise to guarantee keys exist
    normalised: List[Dict[str, Any]] = []
    for item in results:
        row = {
            "grounding": int(item.get("grounding", 1)),
            "precision": int(item.get("precision", 1)),
            "numerical_fidelity": int(item.get("numerical_fidelity", 1)),
            "reason": str(item.get("reason", "")).strip(),
        }
        row["overall"] = _compute_faith_overall(row)
        normalised.append(row)
    # Pad if fewer than 3
    while len(normalised) < 3:
        row = {
            "grounding": 1,
            "precision": 1,
            "numerical_fidelity": 1,
            "reason": "missing",
        }
        row["overall"] = _compute_faith_overall(row)
        normalised.append(row)
    return normalised


# ---------------------------------------------------------------------------
# Quality grader (3 questions at once, mode-specific)
# ---------------------------------------------------------------------------


def grade_quality(
    client: OpenAI,
    all_passages: str,
    qa_pairs: List[Dict[str, str]],
    mode: str,
    *,
    model: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    """
    Grade 3 questions for quality using the mode-specific verifier.

    Returns list of 3 dicts with mode-specific score keys + overall,
    failure_type, reason.
    """
    prompt = _load_quality_prompt(mode)

    candidates = "\n\n".join(
        f"Candidate {i}:\n  Question: {qa['question']}\n  Answer: {qa['answer']}"
        for i, qa in enumerate(qa_pairs)
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"{all_passages}\n\n{candidates}",
            },
        ],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")

    if isinstance(data, dict):
        data = [data]

    results = sorted(data[:3], key=lambda x: x.get("index", 0))

    normalised: List[Dict[str, Any]] = []
    if mode == MODE_TECHNICAL:
        for item in results:
            row = {
                "search_bar_realism": int(item.get("search_bar_realism", 1)),
                "specificity": int(item.get("specificity", 1)),
                "phrasing_economy": int(item.get("phrasing_economy", 1)),
                "focus": int(item.get("focus", 1)),
                "linguistic_quality": int(item.get("linguistic_quality", 1)),
                "failure_type": str(item.get("failure_type", "none")).strip(),
                "reason": str(item.get("reason", "")).strip(),
            }
            row["overall"] = _compute_quality_overall(row, mode)
            normalised.append(row)
        default = {
            "search_bar_realism": 1,
            "specificity": 1,
            "phrasing_economy": 1,
            "focus": 1,
            "linguistic_quality": 1,
            "failure_type": "missing",
            "reason": "missing",
        }
    else:  # semantic
        for item in results:
            row = {
                "search_realism": int(item.get("search_realism", 1)),
                "lexical_distance": int(item.get("lexical_distance", 1)),
                "conceptual_framing": int(item.get("conceptual_framing", 1)),
                "retrievability": int(item.get("retrievability", 1)),
                "linguistic_quality": int(item.get("linguistic_quality", 1)),
                "failure_type": str(item.get("failure_type", "none")).strip(),
                "reason": str(item.get("reason", "")).strip(),
            }
            row["overall"] = _compute_quality_overall(row, mode)
            normalised.append(row)
        default = {
            "search_realism": 1,
            "lexical_distance": 1,
            "conceptual_framing": 1,
            "retrievability": 1,
            "linguistic_quality": 1,
            "failure_type": "missing",
            "reason": "missing",
        }

    while len(normalised) < 3:
        row = dict(default)
        row["overall"] = _compute_quality_overall(row, mode)
        normalised.append(row)
    return normalised


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------


def _compute_total_score(faith: Dict[str, Any], qual: Dict[str, Any], mode: str) -> int:
    """Sum the faithfulness and quality aggregates."""
    return int(faith.get("overall", 0)) + int(qual.get("overall", 0))


def _build_output_row(
    qa: Dict[str, str],
    faith: Dict[str, Any],
    qual: Dict[str, Any],
    mode: str,
    *,
    corpus_id: str,
    publication_number: str,
    question_language: str,
    context_language: str,
) -> Dict[str, Any]:
    """Build a single output CSV row from generation + grading results."""
    row: Dict[str, Any] = {
        "corpus_id": corpus_id,
        "publication_number": publication_number,
        "question_language": question_language,
        "context_language": context_language,
        "question": qa["question"],
        "answer": qa["answer"],
    }

    # Mode-specific category field
    if mode == MODE_TECHNICAL:
        row["question_type"] = qa.get("question_type", "")
    else:
        row["framing"] = qa.get("framing", "")

    # Faithfulness scores
    row["faith_grounding"] = faith["grounding"]
    row["faith_precision"] = faith["precision"]
    row["faith_numerical_fidelity"] = faith["numerical_fidelity"]
    row["faith_overall"] = faith["overall"]

    # Quality scores
    if mode == MODE_TECHNICAL:
        row["qual_search_bar_realism"] = qual["search_bar_realism"]
        row["qual_specificity"] = qual["specificity"]
        row["qual_phrasing_economy"] = qual["phrasing_economy"]
        row["qual_focus"] = qual["focus"]
        row["qual_linguistic_quality"] = qual["linguistic_quality"]
        row["qual_overall"] = qual["overall"]
    else:
        row["qual_search_realism"] = qual["search_realism"]
        row["qual_lexical_distance"] = qual["lexical_distance"]
        row["qual_conceptual_framing"] = qual["conceptual_framing"]
        row["qual_retrievability"] = qual["retrievability"]
        row["qual_linguistic_quality"] = qual["linguistic_quality"]
        row["qual_overall"] = qual["overall"]

    row["qual_failure_type"] = qual.get("failure_type", "none")
    row["total_score"] = _compute_total_score(faith, qual, mode)

    return row


# ---------------------------------------------------------------------------
# CSV field names
# ---------------------------------------------------------------------------


def _get_fieldnames(mode: str) -> List[str]:
    base = [
        "corpus_id", "publication_number", "question_language",
        "context_language", "question", "answer",
    ]
    if mode == MODE_TECHNICAL:
        base.append("question_type")
        base.extend(FAITHFULNESS_FIELDS)
        base.extend(TECHNICAL_QUALITY_FIELDS)
    else:
        base.append("framing")
        base.extend(FAITHFULNESS_FIELDS)
        base.extend(SEMANTIC_QUALITY_FIELDS)
    base.append("qual_failure_type")
    base.append("total_score")
    return base


# ---------------------------------------------------------------------------
# Process one document group
# ---------------------------------------------------------------------------


def _process_document(
    pub_num: str,
    rows: List[Dict[str, Any]],
    *,
    strategy: int,
    mode: str,
    model: str,
) -> List[Dict[str, Any]]:
    """
    Generate and grade Q&A triplets for one publication.

    For each target language:
      1. Generate 3 Q&A pairs
      2. Grade all 3 for faithfulness (1 call)
      3. Grade all 3 for quality (1 call)
      4. Compute total scores and sort best-first

    Returns a list of output rows (3 per target language, sorted by score).
    """
    available_langs = [r["language"] for r in rows]
    target_langs = pick_target_languages(strategy, available_langs)
    client = _get_client()
    all_passages = _build_all_passages_text(rows)
    context_languages = _serialize_context_languages(rows)
    results: List[Dict[str, Any]] = []

    for target_lang in target_langs:
        context_row, context_text = _pick_context(rows, target_lang)
        if not context_text.strip():
            tqdm.write(f"  {pub_num} [{target_lang}]: skipped (empty context)")
            continue

        # Step 1: Generate 3 Q&A pairs (all passages sent to generator)
        try:
            qa_pairs = generate_qa_batch(
                client, all_passages, target_lang, mode, model=model,
            )
        except Exception as exc:
            tqdm.write(f"  {pub_num} [{target_lang}]: generation error: {exc}")
            continue

        if len(qa_pairs) < 3:
            tqdm.write(
                f"  {pub_num} [{target_lang}]: only {len(qa_pairs)} questions generated, skipping"
            )
            continue

        # Step 2: Faithfulness grading (all 3 at once)
        try:
            faith_grades = grade_faithfulness(
                client, all_passages, qa_pairs, model=model,
            )
        except Exception as exc:
            tqdm.write(f"  {pub_num} [{target_lang}]: faithfulness grading error: {exc}")
            continue

        # Step 3: Quality grading (all 3 at once)
        try:
            qual_grades = grade_quality(
                client, all_passages, qa_pairs, mode, model=model,
            )
        except Exception as exc:
            tqdm.write(f"  {pub_num} [{target_lang}]: quality grading error: {exc}")
            continue

        # Step 4: Build output rows and sort by total_score (best first)
        doc_rows: List[Dict[str, Any]] = []
        for i in range(3):
            row = _build_output_row(
                qa_pairs[i],
                faith_grades[i],
                qual_grades[i],
                mode,
                corpus_id=context_row.get("id", ""),
                publication_number=pub_num,
                question_language=target_lang,
                context_language=context_languages,
            )
            doc_rows.append(row)

        # Sort by total_score descending (best first)
        doc_rows.sort(key=lambda r: r["total_score"], reverse=True)
        results.extend(doc_rows)

        best_score = doc_rows[0]["total_score"]
        cat_field = "question_type" if mode == MODE_TECHNICAL else "framing"
        tqdm.write(
            f"  {pub_num} [{target_lang}]: ok (best={best_score}, "
            f"{cat_field}={doc_rows[0].get(cat_field, '?')})"
        )

    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_multilingual_qa_pipeline(
    corpus_path: Path,
    output_path: Path,
    *,
    mode: str = MODE_TECHNICAL,
    strategy: int = STRATEGY_RANDOM_ANY,
    model: str = DEFAULT_MODEL,
    seed: int = 42,
    limit: Optional[int] = None,
) -> int:
    """
    Generate Q&A pairs from a multilingual corpus using the given strategy
    and mode.

    Parameters
    ----------
    corpus_path : Path to multilingual_corpus.csv
    output_path : Path for the output QAC CSV
    mode        : 'technical' or 'semantic'
    strategy    : 1=random_any, 2=random_missing, 3=random_existing, 4=all
    model       : OpenAI model name
    seed        : random seed for reproducibility
    limit       : if set, only process this many documents (for testing)

    Returns number of QAC rows written.
    """
    random.seed(seed)
    groups = load_multilingual_corpus(corpus_path)
    pub_nums = list(groups.keys())

    if limit and limit < len(pub_nums):
        pub_nums = pub_nums[:limit]

    print(
        f"Multilingual QA generation: {len(pub_nums)} documents, "
        f"mode={mode}, strategy={STRATEGY_NAMES.get(strategy, strategy)}, "
        f"model={model}"
    )

    fieldnames = _get_fieldnames(mode)
    all_rows: List[Dict[str, Any]] = []
    progress = tqdm(pub_nums, desc="Generate Q&A", unit="doc")
    for pub_num in progress:
        rows = groups[pub_num]
        try:
            results = _process_document(
                pub_num,
                rows,
                strategy=strategy,
                mode=mode,
                model=model,
            )
            all_rows.extend(results)
        except Exception as exc:
            tqdm.write(f"  {pub_num}: error: {exc}")

    # Write full output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} QAC rows -> {output_path}")

    # Write best-only output (top question per publication + language)
    best_rows: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in all_rows:
        key = (row["publication_number"], row["question_language"])
        if key not in seen:
            seen.add(key)
            best_rows.append(row)

    best_path = output_path.with_name(
        output_path.stem + "_best" + output_path.suffix
    )
    with best_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_rows)

    print(f"Wrote {len(best_rows)} best QAC rows -> {best_path}")
    return len(all_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Generate multilingual Q&A from documents existing in multiple languages. "
            "No translation step — questions are generated directly in the target language."
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/google_patents/multilingual_corpus.csv"),
        help="Path to multilingual corpus CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output QAC CSV path. Defaults to "
            "data/google_patents/qac/{mode}_{strategy}_qac.csv"
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=MODE_TECHNICAL,
        choices=[MODE_TECHNICAL, MODE_SEMANTIC],
        help="Generation mode: 'technical' (default) or 'semantic'",
    )
    parser.add_argument(
        "--strategy",
        type=int,
        default=STRATEGY_RANDOM_ANY,
        choices=[
            STRATEGY_RANDOM_ANY,
            STRATEGY_RANDOM_MISSING,
            STRATEGY_RANDOM_EXISTING,
            STRATEGY_ALL,
        ],
        help=(
            "Language selection strategy: "
            "1=random from {en,de,fr,es}, "
            "2=random from languages NOT in the document, "
            "3=random from languages IN the document, "
            "4=all 4 languages (default: 1)"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only this many documents (for testing)",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        strategy_name = STRATEGY_NAMES.get(args.strategy, str(args.strategy))
        output_path = Path(f"data/google_patents/qac/{args.mode}_{strategy_name}_qac.csv")

    run_multilingual_qa_pipeline(
        corpus_path=args.corpus,
        output_path=output_path,
        mode=args.mode,
        strategy=args.strategy,
        model=args.model,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
