from __future__ import annotations

import csv
import random
from pathlib import Path
import sys
from typing import Any, Dict, List

from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from multilingual_qa import (
        FAITHFULNESS_FIELDS,
        MODE_SEMANTIC,
        MODE_TECHNICAL,
        SEMANTIC_QUALITY_FIELDS,
        STRATEGY_ALL,
        STRATEGY_NAMES,
        STRATEGY_RANDOM_ANY,
        STRATEGY_RANDOM_EXISTING,
        STRATEGY_RANDOM_MISSING,
        TECHNICAL_QUALITY_FIELDS,
        _process_document,
        load_multilingual_corpus,
    )
else:
    from .multilingual_qa import (
        FAITHFULNESS_FIELDS,
        MODE_SEMANTIC,
        MODE_TECHNICAL,
        SEMANTIC_QUALITY_FIELDS,
        STRATEGY_ALL,
        STRATEGY_NAMES,
        STRATEGY_RANDOM_ANY,
        STRATEGY_RANDOM_EXISTING,
        STRATEGY_RANDOM_MISSING,
        TECHNICAL_QUALITY_FIELDS,
        _process_document,
        load_multilingual_corpus,
    )

STRATEGIES = [
    STRATEGY_RANDOM_ANY,
    STRATEGY_RANDOM_MISSING,
    STRATEGY_RANDOM_EXISTING,
    STRATEGY_ALL,
]


def _allocate_question_quotas(total_questions: int) -> Dict[int, int]:
    """
    Allocate question counts across the four strategies as evenly as possible.

    Strategies 1-3 produce one final question per document. Strategy 4 produces
    four final questions per document, so its quota must be a multiple of 4.
    """
    if total_questions < 4:
        raise ValueError("total_questions must be at least 4")

    best_counts: Dict[int, int] | None = None
    best_score: tuple[int, int] | None = None

    for strategy_4_docs in range(1, total_questions // 4 + 1):
        strategy_4_questions = strategy_4_docs * 4
        remaining = total_questions - strategy_4_questions
        if remaining < 3:
            continue

        base, remainder = divmod(remaining, 3)
        counts = {
            STRATEGY_RANDOM_ANY: base + (1 if remainder >= 1 else 0),
            STRATEGY_RANDOM_MISSING: base + (1 if remainder >= 2 else 0),
            STRATEGY_RANDOM_EXISTING: base,
            STRATEGY_ALL: strategy_4_questions,
        }

        values = list(counts.values())
        spread = max(values) - min(values)
        target_gap = abs(strategy_4_questions - (total_questions / 4))
        score = (spread, int(target_gap * 1000))

        if best_score is None or score < best_score:
            best_score = score
            best_counts = counts

    if best_counts is None:
        raise ValueError(f"Unable to allocate balanced quotas for {total_questions} questions")

    return best_counts


def _build_generation_plan(
    pub_nums: List[str],
    *,
    sample_pool_size: int,
    questions_per_mode: int,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if sample_pool_size > len(pub_nums):
        raise ValueError(
            f"Requested sample_pool_size={sample_pool_size}, but corpus only has {len(pub_nums)} unique documents"
        )

    rng = random.Random(seed)
    sampled_pool = rng.sample(pub_nums, sample_pool_size)
    quotas = _allocate_question_quotas(questions_per_mode)

    docs_needed_per_mode = (
        quotas[STRATEGY_RANDOM_ANY]
        + quotas[STRATEGY_RANDOM_MISSING]
        + quotas[STRATEGY_RANDOM_EXISTING]
        + (quotas[STRATEGY_ALL] // 4)
    )
    total_docs_needed = docs_needed_per_mode * 2
    if total_docs_needed > len(sampled_pool):
        raise ValueError(
            f"Need {total_docs_needed} documents to generate the requested sample, "
            f"but sample_pool_size={sample_pool_size}"
        )

    technical_pool = sampled_pool[: sample_pool_size // 2]
    semantic_pool = sampled_pool[sample_pool_size // 2 :]

    if len(technical_pool) < docs_needed_per_mode or len(semantic_pool) < docs_needed_per_mode:
        raise ValueError(
            "Sample pool split is too small for the requested generation plan. "
            "Increase sample_pool_size."
        )

    plan: List[Dict[str, Any]] = []

    def add_mode_plan(mode: str, candidates: List[str]) -> None:
        cursor = 0
        for strategy in STRATEGIES:
            doc_count = quotas[strategy] if strategy != STRATEGY_ALL else quotas[strategy] // 4
            expected_questions = 1 if strategy != STRATEGY_ALL else 4
            for _ in range(doc_count):
                pub_num = candidates[cursor]
                cursor += 1
                plan.append(
                    {
                        "publication_number": pub_num,
                        "mode": mode,
                        "strategy": strategy,
                        "strategy_name": STRATEGY_NAMES[strategy],
                        "expected_question_count": expected_questions,
                    }
                )

    add_mode_plan(MODE_TECHNICAL, technical_pool)
    add_mode_plan(MODE_SEMANTIC, semantic_pool)

    manifest: List[Dict[str, Any]] = []
    selected_by_pub = {
        row["publication_number"]: row for row in plan
    }
    for index, pub_num in enumerate(sampled_pool, start=1):
        selection = selected_by_pub.get(pub_num)
        manifest.append(
            {
                "pool_rank": index,
                "publication_number": pub_num,
                "selected_for_generation": "yes" if selection else "no",
                "mode": selection["mode"] if selection else "",
                "strategy": selection["strategy"] if selection else "",
                "strategy_name": selection["strategy_name"] if selection else "",
                "expected_question_count": selection["expected_question_count"] if selection else 0,
            }
        )

    return plan, manifest


def _select_best_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the best row per (publication_number, question_language)."""
    best_rows: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for row in rows:
        key = (row["publication_number"], row["question_language"])
        if key not in seen:
            seen.add(key)
            best_rows.append(row)

    return best_rows


def _output_fieldnames() -> List[str]:
    return [
        "mode",
        "strategy",
        "strategy_name",
        "corpus_id",
        "publication_number",
        "question_language",
        "context_language",
        "question",
        "answer",
        "question_type",
        "framing",
        *FAITHFULNESS_FIELDS,
        *TECHNICAL_QUALITY_FIELDS,
        *SEMANTIC_QUALITY_FIELDS,
        "qual_failure_type",
        "total_score",
    ]


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {field: "" for field in _output_fieldnames()}
    normalized.update(row)
    return normalized


def run_balanced_multilingual_qa(
    corpus_path: Path,
    output_path: Path,
    *,
    sample_pool_size: int = 500,
    questions_per_mode: int = 50,
    model: str = "gpt-5-mini",
    seed: int = 42,
    dry_run: bool = False,
) -> int:
    groups = load_multilingual_corpus(corpus_path)
    pub_nums = sorted(groups.keys())
    plan, manifest = _build_generation_plan(
        pub_nums,
        sample_pool_size=sample_pool_size,
        questions_per_mode=questions_per_mode,
        seed=seed,
    )

    print(
        f"Balanced multilingual QA generation from a {sample_pool_size}-document pool: "
        f"{questions_per_mode} technical questions + {questions_per_mode} semantic questions"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_name(output_path.stem + "_manifest" + output_path.suffix)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pool_rank",
                "publication_number",
                "selected_for_generation",
                "mode",
                "strategy",
                "strategy_name",
                "expected_question_count",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)

    print(f"Wrote sample manifest -> {manifest_path}")

    if dry_run:
        print("Dry run only: manifest written, generation skipped.")
        return 0

    fieldnames = _output_fieldnames()
    best_rows_written = 0
    all_rows_written = 0
    all_output_path = output_path.with_name(output_path.stem + "_all_generated" + output_path.suffix)
    progress = tqdm(plan, desc="Generate balanced Q&A", unit="doc")
    with output_path.open("w", encoding="utf-8", newline="") as best_f, all_output_path.open(
        "w", encoding="utf-8", newline=""
    ) as all_f:
        best_writer = csv.DictWriter(best_f, fieldnames=fieldnames)
        all_writer = csv.DictWriter(all_f, fieldnames=fieldnames)
        best_writer.writeheader()
        all_writer.writeheader()
        best_f.flush()
        all_f.flush()

        for item in progress:
            pub_num = item["publication_number"]
            rows = groups[pub_num]
            results = _process_document(
                pub_num,
                rows,
                strategy=item["strategy"],
                mode=item["mode"],
                model=model,
            )

            all_rows: List[Dict[str, Any]] = []
            for row in results:
                row["mode"] = item["mode"]
                row["strategy"] = item["strategy"]
                row["strategy_name"] = item["strategy_name"]
                all_rows.append(_normalize_row(row))

            best_rows = _select_best_rows(results)
            best_output_rows: List[Dict[str, Any]] = []
            for row in best_rows:
                best_output_rows.append(_normalize_row(row))

            if all_rows:
                all_writer.writerows(all_rows)
                all_f.flush()
                all_rows_written += len(all_rows)

            if best_output_rows:
                best_writer.writerows(best_output_rows)
                best_f.flush()
                best_rows_written += len(best_output_rows)

            if len(best_rows) != item["expected_question_count"]:
                tqdm.write(
                    f"  {pub_num} [{item['mode']}/{item['strategy_name']}]: "
                    f"expected {item['expected_question_count']} questions, got {len(best_rows)}"
                )

    print(f"\nWrote {best_rows_written} balanced best-only QAC rows -> {output_path}")
    print(f"Wrote {all_rows_written} balanced all-generated QAC rows -> {all_output_path}")
    return best_rows_written


def main() -> None:
    import argparse

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Generate a balanced multilingual QAC sample with 50 technical and "
            "50 semantic questions from a 500-document pool."
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
        default=Path("data/google_patents/qac/balanced_100_qac.csv"),
        help="Output CSV path for the final 100-question sample",
    )
    parser.add_argument(
        "--sample-pool-size",
        type=int,
        default=500,
        help="How many unique documents to sample from the corpus before selecting generation jobs",
    )
    parser.add_argument(
        "--questions-per-mode",
        type=int,
        default=50,
        help="Number of final questions to generate per mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write only the sampled 500-document manifest and skip API generation",
    )
    args = parser.parse_args()

    run_balanced_multilingual_qa(
        corpus_path=args.corpus,
        output_path=args.output,
        sample_pool_size=args.sample_pool_size,
        questions_per_mode=args.questions_per_mode,
        model=args.model,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
