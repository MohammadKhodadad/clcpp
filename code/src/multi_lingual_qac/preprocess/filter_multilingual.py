"""
Filter preprocessed patent CSVs to find documents that appear in at least 2
of the target languages (en, es, de, fr) and write them to a single output CSV.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional


TARGET_LANGS = ["en", "es", "de", "fr"]
MIN_LANG_COUNT = 2


def find_multilingual_documents(
    preprocessed_dir: Path,
    output_path: Path,
    *,
    languages: Optional[list[str]] = None,
    min_lang_count: int = MIN_LANG_COUNT,
) -> dict[str, object]:
    """
    Scan per-language CSVs, find publications appearing in >= min_lang_count
    languages, and write all their rows to a single output CSV.

    Returns a summary dict with counts.
    """
    languages = languages or TARGET_LANGS
    preprocessed_dir = Path(preprocessed_dir)
    output_path = Path(output_path)

    # Pass 1: collect publication_number -> set of languages
    pub_langs: dict[str, set[str]] = defaultdict(set)
    for lang in languages:
        csv_path = preprocessed_dir / f"{lang}.csv"
        if not csv_path.exists():
            print(f"  Warning: {csv_path} not found, skipping {lang}")
            continue
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pub = row["publication_number"]
                pub_langs[pub].add(lang)

    multilingual_pubs = {
        pub for pub, langs in pub_langs.items() if len(langs) >= min_lang_count
    }
    print(f"Found {len(multilingual_pubs)} publications in >= {min_lang_count} of {languages}")

    # Pass 2: collect all rows for those publications
    fieldnames = [
        "id", "language", "title", "abstract", "description",
        "first_claim", "context", "publication_number",
        "country_code", "publication_date", "source", "ipc_codes",
    ]
    rows_out: list[dict[str, str]] = []
    for lang in languages:
        csv_path = preprocessed_dir / f"{lang}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["publication_number"] in multilingual_pubs:
                    rows_out.append(row)

    # Sort by publication_number then language for readability
    rows_out.sort(key=lambda r: (r["publication_number"], r["language"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)

    # Build per-language counts
    lang_counts: dict[str, int] = defaultdict(int)
    for r in rows_out:
        lang_counts[r["language"]] += 1

    # Count by how many languages each publication covers
    coverage: dict[int, int] = defaultdict(int)
    for pub in multilingual_pubs:
        coverage[len(pub_langs[pub])] += 1

    summary = {
        "total_publications": len(multilingual_pubs),
        "total_rows": len(rows_out),
        "per_language": dict(lang_counts),
        "coverage_distribution": {f"{k}_langs": v for k, v in sorted(coverage.items())},
        "output_path": str(output_path),
    }

    print(f"Wrote {len(rows_out)} rows ({len(multilingual_pubs)} publications) -> {output_path}")
    print(f"  Per language: {dict(lang_counts)}")
    print(f"  Coverage: {dict(sorted(coverage.items()))} (num_languages -> num_publications)")

    return summary


def main() -> None:
    """CLI entry point: run from project root."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter preprocessed patents to multilingual-only documents."
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("data/google_patents/preprocessed"),
        help="Directory with per-language CSVs (default: data/google_patents/preprocessed)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/google_patents/multilingual_corpus.csv"),
        help="Output CSV path (default: data/google_patents/multilingual_corpus.csv)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=TARGET_LANGS,
        help=f"Languages to consider (default: {TARGET_LANGS})",
    )
    parser.add_argument(
        "--min-langs",
        type=int,
        default=MIN_LANG_COUNT,
        help=f"Minimum number of languages a document must appear in (default: {MIN_LANG_COUNT})",
    )
    args = parser.parse_args()

    find_multilingual_documents(
        preprocessed_dir=args.preprocessed_dir,
        output_path=args.output,
        languages=args.languages,
        min_lang_count=args.min_langs,
    )


if __name__ == "__main__":
    main()
