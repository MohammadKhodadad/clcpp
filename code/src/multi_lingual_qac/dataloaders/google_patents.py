"""
Google Patents Public Datasets (BigQuery) loader for chemistry-related patents.

Uses the official BigQuery public datasets:
- patents-public-data.patents.publications
- patents-public-data.google_patents_research.publications (optional)
- patents-public-data.ebi_surechembl.match (optional, chemistry-specific)

Output: NDJSON with multilingual title_localized, abstract_localized, etc.

Requires: GOOGLE_APPLICATION_CREDENTIALS or gcloud auth, and a Google Cloud project.
"""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm


def clean_text(s: str) -> str:
    """Decode HTML entities and normalize whitespace."""
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Fields that BigQuery returns as repeated RECORDs; need JSON serialization
_RECORD_KEYS = [
    "title_localized",
    "abstract_localized",
    "description_localized",
    "description_localized_html",
    "claims_localized",
    "claims_localized_html",
    "cpc",
    "ipc",
]

DEFAULT_CPC_PREFIXES = ["C", "A61K", "A61P"]
DEFAULT_IPC_PREFIXES = ["C", "A61K", "A61P"]
DEFAULT_LANGS = [
    "en", "de", "fr", "es", "ja", "ko", "zh",
    "ru", "pt", "it", "nl",
    "ar", "fa", "tr", "pl", "hi",
]
MIN_ABSTRACT_WORDS = 50
MIN_ABSTRACT_CHARS = 300
MIN_DESCRIPTION_CHARS = 200
DESCRIPTION_SNIPPET_MAX_CHARS = 1500
FIRST_CLAIM_MAX_CHARS = 1500
PER_LANGUAGE_OVERFETCH_FACTOR = 1.25
PER_LANGUAGE_OVERFETCH_MIN = 10


def sql_list(values: List[str]) -> str:
    return ", ".join(f"'{v}'" for v in values)


def word_count(text: str) -> int:
    """Count whitespace-delimited words in cleaned text."""
    return len((text or "").split())


def min_abstract_chars_for_sql(min_words: int) -> int:
    """Approximate a word-count gate with a cheaper character-count gate."""
    return max(MIN_ABSTRACT_CHARS, min_words * 5)


def build_query(
    *,
    languages: Optional[List[str]] = None,
    cpc_prefixes: Optional[List[str]] = None,
    ipc_prefixes: Optional[List[str]] = None,
    use_surechembl: bool = True,
    use_classification: bool = True,
    require_multilingual: bool = False,
    min_language_count: int = 2,
    limit: Optional[int] = None,
    primary_lang: Optional[str] = None,
    min_primary_abstract_words: Optional[int] = None,
    require_primary_description: bool = False,
    require_primary_claim: bool = False,
    require_any_claim: bool = False,
    start_date: Optional[int] = None,
    end_date: Optional[int] = None,
    country_codes: Optional[List[str]] = None,
) -> str:
    """Build BigQuery SQL for chemistry-related multilingual patents."""
    languages = languages or DEFAULT_LANGS
    cpc_prefixes = cpc_prefixes or DEFAULT_CPC_PREFIXES
    ipc_prefixes = ipc_prefixes or DEFAULT_IPC_PREFIXES

    if not use_surechembl and not use_classification:
        raise ValueError("At least one of use_surechembl or use_classification must be True.")

    lang_sql = sql_list(languages)

    date_filter = ""
    if start_date is not None:
        date_filter += f"\n  AND p.publication_date >= {start_date}"
    if end_date is not None:
        date_filter += f"\n  AND p.publication_date <= {end_date}"

    country_filter = ""
    if country_codes:
        cc_sql = sql_list(country_codes)
        country_filter = f"\n  AND p.country_code IN ({cc_sql})"

    chemistry_predicates: List[str] = []

    if use_classification:
        cpc_preds = " OR ".join(f"STARTS_WITH(c.code, {p!r})" for p in cpc_prefixes)
        ipc_preds = " OR ".join(f"STARTS_WITH(i.code, {p!r})" for p in ipc_prefixes)
        chemistry_predicates.append(
            f"""
            EXISTS (
              SELECT 1
              FROM UNNEST(IFNULL(p.cpc, [])) AS c
              WHERE {cpc_preds}
            )
            """
        )
        chemistry_predicates.append(
            f"""
            EXISTS (
              SELECT 1
              FROM UNNEST(IFNULL(p.ipc, [])) AS i
              WHERE {ipc_preds}
            )
            """
        )

    surechembl_join = ""
    surechembl_flag = "FALSE AS has_surechembl_match"
    if use_surechembl:
        surechembl_join = """
        LEFT JOIN (
          SELECT DISTINCT publication_number
          FROM `patents-public-data.ebi_surechembl.match`
        ) sc
        ON p.publication_number = sc.publication_number
        """
        surechembl_flag = "sc.publication_number IS NOT NULL AS has_surechembl_match"
        chemistry_predicates.append("sc.publication_number IS NOT NULL")

    chemistry_where = " OR ".join(f"({p.strip()})" for p in chemistry_predicates)

    # Filter to patents that have a usable localized abstract for primary_lang
    # before limiting. This lets per-language extraction target documents that
    # are likely to survive later preprocessing.
    primary_lang_filter = ""
    if primary_lang:
        pl = primary_lang.strip().lower()
        min_chars_filter = ""
        if min_primary_abstract_words:
            min_chars_filter = f"""
              AND LENGTH(TRIM(a.text)) >= {min_abstract_chars_for_sql(int(min_primary_abstract_words))}
            """
        description_filter = ""
        if require_primary_description:
            # Disabled for now: many non-English patents in BigQuery have a
            # localized abstract but no localized description.
            # description_filter = f"""
            # AND EXISTS (
            #   SELECT 1
            #   FROM UNNEST(IFNULL(p.description_localized, [])) d
            #   WHERE LOWER(COALESCE(d.language, '')) = {pl!r}
            #     AND d.text IS NOT NULL
            #     AND LENGTH(TRIM(d.text)) >= {MIN_DESCRIPTION_CHARS}
            # )
            # """
            pass
        claim_filter = ""
        if require_primary_claim:
            claim_filter = f"""
        AND (
          EXISTS (
            SELECT 1
            FROM UNNEST(IFNULL(p.claims_localized, [])) c
            WHERE LOWER(COALESCE(c.language, '')) = {pl!r}
              AND c.text IS NOT NULL
              AND LENGTH(TRIM(c.text)) > 0
          )
          OR EXISTS (
            SELECT 1
            FROM UNNEST(IFNULL(p.claims_localized_html, [])) c
            WHERE LOWER(COALESCE(c.language, '')) = {pl!r}
              AND c.text IS NOT NULL
              AND LENGTH(TRIM(c.text)) > 0
          )
        )
            """
        primary_lang_filter = f"""
        AND EXISTS (
          SELECT 1
          FROM UNNEST(IFNULL(p.abstract_localized, [])) a
          WHERE LOWER(COALESCE(a.language, '')) = {pl!r}
            AND a.text IS NOT NULL
            AND LENGTH(TRIM(a.text)) > 0
            {min_chars_filter}
        )
        {description_filter}
        {claim_filter}
        """

    any_claim_filter = ""
    if require_any_claim and not primary_lang:
        any_claim_filter = """
        AND (
          EXISTS (
            SELECT 1
            FROM UNNEST(IFNULL(p.claims_localized, [])) c
            WHERE c.text IS NOT NULL
              AND LENGTH(TRIM(c.text)) > 0
          )
          OR EXISTS (
            SELECT 1
            FROM UNNEST(IFNULL(p.claims_localized_html, [])) c
            WHERE c.text IS NOT NULL
              AND LENGTH(TRIM(c.text)) > 0
          )
        )
        """

    multilingual_having = ""
    if require_multilingual:
        multilingual_having = f"HAVING ARRAY_LENGTH(languages_present) >= {min_language_count}"

    limit_clause = f"\nLIMIT {limit}" if limit else ""

    query = f"""
    WITH base AS (
      SELECT
        p.publication_number,
        p.family_id,
        p.country_code,
        p.publication_date,
        p.title_localized,
        p.abstract_localized,
        p.description_localized,
        p.description_localized_html,
        p.claims_localized,
        p.claims_localized_html,
        p.cpc,
        p.ipc,

        ARRAY(
          SELECT DISTINCT t.language
          FROM UNNEST(IFNULL(p.title_localized, [])) t
          WHERE t.language IN ({lang_sql}) AND t.text IS NOT NULL
          UNION DISTINCT
          SELECT DISTINCT a.language
          FROM UNNEST(IFNULL(p.abstract_localized, [])) a
          WHERE a.language IN ({lang_sql}) AND a.text IS NOT NULL
        ) AS languages_present
      FROM `patents-public-data.patents.publications` p
      {surechembl_join}
      WHERE
        ({chemistry_where})
        {primary_lang_filter}
        {any_claim_filter}
        {date_filter}
        {country_filter}
    )
    SELECT *
    FROM base
    {multilingual_having}
    ORDER BY publication_date DESC
    {limit_clause}
    """
    return query


def build_query_per_language_top_n(
    *,
    languages: Optional[List[str]] = None,
    limit_per_lang: int,
    cpc_prefixes: Optional[List[str]] = None,
    ipc_prefixes: Optional[List[str]] = None,
    use_surechembl: bool = True,
    use_classification: bool = True,
    start_date: Optional[int] = None,
    end_date: Optional[int] = None,
    country_codes: Optional[List[str]] = None,
    min_abstract_words: int = MIN_ABSTRACT_WORDS,
    require_description: bool = False,
    require_claim: bool = False,
) -> str:
    """Build one top-N-per-language query for cheaper extraction."""
    languages = languages or DEFAULT_LANGS
    cpc_prefixes = cpc_prefixes or DEFAULT_CPC_PREFIXES
    ipc_prefixes = ipc_prefixes or DEFAULT_IPC_PREFIXES

    if not use_surechembl and not use_classification:
        raise ValueError("At least one of use_surechembl or use_classification must be True.")

    lang_array_sql = ", ".join(f"'{lang}'" for lang in languages)

    date_filter = ""
    if start_date is not None:
        date_filter += f"\n      AND p.publication_date >= {start_date}"
    if end_date is not None:
        date_filter += f"\n      AND p.publication_date <= {end_date}"

    country_filter = ""
    if country_codes:
        cc_sql = sql_list(country_codes)
        country_filter = f"\n      AND p.country_code IN ({cc_sql})"

    chemistry_predicates: List[str] = []
    if use_classification:
        cpc_preds = " OR ".join(f"STARTS_WITH(c.code, {p!r})" for p in cpc_prefixes)
        ipc_preds = " OR ".join(f"STARTS_WITH(i.code, {p!r})" for p in ipc_prefixes)
        chemistry_predicates.append(
            f"""
            EXISTS (
              SELECT 1
              FROM UNNEST(IFNULL(p.cpc, [])) AS c
              WHERE {cpc_preds}
            )
            """
        )
        chemistry_predicates.append(
            f"""
            EXISTS (
              SELECT 1
              FROM UNNEST(IFNULL(p.ipc, [])) AS i
              WHERE {ipc_preds}
            )
            """
        )

    surechembl_join = ""
    if use_surechembl:
        surechembl_join = """
        LEFT JOIN (
          SELECT DISTINCT publication_number
          FROM `patents-public-data.ebi_surechembl.match`
        ) sc
        ON p.publication_number = sc.publication_number
        """
        chemistry_predicates.append("sc.publication_number IS NOT NULL")

    chemistry_where = " OR ".join(f"({p.strip()})" for p in chemistry_predicates)
    min_abstract_chars = min_abstract_chars_for_sql(min_abstract_words)
    description_clause = ""
    if require_description:
        # Disabled for now: requiring localized descriptions collapses
        # multilingual coverage in the public BigQuery table.
        # description_clause = f"""
        # AND EXISTS (
        #   SELECT 1
        #   FROM UNNEST(IFNULL(b.description_localized, [])) d
        #   WHERE LOWER(COALESCE(d.language, '')) = lang
        #     AND d.text IS NOT NULL
        #     AND LENGTH(TRIM(d.text)) >= {MIN_DESCRIPTION_CHARS}
        # )
        # """
        pass

    claim_clause = ""
    if require_claim:
        claim_clause = """
      AND (
        EXISTS (
          SELECT 1
          FROM UNNEST(IFNULL(b.claims_localized, [])) c
          WHERE LOWER(COALESCE(c.language, '')) = lang
            AND c.text IS NOT NULL
            AND LENGTH(TRIM(c.text)) > 0
        )
        OR EXISTS (
          SELECT 1
          FROM UNNEST(IFNULL(b.claims_localized_html, [])) c
          WHERE LOWER(COALESCE(c.language, '')) = lang
            AND c.text IS NOT NULL
            AND LENGTH(TRIM(c.text)) > 0
        )
      )
        """

    return f"""
    WITH base AS (
      SELECT
        p.publication_number,
        p.family_id,
        p.country_code,
        p.publication_date,
        p.title_localized,
        p.abstract_localized,
        p.description_localized,
        p.description_localized_html,
        p.claims_localized,
        p.claims_localized_html,
        p.cpc,
        p.ipc
      FROM `patents-public-data.patents.publications` p
      {surechembl_join}
      WHERE
        ({chemistry_where})
        {date_filter}
        {country_filter}
    ),
    ranked AS (
      SELECT
        b.publication_number,
        lang,
        ROW_NUMBER() OVER (
          PARTITION BY lang
          ORDER BY b.publication_date DESC, b.publication_number DESC
        ) AS rn
      FROM base b
      CROSS JOIN UNNEST([{lang_array_sql}]) AS lang
      WHERE EXISTS (
        SELECT 1
        FROM UNNEST(IFNULL(b.abstract_localized, [])) a
        WHERE LOWER(COALESCE(a.language, '')) = lang
          AND a.text IS NOT NULL
          AND LENGTH(TRIM(a.text)) >= {min_abstract_chars}
      )
      {description_clause}
      {claim_clause}
    ),
    selected AS (
      SELECT DISTINCT publication_number
      FROM ranked
      WHERE rn <= {limit_per_lang}
    )
    SELECT
      b.publication_number,
      b.family_id,
      b.country_code,
      b.publication_date,
      b.title_localized,
      b.abstract_localized,
      b.description_localized,
      b.description_localized_html,
      b.claims_localized,
      b.claims_localized_html,
      b.cpc,
      b.ipc
    FROM base b
    INNER JOIN selected s
      USING (publication_number)
    ORDER BY b.publication_date DESC, b.publication_number DESC
    """


def _serialize_record(obj: Any) -> Any:
    """Convert BigQuery row values to JSON-serializable Python types."""
    return json.loads(json.dumps(obj, default=str))


def _run_query_iter(
    project_id: str,
    query: str,
    *,
    page_size: int = 1000,
):
    """Run BigQuery query and yield serialized record dicts."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    job_config = bigquery.QueryJobConfig(use_legacy_sql=False)
    query_job = client.query(query, job_config=job_config)
    result = query_job.result(page_size=page_size)

    for row in result:
        record: Dict[str, Any] = dict(row.items())
        for key in _RECORD_KEYS:
            if key in record and record[key] is not None:
                record[key] = _serialize_record(record[key])
        yield record


def run_query(
    project_id: str,
    query: str,
    output_path: Path,
    *,
    page_size: int = 1000,
) -> int:
    """
    Run BigQuery query and write results as NDJSON.
    Returns the number of rows written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in _run_query_iter(project_id, query, page_size=page_size):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count % 1000 == 0:
                print(f"Wrote {count:,} rows...")
    print(f"Done. Wrote {count:,} rows to: {output_path}")
    return count


def extract_chemistry_patents(
    project_id: str,
    output_path: Path,
    *,
    languages: Optional[List[str]] = None,
    cpc_prefixes: Optional[List[str]] = None,
    ipc_prefixes: Optional[List[str]] = None,
    use_surechembl: bool = True,
    use_classification: bool = True,
    require_multilingual: bool = False,
    min_language_count: int = 2,
    limit: Optional[int] = None,
    primary_lang: Optional[str] = None,
    start_date: Optional[int] = None,
    end_date: Optional[int] = None,
    country_codes: Optional[List[str]] = None,
) -> int:
    """
    Extract chemistry-related multilingual patents from Google Patents BigQuery.
    Writes NDJSON to output_path.
    Returns number of rows written.
    """
    query = build_query(
        languages=languages,
        cpc_prefixes=cpc_prefixes,
        ipc_prefixes=ipc_prefixes,
        use_surechembl=use_surechembl,
        use_classification=use_classification,
        require_multilingual=require_multilingual,
        min_language_count=min_language_count,
        limit=limit,
        primary_lang=primary_lang,
        min_primary_abstract_words=MIN_ABSTRACT_WORDS if primary_lang else None,
        require_primary_description=False,
        require_primary_claim=False,
        require_any_claim=False,
        start_date=start_date,
        end_date=end_date,
        country_codes=country_codes,
    )
    return run_query(
        project_id=project_id,
        query=query,
        output_path=Path(output_path),
    )


def extract_chemistry_patents_per_language(
    project_id: str,
    output_path: Path,
    *,
    languages: Optional[List[str]] = None,
    limit_per_lang: int = 100,
    cpc_prefixes: Optional[List[str]] = None,
    ipc_prefixes: Optional[List[str]] = None,
    use_surechembl: bool = True,
    use_classification: bool = True,
) -> int:
    """
    Pull one shared set of patents that covers up to limit_per_lang per language.

    This avoids scanning the same large public tables once per language, which
    is the main BigQuery cost driver in the naive implementation.
    """
    languages = languages or DEFAULT_LANGS
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fetch_limit = max(
        limit_per_lang + PER_LANGUAGE_OVERFETCH_MIN,
        int(limit_per_lang * PER_LANGUAGE_OVERFETCH_FACTOR),
    )
    query = build_query_per_language_top_n(
        languages=languages,
        limit_per_lang=fetch_limit,
        cpc_prefixes=cpc_prefixes,
        ipc_prefixes=ipc_prefixes,
        use_surechembl=use_surechembl,
        use_classification=use_classification,
        min_abstract_words=MIN_ABSTRACT_WORDS,
        require_description=False,
        require_claim=False,
    )
    tqdm.write(
        f"Running one top-N-per-language query for {len(languages)} languages "
        f"(target {limit_per_lang}, fetch {fetch_limit} per language)."
    )
    return run_query(project_id=project_id, query=query, output_path=output_path)


def _get_localized_text(
    items: Optional[List[Dict[str, Any]]],
    lang: str,
) -> Optional[str]:
    """Extract text for a given language from title_localized or abstract_localized."""
    if not items:
        return None
    for item in items:
        if isinstance(item, dict):
            lang_val = (item.get("language") or "").lower()
            if lang_val == lang.lower():
                text = item.get("text")
                if text and text.strip():
                    return text.strip()
    return None


def _build_first_claim(
    claim_items: Optional[List[Dict[str, Any]]],
    claim_html_items: Optional[List[Dict[str, Any]]],
    lang: str,
    *,
    max_chars: int = FIRST_CLAIM_MAX_CHARS,
) -> str:
    """Extract and lightly clean the first localized claim."""
    claim_html = _get_localized_text(claim_html_items, lang)
    if claim_html:
        first_claim = _extract_first_claim_from_html(claim_html)
        if first_claim:
            return _truncate_text(first_claim, max_chars=max_chars)

    claim_text = _get_localized_text(claim_items, lang)
    if claim_text:
        first_claim = _extract_first_claim_from_text(claim_text)
        if first_claim:
            return _truncate_text(first_claim, max_chars=max_chars)

    return ""


def _extract_first_claim_from_html(claim_html: str) -> str:
    """Parse the first claim from claims_localized_html."""
    match = re.search(r"<claim\b[^>]*>(.*?)</claim>", claim_html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""

    claim_block = re.sub(r"<chemistry\b.*?</chemistry>", " ", match.group(1), flags=re.IGNORECASE | re.DOTALL)
    claim_block = re.sub(r"</?claim-text\b[^>]*>", " ", claim_block, flags=re.IGNORECASE)
    claim_block = re.sub(r"<[^>]+>", " ", claim_block)
    return _normalize_claim_text(claim_block)


def _extract_first_claim_from_text(claim_text: str) -> str:
    """Parse the first claim from claims_localized plain text."""
    cleaned = clean_text(claim_text)
    if not cleaned:
        return ""

    cleaned = re.sub(r"^\s*1\s*[\.\):\-]*\s*", "", cleaned)
    next_claim = re.search(r"\s2\s*[\.\):\-]\s+", cleaned)
    if next_claim:
        cleaned = cleaned[:next_claim.start()]
    return _normalize_claim_text(cleaned)


def _normalize_claim_text(text: str) -> str:
    """Normalize parsed claim text and drop the leading claim number."""
    text = clean_text(text)
    text = re.sub(r"^\s*1\s*[\.\):\-]*\s*", "", text)
    return text.strip()


def _truncate_text(text: str, *, max_chars: int) -> str:
    """Trim long text without splitting the last word when possible."""
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    snippet = text[:max_chars].rsplit(" ", 1)[0].strip()
    return snippet or text[:max_chars].strip()


def _build_description_snippet(
    items: Optional[List[Dict[str, Any]]],
    lang: str,
    *,
    max_chars: int = DESCRIPTION_SNIPPET_MAX_CHARS,
) -> str:
    """Extract and lightly clean a localized description snippet."""
    description = _get_localized_text(items, lang)
    if not description:
        return ""

    description = re.sub(r"\[\d{3,5}\]", " ", description)
    description = clean_text(description)
    return _truncate_text(description, max_chars=max_chars)


def preprocess_ndjson_to_csv(
    ndjson_path: Path,
    output_dir: Path,
    *,
    languages: Optional[List[str]] = None,
    per_lang_limit: Optional[int] = None,
    min_abstract_words: int = MIN_ABSTRACT_WORDS,
) -> Dict[str, int]:
    """
    Preprocess NDJSON patent data into per-language CSVs for QAC generation.

    For each language, extracts records that have title/abstract in that
    language, dedupes by publication_number, optionally caps at per_lang_limit
    rows, writes CSV. When available, the first claim is added as an optional
    context enrichment field.

    CSV columns: id, language, title, abstract, description, first_claim, context,
    publication_number, country_code, publication_date, source

    Returns dict mapping language -> row count.
    """
    languages = languages or DEFAULT_LANGS
    ndjson_path = Path(ndjson_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all records once
    records: List[Dict[str, Any]] = []
    with ndjson_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    counts: Dict[str, int] = {}
    fieldnames = [
        "id",
        "language",
        "title",
        "abstract",
        "description",
        "first_claim",
        "context",
        "publication_number",
        "country_code",
        "publication_date",
        "source",
        "ipc_codes",
    ]

    for lang in tqdm(languages, desc="Preprocess languages", unit="lang"):
        rows: List[Dict[str, Any]] = []
        seen_pub: set = set()
        skipped_short = 0
        missing_claim = 0
        for rec in records:
            title = _get_localized_text(rec.get("title_localized"), lang)
            abstract = _get_localized_text(rec.get("abstract_localized"), lang)
            # Disabled for now: keep the column for schema stability, but build
            # contexts from title + abstract, with first claim when available.
            description = ""
            first_claim = _build_first_claim(
                rec.get("claims_localized"),
                rec.get("claims_localized_html"),
                lang,
            )

            if not abstract and not title:
                continue

            pub_num = rec.get("publication_number") or ""
            if pub_num in seen_pub:
                continue
            seen_pub.add(pub_num)

            title = clean_text(title or "")
            abstract = clean_text(abstract or "")
            if word_count(abstract) < min_abstract_words:
                skipped_short += 1
                continue
            if not first_claim:
                missing_claim += 1
            context_parts = []
            if title:
                context_parts.append(f"Title: {title}")
            if abstract:
                context_parts.append(f"Abstract: {abstract}")
            if first_claim:
                context_parts.append(f"First claim: {first_claim}")
            context = "\n\n".join(context_parts).strip()

            ipc_codes = "|".join(
                code
                for entry in (rec.get("ipc") or [])
                if (code := (entry.get("code") or "").strip())
            )

            rows.append({
                "id": f"{pub_num}_{lang}",
                "language": lang,
                "title": title,
                "abstract": abstract,
                "description": description,
                "first_claim": first_claim,
                "context": context,
                "publication_number": pub_num,
                "country_code": rec.get("country_code") or "",
                "publication_date": rec.get("publication_date") or "",
                "source": "google_patents",
                "ipc_codes": ipc_codes,
            })

            if per_lang_limit and len(rows) >= per_lang_limit:
                break

        out_path = output_dir / f"{lang}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        counts[lang] = len(rows)
        tqdm.write(
            f"  {lang}: {len(rows):,} rows -> {out_path}"
            f" (skipped {skipped_short:,} short/title-only records,"
            f" {missing_claim:,} without claims)"
        )

    return counts


def merge_corpus_csv(
    preprocessed_dir: Path,
    output_path: Path,
    *,
    languages: Optional[List[str]] = None,
    min_abstract_words: int = MIN_ABSTRACT_WORDS,
) -> int:
    """
    Merge all per-language CSVs into one corpus CSV. Applies clean_text.
    Corpus = documents for retrieval; queries/answers come from QAC generation.
    """
    preprocessed_dir = Path(preprocessed_dir)
    output_path = Path(output_path)
    languages = languages or DEFAULT_LANGS

    rows: List[Dict[str, Any]] = []
    for lang in languages:
        p = preprocessed_dir / f"{lang}.csv"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                row["title"] = clean_text(row.get("title", ""))
                row["abstract"] = clean_text(row.get("abstract", ""))
                row["description"] = clean_text(row.get("description", ""))
                row["first_claim"] = clean_text(row.get("first_claim", ""))
                row["context"] = clean_text(row.get("context", ""))
                row["ipc_codes"] = row.get("ipc_codes", "")
                if word_count(row["abstract"]) < min_abstract_words:
                    continue
                if not row["context"]:
                    context_parts = []
                    if row["title"]:
                        context_parts.append(f"Title: {row['title']}")
                    if row["abstract"]:
                        context_parts.append(f"Abstract: {row['abstract']}")
                    if row["first_claim"]:
                        context_parts.append(f"First claim: {row['first_claim']}")
                    row["context"] = "\n\n".join(context_parts).strip()
                rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "language", "title", "abstract", "description", "first_claim", "context",
        "publication_number", "country_code", "publication_date", "source", "ipc_codes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Merged {len(rows):,} rows -> {output_path}")
    return len(rows)
