"""
Wikipedia multilingual-name bridge for ChEBI concepts.

A concept's stable identity is its ChEBI id. Wikidata records that id on the
matching item via property ``P683`` (ChEBI ID), and every Wikidata item carries
Wikipedia sitelinks per language. So ChEBI id -> Wikidata item (via P683) ->
Wikipedia article titles gives us, for the *same* concept, the name people
actually use in each target language -- without any string matching that could
break the shared-identity guarantee.

Names are fetched in batches from the Wikidata Query Service (SPARQL) and cached
to disk keyed by ChEBI id, so only never-seen ids ever hit the network. Ids that
resolve to no Wikipedia title are cached as ``{}`` so they are not re-queried.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import requests


SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
# WQS policy requires a descriptive User-Agent with contact info.
USER_AGENT = "multi-lingual-qac/0.1 (research; +https://github.com/) chebi-wikipedia-bridge"
DEFAULT_LANGS: Sequence[str] = ("zh", "en", "de", "fr", "es")

_BATCH_SIZE = 50
_SLEEP_BETWEEN_BATCHES = 1.0
_MAX_RETRIES = 4
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _numeric_id(chebi_id: str) -> str:
    """`CHEBI:15365` -> `15365` (P683 stores the bare number)."""
    return chebi_id.split(":", 1)[-1]


def _build_query(numeric_ids: Sequence[str], langs: Sequence[str]) -> str:
    values = " ".join(f'"{n}"' for n in numeric_ids)
    lang_filter = ", ".join(f'"{lang}"' for lang in langs)
    return f"""
SELECT ?chebi ?lang ?name WHERE {{
  VALUES ?chebi {{ {values} }}
  ?item wdt:P683 ?chebi .
  ?article schema:about ?item ;
           schema:inLanguage ?lang ;
           schema:isPartOf [ wikibase:wikiGroup "wikipedia" ] ;
           schema:name ?name .
  FILTER(?lang in ({lang_filter}))
}}
"""


def _load_cache(path: Path) -> Dict[str, Dict[str, str]]:
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_cache(path: Path, cache: Dict[str, Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=0)
    tmp.replace(path)


def _query_batch(
    numeric_ids: Sequence[str], langs: Sequence[str]
) -> Dict[str, Dict[str, str]]:
    # POST (not GET): the VALUES list makes the query long, and WQS 502s on
    # oversized URLs. Retry transient 5xx/429 with exponential backoff.
    query = _build_query(numeric_ids, langs)
    headers = {"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"}
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.post(
                SPARQL_ENDPOINT,
                data={"query": query, "format": "json"},
                headers=headers,
                timeout=120,
            )
            if resp.status_code in _RETRY_STATUS:
                raise requests.HTTPError(f"{resp.status_code} from WQS", response=resp)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
    bindings = resp.json()["results"]["bindings"]
    out: Dict[str, Dict[str, str]] = {}
    for row in bindings:
        chebi_id = f"CHEBI:{row['chebi']['value']}"
        lang = row["lang"]["value"]
        out.setdefault(chebi_id, {})[lang] = row["name"]["value"]
    return out


def fetch_wikipedia_names(
    chebi_ids: Iterable[str],
    langs: Sequence[str] = DEFAULT_LANGS,
    cache_path: Path | str = "data/chebi/wiki_names_cache.json",
) -> Dict[str, Dict[str, str]]:
    """
    Return ``{chebi_id: {lang: wikipedia_title}}`` for the requested ids.

    Only ids absent from the cache are queried (batched). A failed batch is
    skipped (not cached), so the run still produces KG-only names and the batch
    can be retried later. The cache is persisted incrementally after each batch.
    """
    cache_path = Path(cache_path)
    cache = _load_cache(cache_path)

    wanted = list(dict.fromkeys(chebi_ids))  # de-dup, keep order
    missing = [cid for cid in wanted if cid not in cache]
    if missing:
        print(
            f"Fetching Wikipedia names for {len(missing)} ChEBI ids "
            f"({len(wanted) - len(missing)} cached) in {langs} ..."
        )
    for start in range(0, len(missing), _BATCH_SIZE):
        batch = missing[start : start + _BATCH_SIZE]
        numeric = [_numeric_id(cid) for cid in batch]
        try:
            results = _query_batch(numeric, langs)
        except Exception as exc:  # network / SPARQL hiccup: skip, keep KG names
            print(f"  Wikidata batch {start // _BATCH_SIZE} failed ({exc}); skipping.")
            continue
        for cid in batch:
            cache[cid] = results.get(cid, {})  # {} = queried, no Wikipedia article
        _save_cache(cache_path, cache)
        time.sleep(_SLEEP_BETWEEN_BATCHES)

    return {cid: cache.get(cid, {}) for cid in wanted}
