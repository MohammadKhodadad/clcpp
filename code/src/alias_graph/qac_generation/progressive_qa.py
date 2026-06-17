"""
Query generation for the progressive (cumulative-ladder) code-switched variants.

Each base document produces ONE fixed query, about the concept whose term is
swapped at the **first** rung of the ladder (e.g. "carbon dioxide"). That query is
reused as the query for every depth 0..N of the ladder; the gold for the query is
the variant document at each depth, so the eval can measure how retrieval of the
(increasingly code-switched) document decays while the query stays constant.

Generation is **identical to the Alias-Graph concept-query pipeline**
(``concept_qa``): the exact same per-language prompt
(``concept_query_generation_prompts/<lang>.txt``), the same "describe the concept,
never name it or its aliases" contract, the same answer selection (``_pick_answer``)
and the same faithfulness + technical-quality verifiers. As in the alias graph, the
document is passed in **all of its available languages** at once
(``_build_all_passages_text`` over the full multilingual group for the
publication); only the query language differs — here it is the ladder's anchor
language (the language the variant documents are written in).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.alias_graph.qac_generation.claude_grading import (
    DEFAULT_GRADER_MODEL,
    get_grader_client,
    grade_faithfulness_claude,
    grade_quality_claude,
)
from src.alias_graph.qac_generation.concept_qa import (
    _all_aliases,
    _pick_answer,
    generate_concept_query,
)
from src.alias_graph.qac_generation.variant_qa import _grade_fields
from src.multi_lingual_qac.qac_generation.multilingual_qa import (
    DEFAULT_MODEL,
    FAITHFULNESS_FIELDS,
    STRATEGY_ALL,
    STRATEGY_NAMES,
    TECHNICAL_QUALITY_FIELDS,
    _build_all_passages_text,
    _get_client,
    load_multilingual_corpus,
    pick_target_languages,
)

_MAX_GEN_RETRIES = 3  # same retry budget as the alias-graph concept-query pipeline

OUTPUT_FIELDS: List[str] = [
    "base_id", "query_id", "n_replacements", "concept_chebi_id", "concept_name",
    "query_language", "strategy", "term_used", "question", "answer", "question_type",
    *FAITHFULNESS_FIELDS, *TECHNICAL_QUALITY_FIELDS, "qual_failure_type", "total_score",
    "gold_id", "source_id",
]


def _query_for_lang(
    gen_client, grader_client, base: Dict[str, Any], all_passages: str,
    name_set: dict, aliases: List[str], lang: str, model: str, grader_model: str,
) -> List[Dict[str, Any]]:
    """Generate + grade ONE concept-query in ``lang`` and emit its ladder rows.

    Generation is exactly like the alias graph (gpt-5-mini, all-language passages,
    describe-don't-name); the two feedback verifiers run on Claude Sonnet 4.5.
    """
    cname = base["concept_name"]
    gen = None
    for _ in range(_MAX_GEN_RETRIES):
        try:
            cand = generate_concept_query(gen_client, all_passages, cname, aliases, lang, model=model)
        except Exception as exc:
            tqdm.write(f"  {base['concept_chebi_id']} [{lang}]: generation error: {exc}")
            continue
        if cand["question"]:
            gen = cand
            break
    if gen is None:
        return []

    answer, _alang, _ground = _pick_answer(name_set, lang, all_passages)
    if not answer:
        return []
    qa = {"question": gen["question"], "answer": answer}
    try:
        faith = grade_faithfulness_claude(grader_client, all_passages, qa, model=grader_model)
        qual = grade_quality_claude(grader_client, all_passages, qa, model=grader_model)
    except Exception as exc:
        tqdm.write(f"  {base['concept_chebi_id']} [{lang}]: grading error ({grader_model}): {exc}")
        return []
    fields = _grade_fields(faith, qual)

    query_id = f"{base['source_id']}__q_{lang}"
    rows: List[Dict[str, Any]] = []
    for depth, gold_id in sorted(base["variants"]):
        rows.append({
            "base_id": base["source_id"], "query_id": query_id, "n_replacements": depth,
            "concept_chebi_id": base["concept_chebi_id"], "concept_name": cname,
            "query_language": lang, "strategy": base["strategy"], "term_used": base["original_term"],
            "question": gen["question"], "answer": answer,
            "question_type": gen["question_type"], **fields,
            "gold_id": gold_id, "source_id": base["source_id"],
        })
    return rows


def _process_base(base: Dict[str, Any], groups: Dict[str, List[dict]],
                  name_set_by_cid: Dict[str, dict], model: str, grader_model: str) -> List[Dict[str, Any]]:
    """Generate one concept-query PER target language for a base doc (the query
    language set is chosen by the strategy), each reused across the base's ladder.
    All language versions of the source publication are passed together as
    passages, exactly like the alias-graph pipeline."""
    doc_rows = groups.get(base["publication_number"])
    if not doc_rows:
        return []
    all_passages = _build_all_passages_text(doc_rows)
    if not all_passages.strip():
        return []
    gen_client = _get_client()              # gpt-5-mini (OpenAI) for generation
    grader_client = get_grader_client()     # Claude Sonnet 4.5 (OpenRouter) for feedback
    name_set = name_set_by_cid.get(base["concept_chebi_id"], {})
    aliases = _all_aliases(name_set)

    rows: List[Dict[str, Any]] = []
    for lang in base["target_langs"]:
        rows.extend(_query_for_lang(
            gen_client, grader_client, base, all_passages, name_set, aliases,
            lang, model, grader_model,
        ))
    return rows


def run_progressive_qa(
    corpus_csv: Path,
    source_corpus: Path,
    alias_json: Path,
    output_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    grader_model: str = DEFAULT_GRADER_MODEL,
    strategy: int = STRATEGY_ALL,
    seed: int = 42,
    limit: Optional[int] = None,
    workers: int = 1,
) -> int:
    """Generate the per-base concept-queries for the progressive variants.

    The query language(s) for each base are chosen by ``strategy`` (the same four
    alias-graph strategies); ``STRATEGY_ALL`` emits one query per language (5 per
    base). Each query is reused across its base's ladder depths. Returns rows
    written (one per (base, query language, depth))."""
    import random

    from src.alias_graph.builder import _read_corpus

    rows = _read_corpus(corpus_csv)
    groups = load_multilingual_corpus(source_corpus)
    with Path(alias_json).open(encoding="utf-8") as fh:
        name_set_by_cid = {c["chebi_id"]: c.get("name_set", {}) for c in json.load(fh)["concepts"]}

    # Group ladder rows by base document; collect (depth, gold variant id).
    bases: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        base_id = r["base_id"]
        b = bases.setdefault(base_id, {
            "source_id": base_id,
            "publication_number": r["source_publication_number"],
            "concept_chebi_id": r["question_concept_chebi_id"],
            "concept_name": r["question_concept_name"],
            "query_language": r["anchor_language"],
            "original_term": r["question_original_term"],
            "strategy": strategy,
            "variants": [],
        })
        b["variants"].append((int(r["n_replacements"]), r["id"]))

    base_list = list(bases.values())
    if limit is not None:
        base_list = base_list[:limit]

    # Pick each base's query languages up front (single-threaded) so the run is
    # reproducible; STRATEGY_ALL returns all five languages deterministically.
    random.seed(seed)
    for b in base_list:
        present = [r.get("language") for r in groups.get(b["publication_number"], [])]
        b["target_langs"] = pick_target_languages(strategy, present)
    n_queries = sum(len(b["target_langs"]) for b in base_list)
    print(f"Progressive QA: {len(base_list)} base docs x strategy="
          f"{STRATEGY_NAMES.get(strategy, strategy)} -> {n_queries} queries; "
          f"gen_model={model}, grader_model={grader_model}, workers={workers}")

    def run_job(b):
        return _process_base(b, groups, name_set_by_cid, model, grader_model)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write rows to the CSV as each base completes (flushed in place), so an
    # interrupted run keeps everything generated so far.
    written = 0
    query_ids: set = set()
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        fh.flush()

        def persist(rows: List[Dict[str, Any]]) -> None:
            nonlocal written
            for row in rows:
                writer.writerow(row)
                written += 1
                query_ids.add(row["query_id"])
            fh.flush()

        if workers > 1 and base_list:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(run_job, b) for b in base_list]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Progressive QA", unit="base"):
                    persist(fut.result())
        else:
            for b in tqdm(base_list, desc="Progressive QA", unit="base"):
                persist(run_job(b))

    print(f"\nWrote {written} progressive-QA rows ({len(query_ids)} queries) -> {output_path}")
    return written
