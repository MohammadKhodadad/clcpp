"""
QA generation for the code-switched document variants (idea 2).

For variants B/C/D/F the question is generated "like alias graph" but must use the
**original** term verbatim (the term that was swapped out, e.g. "[CO2]"); the
retrieval gold is the variant document (which now contains the *replacement*). One
query is generated per (source doc, concept, original term) and reused for each of
that group's variant docs. For variant E it is a normal document QA, exactly like
``multi_lingual_qac.qac_generation.multilingual_qa``, with the E doc as gold.

Reuses the two single-query verifiers from ``concept_qa`` (B/C/D/F) and the
3-candidate generator + graders from ``multilingual_qa`` (E).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from src.alias_graph.builder import _read_corpus
from src.alias_graph.qac_generation.concept_qa import (
    _pick_answer,
    grade_faithfulness_single,
    grade_quality_single,
)
from src.multi_lingual_qac.qac_generation.multilingual_qa import (
    DEFAULT_GENERATION_REASONING_EFFORT,
    DEFAULT_MODEL,
    FAITHFULNESS_FIELDS,
    MODE_TECHNICAL,
    TECHNICAL_QUALITY_FIELDS,
    _build_all_passages_text,
    _compute_total_score,
    _get_client,
    _parse_json_response,
    generate_qa_batch,
    grade_faithfulness,
    grade_quality,
)

_PROMPT_DIR = Path(__file__).resolve().parent / "concept_query_with_term_prompts"
_prompt_cache: Dict[str, str] = {}
_MAX_GEN_ATTEMPTS = 2

OUTPUT_FIELDS: List[str] = [
    "variant", "concept_chebi_id", "concept_name", "query_language", "term_used",
    "question", "answer", "question_type",
    *FAITHFULNESS_FIELDS, *TECHNICAL_QUALITY_FIELDS, "qual_failure_type", "total_score",
    "gold_id", "source_id",
]


def _load_prompt(lang: str) -> str:
    path = _PROMPT_DIR / f"{lang}.txt"
    key = str(path)
    if key not in _prompt_cache:
        if not path.exists():
            raise FileNotFoundError(f"Term-query prompt not found: {path}")
        _prompt_cache[key] = path.read_text(encoding="utf-8").strip()
    return _prompt_cache[key]


def generate_term_query(
    client, all_passages: str, concept_name: str, term: str, lang: str, *, model: str
) -> Optional[Dict[str, str]]:
    """Generate ONE query in ``lang`` that uses ``term`` verbatim. Returns None if
    the model never includes the term exactly (after a retry)."""
    prompt = _load_prompt(lang)
    base_user = f"CONCEPT: {concept_name}\nTERM (use exactly, verbatim): {term}\n\nPASSAGES:\n{all_passages}"
    for attempt in range(_MAX_GEN_ATTEMPTS):
        user = base_user
        if attempt:
            user += (
                f"\n\nThe previous attempt did not contain the TERM exactly. "
                f"You MUST include this exact string in the question, unchanged: {term}"
            )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user}],
            reasoning_effort=DEFAULT_GENERATION_REASONING_EFFORT,
        )
        data = _parse_json_response(resp.choices[0].message.content or "")
        if isinstance(data, list):
            data = data[0] if data else {}
        question = str(data.get("question", "")).strip()
        if question and term in question:  # verbatim, exact substring
            return {"question": question, "question_type": str(data.get("question_type", "other")).strip()}
    return None


def _grade_fields(faith: Dict[str, Any], qual: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "faith_grounding": faith["grounding"], "faith_precision": faith["precision"],
        "faith_numerical_fidelity": faith["numerical_fidelity"], "faith_overall": faith["overall"],
        "qual_search_bar_realism": qual["search_bar_realism"], "qual_specificity": qual["specificity"],
        "qual_phrasing_economy": qual["phrasing_economy"], "qual_focus": qual["focus"],
        "qual_linguistic_quality": qual["linguistic_quality"], "qual_overall": qual["overall"],
        "qual_failure_type": qual.get("failure_type", "none"),
        "total_score": _compute_total_score(faith, qual, MODE_TECHNICAL),
    }


def _process_group(group: Dict[str, Any], source_by_id: Dict[str, dict],
                   name_set_by_cid: Dict[str, dict], model: str) -> List[Dict[str, Any]]:
    """B/C/D/F: one term-query, reused across the group's variant docs."""
    src = source_by_id.get(group["source_id"])
    if src is None:
        return []
    all_passages = _build_all_passages_text([src])
    if not all_passages.strip():
        return []
    client = _get_client()
    lang = group["anchor_language"]
    term = group["original_term"]
    cname = group["concept_name"]
    gen = generate_term_query(client, all_passages, cname, term, lang, model=model)
    if gen is None:
        return []
    name_set = name_set_by_cid.get(group["concept_chebi_id"], {})
    answer, _alang, _ground = _pick_answer(name_set, lang, all_passages)
    qa = {"question": gen["question"], "answer": answer}
    faith = grade_faithfulness_single(client, all_passages, qa, model=model)
    qual = grade_quality_single(client, all_passages, qa, model=model)
    fields = _grade_fields(faith, qual)
    rows: List[Dict[str, Any]] = []
    for variant, gold_id in group["variants"]:
        rows.append({
            "variant": variant, "concept_chebi_id": group["concept_chebi_id"],
            "concept_name": cname, "query_language": lang, "term_used": term,
            "question": gen["question"], "answer": answer,
            "question_type": gen["question_type"], **fields,
            "gold_id": gold_id, "source_id": group["source_id"],
        })
    return rows


def _process_e_row(row: Dict[str, str], model: str) -> List[Dict[str, Any]]:
    """E: a normal document QA (like multilingual_qa) on the E variant doc."""
    all_passages = _build_all_passages_text([row])
    if not all_passages.strip():
        return []
    client = _get_client()
    lang = row.get("anchor_language") or row.get("language", "en")
    try:
        qa_pairs = generate_qa_batch(client, all_passages, lang, MODE_TECHNICAL, model=model)
        if not qa_pairs:
            return []
        faith = grade_faithfulness(client, all_passages, qa_pairs, model=model)
        qual = grade_quality(client, all_passages, qa_pairs, MODE_TECHNICAL, model=model)
    except Exception as exc:
        tqdm.write(f"  E {row['id']}: error {exc}")
        return []
    best_i = max(range(len(qa_pairs)), key=lambda i: faith[i]["overall"] + qual[i]["overall"])
    fields = _grade_fields(faith[best_i], qual[best_i])
    return [{
        "variant": "E", "concept_chebi_id": row.get("concept_chebi_id", ""),
        "concept_name": row.get("concept_name", ""), "query_language": lang, "term_used": "",
        "question": qa_pairs[best_i]["question"], "answer": qa_pairs[best_i]["answer"],
        "question_type": qa_pairs[best_i].get("question_type", ""), **fields,
        "gold_id": row["id"], "source_id": row.get("source_id", ""),
    }]


def run_variant_qa(
    corpus_csv: Path,
    source_corpus: Path,
    alias_json: Path,
    output_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    seed: int = 42,
    limit: Optional[int] = None,
    workers: int = 1,
) -> int:
    """Generate questions for the code-switched variants; returns rows written."""
    rows = _read_corpus(corpus_csv)
    source_by_id = {r["id"]: r for r in _read_corpus(source_corpus)}
    with Path(alias_json).open(encoding="utf-8") as fh:
        name_set_by_cid = {c["chebi_id"]: c.get("name_set", {}) for c in json.load(fh)["concepts"]}

    # Group B/C/D/F rows by (source_id, concept, original_term); collect E rows.
    groups: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    e_rows: List[Dict[str, str]] = []
    for r in rows:
        if r["variant"] == "E":
            e_rows.append(r)
            continue
        if r["variant"] == "A":
            continue
        key = (r["source_id"], r["concept_chebi_id"], r["original_term"])
        g = groups.setdefault(key, {
            "source_id": r["source_id"], "concept_chebi_id": r["concept_chebi_id"],
            "concept_name": r["concept_name"], "anchor_language": r["anchor_language"],
            "original_term": r["original_term"], "variants": [],
        })
        g["variants"].append((r["variant"], r["id"]))

    group_list = list(groups.values())
    if limit is not None:
        group_list = group_list[:limit]
        e_rows = e_rows[:limit]
    print(
        f"Variant QA: {len(group_list)} B/C/D/F groups + {len(e_rows)} E docs, "
        f"model={model}, workers={workers}"
    )

    jobs = [("g", g) for g in group_list] + [("e", r) for r in e_rows]
    out_rows: List[Dict[str, Any]] = []

    def run_job(job):
        kind, payload = job
        if kind == "g":
            return _process_group(payload, source_by_id, name_set_by_cid, model)
        return _process_e_row(payload, model)

    if workers > 1 and jobs:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run_job, j) for j in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Variant QA", unit="job"):
                out_rows.extend(fut.result())
    else:
        for job in tqdm(jobs, desc="Variant QA", unit="job"):
            out_rows.extend(run_job(job))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    from collections import Counter
    print(f"\nWrote {len(out_rows)} variant-QA rows -> {output_path}")
    print(f"  per variant: {dict(sorted(Counter(r['variant'] for r in out_rows).items()))}")
    return len(out_rows)
