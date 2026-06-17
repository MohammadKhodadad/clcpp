"""
Claude (via OpenRouter) feedback graders for the progressive pipeline.

Query *generation* stays on gpt-5-mini (OpenAI); the two *feedback* verifiers —
faithfulness and technical quality — run on Claude Sonnet 4.5. The grading prompts
and the score computation are byte-identical to the alias-graph single-query
graders; the only difference is the transport: Claude via OpenRouter does not
accept ``reasoning_effort``, so we use Claude's ``thinking`` budget instead (mirrors
``scripts/regrade_with_openrouter.py``).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

from openai import OpenAI

from src.alias_graph.qac_generation.concept_qa import (
    _FAITHFULNESS_PROMPT,
    _QUALITY_PROMPT,
    _load_text,
)
from src.multi_lingual_qac.qac_generation.multilingual_qa import (
    MODE_TECHNICAL,
    _compute_faith_overall,
    _compute_quality_overall,
    _parse_json_response,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_GRADER_MODEL = "anthropic/claude-sonnet-4.5"
_THINKING_BUDGET_TOKENS = 8000
_MAX_TOKENS = 12000


@lru_cache(maxsize=1)
def get_grader_client() -> OpenAI:
    """OpenRouter client (OpenAI-compatible) for Claude grading. Cached singleton."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY in .env for Claude (OpenRouter) grading.")
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def _grade(client: OpenAI, model: str, system_prompt: str, all_passages: str, qa: Dict[str, str]) -> Dict[str, Any]:
    user = f"{all_passages}\n\nQuestion: {qa['question']}\nAnswer: {qa['answer']}"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
        max_tokens=_MAX_TOKENS,
        extra_body={"thinking": {"type": "enabled", "budget_tokens": _THINKING_BUDGET_TOKENS}},
    )
    data = _parse_json_response(resp.choices[0].message.content or "")
    if isinstance(data, list):
        data = data[0] if data else {}
    return data if isinstance(data, dict) else {}


def grade_faithfulness_claude(
    client: OpenAI, all_passages: str, qa: Dict[str, str], *, model: str = DEFAULT_GRADER_MODEL
) -> Dict[str, Any]:
    """Faithfulness grade for ONE (question, answer) pair — same rubric as the
    alias-graph grader, run on Claude via OpenRouter."""
    data = _grade(client, model, _load_text(_FAITHFULNESS_PROMPT), all_passages, qa)
    row = {
        "grounding": int(data.get("grounding", 1)),
        "precision": int(data.get("precision", 1)),
        "numerical_fidelity": int(data.get("numerical_fidelity", 1)),
        "reason": str(data.get("reason", "")).strip(),
    }
    row["overall"] = _compute_faith_overall(row)
    return row


def grade_quality_claude(
    client: OpenAI, all_passages: str, qa: Dict[str, str], *, model: str = DEFAULT_GRADER_MODEL
) -> Dict[str, Any]:
    """Technical-quality grade for ONE question — same rubric as the alias-graph
    grader, run on Claude via OpenRouter."""
    data = _grade(client, model, _load_text(_QUALITY_PROMPT), all_passages, qa)
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
