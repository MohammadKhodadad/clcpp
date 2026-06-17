"""
Variant E (non-chemistry control) for code-switching — the only variant that
needs an LLM. The model picks an ordinary, non-chemistry noun that appears in the
passage (and does not overlap any chemistry term) and gives its translation in a
target language; the caller then swaps all occurrences.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

_PROMPT = Path(__file__).resolve().parent / "code_switch_prompts" / "nonchem_swap.txt"


def get_nonchem_swapper(
    model: str = "gpt-5-mini",
) -> Callable[[str, List[str], str], Optional[Tuple[str, str]]]:
    """Return ``swap(doc_text, avoid_terms, target_lang) -> (original, replacement) | None``."""
    from src.multi_lingual_qac.qac_generation.multilingual_qa import (
        DEFAULT_REASONING_EFFORT,
        LANG_NAMES,
        _get_client,
        _parse_json_response,
    )

    client = _get_client()
    prompt = _PROMPT.read_text(encoding="utf-8").strip()

    def swap(doc_text: str, avoid_terms: List[str], target_lang: str) -> Optional[Tuple[str, str]]:
        lang_name = LANG_NAMES.get(target_lang, target_lang)
        user = (
            f"TARGET LANGUAGE: {lang_name}\n"
            f"CHEMISTRY TERMS TO AVOID (never pick any of these): {', '.join(avoid_terms)}\n\n"
            f"PASSAGE:\n{doc_text}"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user},
            ],
            reasoning_effort=DEFAULT_REASONING_EFFORT,
        )
        data = _parse_json_response(resp.choices[0].message.content or "")
        if isinstance(data, list):
            data = data[0] if data else {}
        original = str(data.get("original_term", "")).strip()
        replacement = str(data.get("replacement_term", "")).strip()
        if not original or not replacement:
            return None
        if original.casefold() in {a.casefold() for a in avoid_terms}:
            return None  # model picked a chemistry term despite instructions
        return original, replacement

    return swap
