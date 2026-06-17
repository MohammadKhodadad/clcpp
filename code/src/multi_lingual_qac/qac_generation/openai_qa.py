"""
OpenAI-based Q&A generation (Option A: English first, translate to all languages).

Samples corpus, generates (question, answer) in English per document,
then translates to all target languages.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

# Languages to translate into (exclude en; we generate in English)
DEFAULT_TARGET_LANGS = [
    "de", "fr", "es", "ja", "ko", "zh", "ru", "pt", "it", "nl", "ar", "fa", "tr", "pl", "hi",
]

LANG_NAMES = {
    "de": "German", "fr": "French", "es": "Spanish", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "ru": "Russian", "pt": "Portuguese", "it": "Italian", "nl": "Dutch",
    "ar": "Arabic", "fa": "Farsi", "tr": "Turkish", "pl": "Polish", "hi": "Hindi", "en": "English",
}

DEFAULT_GENERATION_MODEL = "gpt-5-mini"
DEFAULT_QUALITY_MODEL = "gpt-5-mini"
DEFAULT_SUPPORT_MODEL = "gpt-5-mini"
DEFAULT_TRANSLATION_MODEL = "gpt-5-mini"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_GENERATION_REASONING_EFFORT = "medium"
DEFAULT_TRANSLATION_REASONING_EFFORT = "medium"


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in .env for Q&A generation.")
    return OpenAI(api_key=api_key)


def _parse_json_response(text: str) -> Dict[str, Any]:
    """Parse a JSON object from a model response."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def load_corpus(corpus_path: Path) -> List[Dict[str, Any]]:
    """Load corpus CSV into list of dicts."""
    rows: List[Dict[str, Any]] = []
    with corpus_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def sample_corpus(
    rows: List[Dict[str, Any]],
    sample_size: int,
    *,
    stratify_by_language: bool = True,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Sample rows from corpus. If stratify_by_language, take proportionally from each language.
    """
    if seed is not None:
        random.seed(seed)
    if sample_size >= len(rows):
        return rows
    if not stratify_by_language:
        return random.sample(rows, sample_size)
    # Stratify: group by language
    by_lang: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        lang = row.get("language", "en")
        by_lang.setdefault(lang, []).append(row)
    # Sample proportionally
    total = len(rows)
    sampled: List[Dict[str, Any]] = []
    for lang, lang_rows in by_lang.items():
        n = max(1, round(sample_size * len(lang_rows) / total))
        n = min(n, len(lang_rows))
        sampled.extend(random.sample(lang_rows, n))
    # If we got more or fewer, trim or pad
    random.shuffle(sampled)
    return sampled[:sample_size]


def generate_qa_english(
    client: OpenAI,
    context: str,
    *,
    previous_question: Optional[str] = None,
    previous_answer: Optional[str] = None,
    previous_feedback: Optional[str] = None,
    model: str = DEFAULT_GENERATION_MODEL,
) -> Dict[str, str]:
    """
    Generate one validated-target Q&A pair in English from the given context.
    Returns question, answer, and supporting_text.
    """
    prompt = """You are an expert at creating chemistry and patent retrieval questions.

The source context may be in any language, but your output must be in English only.

Generate exactly ONE question-answer pair from the context.

Rules:
- Output must be in natural English only.
- Do not copy the source language unless a chemical name, formula, identifier, or proper noun should remain unchanged.
- The question must read like a realistic retrieval query that a researcher, engineer, or technical reader might actually type into a search system.
- Prefer short, natural, user-like wording over patent-summary wording.
- Prefer a specific question about one of these: purpose, application, composition, method step, property, technical advantage, operating condition, material relationship, mechanism, effect, or functional role.
- The question must be answerable from the text and specific enough to be useful for retrieval.
- Prefer semantically challenging questions that dense retrieval should handle better than simple keyword matching.
- For this task, semantic reformulation is more important than choosing the easiest extractive fact.
- First look for a question about rationale, role, effect, mechanism, interaction, implication, or process purpose tied to a specific step.
- Only fall back to exact ranges, exact ratios, exact named lists, or exact composition tables when the context does not support a stronger semantic question.
- If both are possible, prefer the question that requires understanding what the detail does or why it matters, not the one that only asks for the raw value.
- Prefer a question about one core technical fact, not a bundled summary of multiple advantages or multiple properties, unless the source presents them as one inseparable claim.
- Ask about function, effect, mechanism, role, use condition, or technical implication when possible, not just surface wording.
- Vary the question form across examples. Do not default to "How does ..." if another natural opening fits the fact better.
- Match the question opening to the fact type. Use forms such as:
  - "Why is ..." for step rationale or process purpose tied to a specific step
  - "Which ..." for identified biomarker pairs, materials, components, or options
  - "What function does ..." for a component's role
  - "What condition ..." or "At what ..." for operating constraints or measured ranges
  - "What property allows ..." for enabling characteristics
  - "How does ..." for mechanism, effect, or interaction only when that is the most natural form
- If you choose a method-style question, name the actual step, material, condition, or operation from the context.
- Do not ask vague questions like "What is the purpose of the method?" or "What is something about the process?" when the method contains a specific named step that can be asked about directly.
- Avoid making the question easy for exact-match retrieval by simply lifting the most distinctive nouns from the source into a template question.
- Preserve technical terms only when they are necessary for faithfulness or the question would become unnatural or ambiguous without them.
- Prefer grounded paraphrase over direct lexical overlap.
- Avoid spec-sheet questions when a more semantic alternative exists, especially:
  - exact wt% or mol-ratio lookups
  - exact temperature, density, time, or concentration range lookups
  - exact component inventory or long named-list lookups
  - exact "what does the composition contain" questions
- A numeric question is acceptable only when the number itself is the important retrieval target and the context does not support a better question about function, rationale, or effect.
- Avoid broad fallback wording like:
  - "What is the purpose of ..."
  - "What advantages does ... offer ..."
  - "What benefits does ... provide ..."
  when you can instead ask what a step achieves, why it is done, what a component does, or what effect it has.
- Avoid turning classification/grouping text into a weak taxonomy question if the same text supports a more functional question.
- When the context contains both "what it is" and "what it does", prefer "what it does".
- Avoid generic questions such as:
  - "What is the main object of the invention?"
  - "What is the main feature of the invention?"
  - "What are the main components?"
  unless the text is too short for anything better.
- Avoid document-centered phrasing such as:
  - "described in the invention"
  - "mentioned in the invention"
  - "according to the invention"
  - "in the text"
  - "described in the text"
  - "described in the present disclosure"
  - "used in the invention"
  Rewrite those into natural user-style English instead.
- Do not begin the question with broad template wording such as:
  - "What is the application of ..."
  - "What are the advantages of ..."
  - "What is the benefit of ..."
  - "What is the main technical advantage of ..."
  - "What is the purpose of ..."
  - "What types of products ..."
  - "What is the role of ..." when it only asks for a broad use summary
- Avoid these patterns especially when a more specific question can be asked about:
  - one process step
  - one operating condition
  - one material property
  - one mechanism
  - one component interaction
- Do not turn the title into a question.
- Do not simply wrap a copied title phrase or copied noun phrase in a question template.
- If a title-like wording comes to mind first, rewrite it into a more natural and more semantically reformulated question.
- Do not copy a sentence from the context nearly verbatim.
- Do not just wrap a copied noun phrase in a question template.
- Do not keep unusually high word overlap with the opening sentence or title unless a few technical anchor terms must remain for clarity.
- Do not make the question artificially difficult or obscure just to reduce word overlap.
- Before finalizing, ask yourself:
  - Did I choose the deepest answerable fact rather than the easiest extractive fact?
  - Would this still look like a good query if the exact numbers or list items were hidden?
  - Does this require some semantic understanding rather than simple table lookup?
  - Did I accidentally ask for a broad purpose/advantage summary when a narrower technical question was available?
- If the question starts with "What is the purpose of" or "What advantages does", rewrite it unless no narrower question is possible.
- If the answer to those checks is no, regenerate a better question.
- The answer must be concise (1-2 sentences) and strictly grounded in the context.
- Include a short supporting_text quote copied from the source context that justifies the answer.
- Include a question_type chosen from: purpose, application, composition, method, property, advantage, operating_condition, material_relationship, other.
- Good style examples:
  - "How does the treatment improve hair growth when heat is applied afterward?"
  - "Where would these microcapsules be used in fragranced consumer goods?"
  - "What does the shape deformation layer do when the artificial nail is pressed onto the natural nail?"
  - "Why is cold rolling performed after hot rolling or forging in this steel production process?"
  - "What function does sodium bicarbonate serve in the enteric coating composition?"
  - "Which biomarker pairs are measured to assess early-onset preeclampsia risk?"
  - "At what density is the mixed solution evaporated before filtration?"
  - "What property of the glass substrate supports fine pattern formation?"
- Bad style examples:
  - "What are the recommended application methods for the preparation described in the invention?"
  - "What type of products can include the microcapsules mentioned in the invention?"
  - "What is the application of 8-(4-trifluoromethoxy)benzyloamino-2'-deoxyadenosine?"
  - "What types of products can utilize the hair dye composition described in the text?"
  - "What is the role of the benzoxazole derivatives in detecting GHB in beverages?"
  - "What is the main technical advantage of this method?"
  - "What is the purpose of the process?"
  - "How does the method work?" when the context supports a more specific `Why`, `Which`, `What function`, `What condition`, or `At what` question
  - "What SiO2/Li2O and SiO2/Al2O3 mol ratios does the glass composition require?" when the context also supports a better question about why the composition enables the target property
  - "What are the specified weight percent ranges for silicon and manganese?" when the context also supports a better question about the role or effect of the composition
  - "Which specific fungicides are named as component (2)?" when the context also supports a better question about selection logic, interaction, or functional grouping
  - "What is the purpose of flowing a portion of metal-rich produced water to an evaporation area ...?" when the better question is about what this step causes or why it enables metal recovery
  - "What advantages does the DNA oligonucleotide ... offer ...?" when the better question is about the concrete storage or biosafety property
- Output valid JSON only, no markdown:
  {"question": "...", "answer": "...", "supporting_text": "...", "question_type": "..."}
"""
    retry_note = ""
    if previous_feedback:
        retry_note = (
            "\n\nPrevious attempt issue to fix:\n"
            f"{previous_feedback}\n"
            "Regenerate the question and answer so they fix that issue while staying fully grounded in the context."
        )
    previous_attempt_note = ""
    if previous_question or previous_answer:
        previous_attempt_note = (
            "\n\nPrevious failed attempt to improve upon:\n"
            f"Previous question: {previous_question or ''}\n"
            f"Previous answer: {previous_answer or ''}\n"
            "Use this only as feedback about what to avoid or improve. Do not lightly edit it or reuse its wording as a template. Generate a fresh corrected question-answer pair."
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n\n{context[:4000]}"
                    f"{retry_note}"
                    f"{previous_attempt_note}"
                ),
            },
        ],
        reasoning_effort=DEFAULT_GENERATION_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    return {
        "question": str(data.get("question", "")).strip(),
        "answer": str(data.get("answer", "")).strip(),
        "supporting_text": str(data.get("supporting_text", "")).strip(),
        "question_type": str(data.get("question_type", "other")).strip(),
    }


def check_english_language(
    client: OpenAI,
    question: str,
    answer: str,
    *,
    model: str = DEFAULT_SUPPORT_MODEL,
) -> Tuple[bool, str]:
    """
    Validate that the generated question and answer are written in English.
    Returns (approved, reason).
    """
    prompt = """You are a strict language checker.

Decide whether BOTH the question and answer are written mainly in English.

Approve only if:
- both are natural English,
- they are not primarily written in another language,
- they are not mixed-language outputs except for unavoidable chemical names, formulas, identifiers, or proper nouns.

Output valid JSON only:
{"approved": true, "reason": "..."}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Question: {question}\n\nAnswer: {answer}",
            },
        ],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    approved = bool(data.get("approved", False))
    reason = str(data.get("reason", "")).strip()
    return approved, reason


def check_faithfulness(
    client: OpenAI,
    context: str,
    question: str,
    answer: str,
    supporting_text: str,
    *,
    model: str = DEFAULT_SUPPORT_MODEL,
) -> Tuple[bool, str]:
    """
    Validate that the answer is supported by the source context.
    Returns (approved, reason).
    """
    prompt = """You are a strict faithfulness checker for patent question-answer pairs.

Approve only if:
- the question is answerable from the context,
- the answer is fully supported by the context,
- the answer does not add unsupported details,
- the supporting_text is relevant evidence from the context.

Reject if the answer is generic, speculative, partially unsupported, or not clearly grounded.

Output valid JSON only:
{"approved": true, "reason": "..."}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context[:5000]}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer: {answer}\n\n"
                    f"Supporting text: {supporting_text}"
                ),
            },
        ],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    approved = bool(data.get("approved", False))
    reason = str(data.get("reason", "")).strip()
    return approved, reason


def check_question_quality(
    client: OpenAI,
    context: str,
    question: str,
    answer: str,
    *,
    model: str = DEFAULT_QUALITY_MODEL,
) -> Tuple[bool, str]:
    """
    Validate that the question is retrieval-useful, specific, and not overly generic.
    Returns (approved, reason).
    """
    prompt = """You are a strict quality checker for retrieval questions built from technical patent text.

Approve only if the question:
- sounds like a realistic search or retrieval query,
- is specific enough to distinguish the document,
- asks about a concrete technical point from the context,
- uses natural user-like wording rather than patent-summary wording,
- is phrased semantically rather than as an obvious exact-match template,
- would be easier for a strong semantic retriever than for naive keyword matching,
- is not too generic,
- is not nearly copied from the context verbatim,
- and is useful for retrieval benchmarking.

Reject questions that are broad or repetitive patterns such as:
- "What is the main object of the invention?"
- "What is the main feature?"
- "What are the main components?"
- "What are the applications ...?" when a more specific application question is possible
- "What is the composition ...?" when a more targeted material or component question is possible
- "What is the main technical advantage ...?" when a narrower effect, property, operating condition, or mechanism question is possible
- "What is the advantage ...?" when the answer would bundle multiple benefits instead of one fact
- "What is the purpose ...?" when the question does not name a specific step, component, material, or operation
- "What advantages does ... offer ...?" when the answer mainly bundles several benefits that could be asked about more concretely
unless the context is too short for a better question.

Also reject questions that:
- mostly reuse a title phrase or a distinctive noun phrase from the source with only light reformatting,
- depend mainly on exact keyword overlap rather than semantic understanding,
- ask directly for the name, application, or advantage of a named entity when a more functional or effect-based question is possible.
- ask vaguely about "the method" or "the process" without identifying what part of it is being asked about, even though the context contains a more specific step or condition.
- are primarily spec-sheet or table-lookup questions when the same context supports a better semantic question about rationale, role, effect, mechanism, implication, or process purpose.
- ask only for raw values, ranges, ratios, or enumerated lists even though the document provides enough context to ask what those details enable, affect, control, or explain.

Treat these as common extractive failure modes:
- exact wt% / mol-ratio lookup
- exact temperature / density / time / concentration range lookup
- exact ingredient inventory or long named-list lookup
- direct "what does the composition contain" lookup
- direct "what values are specified for X and Y" lookup

Do NOT reject all numeric questions automatically.
Approve them only when the number or range itself is genuinely the most retrieval-worthy fact in the context and no clearly better semantic question is available.

Also reject document-centered wording such as:
- "described in the invention"
- "mentioned in the invention"
- "according to the invention"
- "in the text"

Also reject broad template openings such as:
- "What is the application of ..."
- "What are the advantages of ..."
- "What advantages does ..."
- "What is the benefit of ..."
- "What is the main technical advantage of ..."
- "What is the purpose of ..."
- "What types of products ..."
- "What is the role of ..."
when they lead to a broad summary question instead of a sharper technical query.

Be especially strict about these two failure modes:
1. title-lift: the question is basically the title or first source phrase converted into a question
2. high-overlap paraphrase: the question keeps too much surface wording from the source and would still be easy for exact-match retrieval
3. overly-extractive: the question is safe and specific but mainly asks for a literal value/list/span rather than a semantic technical point that the same context supports
4. broad-summary: the question asks for a broad purpose/advantage/benefit summary instead of one narrower technical fact
5. bundled-facts: the question asks for multiple loosely related facts at once instead of one core information need
6. weak-query-shape: the question is understandable but not phrased like a strong retrieval query a user would naturally type

Approve borderline cases only if the question is clearly more natural, more specific, and less surface-aligned than those failure modes.

If you reject the question:
- set `failure_type` to exactly one of:
  - `title-lift`
  - `high-overlap`
  - `overly-extractive`
  - `broad-summary`
  - `bundled-facts`
  - `weak-query-shape`
- keep `reason` short and concrete
- provide `better_direction` as ONE short actionable hint for regeneration, for example:
  - `ask about why the step is used`
  - `ask about the component's role`
  - `ask about the effect, not the raw value`
  - `ask what the condition enables`
  - `focus on one narrower technical fact`

If you approve the question:
- set `failure_type` to `none`
- set `better_direction` to an empty string

Output valid JSON only:
{"approved": true, "reason": "...", "failure_type": "none", "better_direction": ""}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context[:5000]}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer: {answer}"
                ),
            },
        ],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    approved = bool(data.get("approved", False))
    reason = str(data.get("reason", "")).strip()
    failure_type = str(data.get("failure_type", "")).strip()
    better_direction = str(data.get("better_direction", "")).strip()
    if failure_type and failure_type != "none":
        reason = f"{failure_type}: {reason}" if reason else failure_type
    if better_direction:
        reason = f"{reason} Better direction: {better_direction}".strip()
    return approved, reason


def translate_qa(
    client: OpenAI,
    context: str,
    question: str,
    answer: str,
    target_langs: List[str],
    *,
    previous_feedback: Optional[str] = None,
    previous_translated_question: Optional[str] = None,
    previous_translated_answer: Optional[str] = None,
    model: str = DEFAULT_TRANSLATION_MODEL,
) -> Dict[str, Tuple[str, str]]:
    """
    Translate (question, answer) to target languages. Returns {lang: (q, a)}.
    """
    if not target_langs:
        return {}
    lang_list = ", ".join(LANG_NAMES.get(l, l) for l in target_langs)
    prompt = f"""Translate the following English retrieval question and answer pair into these languages: {lang_list}.

For each language, produce a natural, native-sounding, retrieval-style translation.
Use the source context to resolve ambiguity and preserve the original information need exactly.
Keep the same meaning, level of specificity, and technical terms where appropriate.
Do not make the question more generic than the original.
Preserve the semantic difficulty of the original question.
Do not simplify the question into a keyword-heavy or literal surface-form restatement.
Prefer natural target-language phrasing over word-for-word translation.
Do not omit or alter numbers, units, ranges, formulas, identifiers, or named technical materials.
Preserve chemical names, abbreviations, symbols, and patent-style identifiers when translating them would be incorrect or unnatural.
Keep the answer faithful to the English answer and consistent with the source context.
Do not add explanation, background, or extra claims not present in the English pair or source context.
If the English question is technical and concise, keep the target-language question technical and concise too.
Avoid translation artifacts:
- choose one natural term, not slash-separated alternatives like `X/Y`
- do not leave editor-style repair traces or synonym bundles
- do not include unnecessary English glosses in parentheses
- avoid code-mixed verbs or phrasing when the target language has a normal technical equivalent
- rewrite into natural target-language syntax instead of following English word order too closely
- keep the text fully in the target language except for unavoidable chemical names, formulas, units, identifiers, abbreviations, or proper nouns
- do not leak words from unrelated languages or scripts into the translation
- if a technical term can stay in Latin script, integrate it naturally into an otherwise target-language sentence
- if the English answer contains multiple supported facts, preserve them cleanly without turning the translation into a glossary or note
- prefer one polished final phrasing, not an exploratory or half-edited wording

Output valid JSON only:
{{"translations": {{"de": {{"question": "...", "answer": "..."}}, "fr": {{...}}, ...}}}}

Languages to include: {json.dumps(target_langs)}
"""
    retry_note = ""
    if previous_feedback:
        retry_note = (
            "\n\nPrevious attempt issue to fix:\n"
            f"{previous_feedback}\n"
            "Revise the translation to fix that issue while preserving meaning and technical details."
        )
    previous_attempt_note = ""
    if previous_translated_question or previous_translated_answer:
        previous_attempt_note = (
            "\n\nPrevious failed translation to improve upon:\n"
            f"Previous translated question: {previous_translated_question or ''}\n"
            f"Previous translated answer: {previous_translated_answer or ''}\n"
            "Use this only as repair context. Do not copy it mechanically. Rewrite it so it sounds more natural in the target language while preserving the exact information need, meaning, and technical details."
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Source context:\n{context[:5000]}\n\n"
                    f"English question: {question}\n\n"
                    f"English answer: {answer}"
                    f"{retry_note}"
                    f"{previous_attempt_note}"
                ),
            },
        ],
        reasoning_effort=DEFAULT_TRANSLATION_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    trans = data.get("translations", data)
    result: Dict[str, Tuple[str, str]] = {}
    for lang in target_langs:
        if lang in trans and isinstance(trans[lang], dict):
            q = trans[lang].get("question", "")
            a = trans[lang].get("answer", "")
            result[lang] = (str(q).strip(), str(a).strip())
    return result


def check_translation_quality(
    client: OpenAI,
    context: str,
    english_question: str,
    english_answer: str,
    translated_question: str,
    translated_answer: str,
    target_lang: str,
    *,
    model: str = DEFAULT_QUALITY_MODEL,
) -> Dict[str, Any]:
    """
    Validate that a translated QA pair is fluent, faithful, and in the target language.
    Returns structured quality signals for approval and retry decisions.
    """
    target_lang_name = LANG_NAMES.get(target_lang, target_lang)
    prompt = f"""You are a strict but practical translation quality checker for multilingual patent retrieval data.

The source context may be in any language. The reference question and answer are in English.
The candidate translation must be in {target_lang_name}.

Judge these dimensions separately:
- `language_ok`: the translated question and answer are clearly written in {target_lang_name}
- `meaning_ok`: the meaning matches the English question and English answer closely
- `technical_ok`: numbers, units, ranges, formulas, identifiers, and important technical terms are preserved
- `specificity_ok`: the translated question keeps the same information need and specificity and does not become more generic
- `terminology_ok`: the translation uses appropriate technical terminology and register for {target_lang_name}
- `artifact_ok`: the translation does not contain repair artifacts such as slash-separated alternatives, unnecessary English glosses, editor-style synonym bundles, or gratuitous code mixing
- `fluency_ok`: the translation sounds natural enough for a native technical reader and is not clearly word-for-word or grammatically broken
- `grammar_ok`: grammar, agreement, case, morphology, and local sentence form are acceptable for {target_lang_name}

Be especially strict about these artifact failures:
- slash alternatives like `X/Y` when one natural wording should be chosen
- parenthetical English glosses like `(oiling)` when they are not required for correctness
- mixed-language repair wording or unresolved synonym pairs
- foreign-script leakage from an unrelated language when the span is not just a formula, identifier, unit, abbreviation, or proper noun
- faithful but clearly literal syntax that still reads like English structure mapped into {target_lang_name}

Severity guidelines:
- `high`: wrong language, meaning drift, dropped or altered technical details, or much more generic wording
- `medium`: meaning is mostly correct but there are clear grammar problems or very awkward/literal phrasing
- `low`: minor stiffness or small fluency issues only

Choose exactly one `failure_type`:
- `none`
- `wrong-language`
- `meaning-error`
- `missing-technical-detail`
- `too-generic`
- `unnatural-phrasing`
- `grammar-morphology`
- `terminology-register`
- `translation-artifact`

If you reject:
- keep `reason` short and concrete
- provide `better_direction` as ONE short actionable repair hint
- examples:
  - `rewrite more naturally for native technical phrasing`
  - `fix grammar and agreement`
  - `restore the missing technical detail exactly`
  - `use the standard technical term in {target_lang_name}`
  - `keep the question as specific as the English original`
  - `choose one natural term instead of slash alternatives`
  - `remove the English gloss and use native technical wording`
  - `rewrite in natural {target_lang_name} syntax`

If you approve:
- set `failure_type` to `none`
- set `better_direction` to an empty string

Approval policy:
- Reject when `language_ok`, `meaning_ok`, `technical_ok`, or `specificity_ok` is false.
- Reject when `terminology_ok` is false.
- Reject when `artifact_ok` is false.
- Reject when `grammar_ok` is false and severity is `medium` or `high`.
- Reject when `fluency_ok` is false and severity is `medium` or `high`.
- Approve when the only issue is minor fluency stiffness with `severity = low`.

Output valid JSON only:
{{"language_ok": true, "meaning_ok": true, "technical_ok": true, "specificity_ok": true, "terminology_ok": true, "artifact_ok": true, "fluency_ok": true, "grammar_ok": true, "severity": "low", "failure_type": "none", "better_direction": "", "reason": "..."}}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context[:5000]}\n\n"
                    f"English question: {english_question}\n\n"
                    f"English answer: {english_answer}\n\n"
                    f"{target_lang_name} question: {translated_question}\n\n"
                    f"{target_lang_name} answer: {translated_answer}"
                ),
            },
        ],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
    )
    data = _parse_json_response(response.choices[0].message.content or "")
    language_ok = bool(data.get("language_ok", False))
    meaning_ok = bool(data.get("meaning_ok", False))
    technical_ok = bool(data.get("technical_ok", False))
    specificity_ok = bool(data.get("specificity_ok", False))
    terminology_ok = bool(data.get("terminology_ok", True))
    artifact_ok = bool(data.get("artifact_ok", True))
    fluency_ok = bool(data.get("fluency_ok", False))
    grammar_ok = bool(data.get("grammar_ok", True))
    severity = str(data.get("severity", "high")).strip().lower() or "high"
    if severity not in {"low", "medium", "high"}:
        severity = "high"
    reason = str(data.get("reason", "")).strip()
    failure_type = str(data.get("failure_type", "none")).strip().lower() or "none"
    if failure_type not in {
        "none",
        "wrong-language",
        "meaning-error",
        "missing-technical-detail",
        "too-generic",
        "unnatural-phrasing",
        "grammar-morphology",
        "terminology-register",
        "translation-artifact",
    }:
        failure_type = "meaning-error"
    better_direction = str(data.get("better_direction", "")).strip()
    approved = (
        language_ok
        and meaning_ok
        and technical_ok
        and specificity_ok
        and terminology_ok
        and artifact_ok
    )
    if approved and not grammar_ok and severity in {"medium", "high"}:
        approved = False
    if approved and not fluency_ok and severity in {"medium", "high"}:
        approved = False

    retry_recommended = (
        (not language_ok)
        or (not meaning_ok)
        or (not technical_ok)
        or (not specificity_ok)
        or (not terminology_ok)
        or (not artifact_ok)
        or ((not grammar_ok) and severity in {"medium", "high"})
        or ((not fluency_ok) and severity in {"medium", "high"})
    )
    return {
        "approved": approved,
        "retry_recommended": retry_recommended,
        "reason": reason,
        "severity": severity,
        "failure_type": failure_type,
        "better_direction": better_direction,
        "language_ok": language_ok,
        "meaning_ok": meaning_ok,
        "technical_ok": technical_ok,
        "specificity_ok": specificity_ok,
        "terminology_ok": terminology_ok,
        "artifact_ok": artifact_ok,
        "fluency_ok": fluency_ok,
        "grammar_ok": grammar_ok,
    }


def _process_sample_row(
    index: int,
    row: Dict[str, Any],
    *,
    target_languages: List[str],
    generation_model: str,
    quality_model: str,
    support_model: str,
    translation_model: str,
    max_attempts: int,
) -> Dict[str, Any]:
    corpus_id = row.get("id", "")
    context = row.get("context", row.get("abstract", "")) or row.get("title", "")
    if not context.strip():
        return {
            "index": index,
            "corpus_id": corpus_id,
            "rows": [],
            "status": "skipped (empty context)",
        }

    try:
        client = _get_client()
        approved = False
        q_en = ""
        a_en = ""
        supporting_text = ""
        question_type = ""
        last_failure = ""
        retry_feedback: Optional[str] = None
        retry_question: Optional[str] = None
        retry_answer: Optional[str] = None

        for _attempt in range(1, max_attempts + 1):
            generated = generate_qa_english(
                client,
                context,
                previous_question=retry_question,
                previous_answer=retry_answer,
                previous_feedback=retry_feedback,
                model=generation_model,
            )
            q_en = generated["question"]
            a_en = generated["answer"]
            supporting_text = generated["supporting_text"]
            question_type = generated["question_type"]
            retry_question = q_en
            retry_answer = a_en

            lang_ok, lang_reason = check_english_language(
                client,
                q_en,
                a_en,
                model=support_model,
            )
            if not lang_ok:
                last_failure = f"language check failed: {lang_reason or 'not English enough'}"
                retry_feedback = (
                    f"{last_failure}. The output must be natural English only."
                )
                continue

            faithful_ok, faithful_reason = check_faithfulness(
                client,
                context,
                q_en,
                a_en,
                supporting_text,
                model=support_model,
            )
            if not faithful_ok:
                last_failure = f"faithfulness check failed: {faithful_reason or 'not grounded enough'}"
                retry_feedback = (
                    f"{last_failure}. Remove unsupported details and keep the answer strictly grounded in the context."
                )
                continue

            quality_ok, quality_reason = check_question_quality(
                client,
                context,
                q_en,
                a_en,
                model=quality_model,
            )
            if not quality_ok:
                last_failure = f"quality check failed: {quality_reason or 'question not useful enough'}"
                retry_feedback = (
                    f"{last_failure}. Use the better direction above if present. Regenerate one fresh question that is more retrieval-useful, more specific, less generic, and less surface-aligned. Prefer one narrower technical fact over a broad summary or literal lookup."
                )
                continue

            approved = True
            break

        if not approved:
            return {
                "index": index,
                "corpus_id": corpus_id,
                "rows": [],
                "status": f"skipped ({last_failure or 'validation failed'})",
            }

        qac_rows = [{
            "corpus_id": corpus_id,
            "language": "en",
            "question": q_en,
            "answer": a_en,
        }]
        approved_translations: Dict[str, Tuple[str, str]] = {}
        failed_languages: List[str] = []
        for lang in target_languages:
            lang_failure = "translation missing"
            retry_feedback: Optional[str] = None
            retry_q: Optional[str] = None
            retry_a: Optional[str] = None
            for _attempt in range(1, max_attempts + 1):
                trans = translate_qa(
                    client,
                    context,
                    q_en,
                    a_en,
                    [lang],
                    previous_feedback=retry_feedback,
                    previous_translated_question=retry_q,
                    previous_translated_answer=retry_a,
                    model=translation_model,
                )
                if lang not in trans:
                    lang_failure = "translation missing"
                    retry_feedback = "The previous translation attempt was missing or malformed. Return valid JSON with a complete translated question and answer."
                    continue

                q, a = trans[lang]
                retry_q = q
                retry_a = a
                trans_check = check_translation_quality(
                    client,
                    context,
                    q_en,
                    a_en,
                    q,
                    a,
                    lang,
                    model=quality_model,
                )
                if not trans_check["approved"]:
                    reason = str(trans_check.get("reason", "")).strip()
                    severity = str(trans_check.get("severity", "high")).strip()
                    failure_type = str(trans_check.get("failure_type", "")).strip()
                    better_direction = str(trans_check.get("better_direction", "")).strip()
                    lang_failure = (
                        "translation quality failed: "
                        f"{reason or 'not fluent/faithful enough'}"
                        f" [severity={severity}]"
                    )
                    feedback_parts = []
                    if failure_type and failure_type != "none":
                        feedback_parts.append(f"Failure type: {failure_type}.")
                    if reason:
                        feedback_parts.append(f"Reason: {reason}.")
                    if better_direction:
                        feedback_parts.append(f"Better direction: {better_direction}.")
                    feedback_parts.append(
                        "Revise the translation by preserving the exact meaning, specificity, numbers, units, and technical details from the English pair and source context."
                    )
                    feedback_parts.append(
                        "If the problem is fluency or grammar, rewrite more naturally in the target language without changing the information need."
                    )
                    retry_feedback = " ".join(feedback_parts)
                    continue

                approved_translations[lang] = (q, a)
                lang_failure = ""
                break

            if lang_failure:
                failed_languages.append(f"{lang} ({lang_failure})")

        for lang, (q, a) in approved_translations.items():
            qac_rows.append({
                "corpus_id": corpus_id,
                "language": lang,
                "question": q,
                "answer": a,
            })
        translation_status = f"{len(approved_translations)} translations"
        if failed_languages:
            translation_status += f", skipped {len(failed_languages)}: {', '.join(failed_languages)}"
        return {
            "index": index,
            "corpus_id": corpus_id,
            "rows": qac_rows,
            "status": f"ok ({question_type or 'validated'} en + {translation_status})",
        }
    except Exception as exc:
        return {
            "index": index,
            "corpus_id": corpus_id,
            "rows": [],
            "status": f"error: {exc}",
        }


def run_qa_pipeline(
    corpus_path: Path,
    output_dir: Path,
    *,
    sample_size: int = 50,
    target_languages: Optional[List[str]] = None,
    model: Optional[str] = None,
    generation_model: str = DEFAULT_GENERATION_MODEL,
    quality_model: str = DEFAULT_QUALITY_MODEL,
    support_model: str = DEFAULT_SUPPORT_MODEL,
    translation_model: str = DEFAULT_TRANSLATION_MODEL,
    max_attempts: int = 3,
    batch_mode: bool = False,
) -> int:
    """
    Sample corpus, generate Q&A in English, translate to all target languages.
    Writes qac.csv (corpus_id, language, question, answer) to output_dir.
    Returns number of QAC rows written.
    """
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    target_languages = target_languages or DEFAULT_TARGET_LANGS
    if model is not None:
        generation_model = model
        quality_model = model
        support_model = model
        translation_model = model
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_corpus(corpus_path)
    sampled = sample_corpus(rows, sample_size, stratify_by_language=True, seed=42)
    print(f"Sampled {len(sampled)} documents from corpus ({len(rows)} total).")
    qac_rows: List[Dict[str, str]] = []
    results: List[Dict[str, Any]] = []

    if batch_mode and sampled:
        available_cpus = os.cpu_count() or 1
        workers = max(1, min(len(sampled), max(1, available_cpus // 2), 4))
        print(
            f"Running batched Q&A generation with {workers} worker(s) "
            f"based on {available_cpus} available CPU(s)."
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _process_sample_row,
                    index,
                    row,
                    target_languages=target_languages,
                    generation_model=generation_model,
                    quality_model=quality_model,
                    support_model=support_model,
                    translation_model=translation_model,
                    max_attempts=max_attempts,
                )
                for index, row in enumerate(sampled)
            ]
            progress = tqdm(as_completed(futures), total=len(futures), desc="Generate Q&A", unit="doc")
            for completed, future in enumerate(progress, start=1):
                result = future.result()
                results.append(result)
                tqdm.write(
                    f"  [{completed}/{len(sampled)}] {result['corpus_id']}... {result['status']}"
                )
    else:
        if sampled:
            print("Running Q&A generation in single-threaded mode.")
        progress = tqdm(sampled, total=len(sampled), desc="Generate Q&A", unit="doc")
        for index, row in enumerate(progress, start=1):
            result = _process_sample_row(
                index - 1,
                row,
                target_languages=target_languages,
                generation_model=generation_model,
                quality_model=quality_model,
                support_model=support_model,
                translation_model=translation_model,
                max_attempts=max_attempts,
            )
            results.append(result)
            tqdm.write(f"  [{index}/{len(sampled)}] {result['corpus_id']}... {result['status']}")

    for result in sorted(results, key=lambda item: item["index"]):
        qac_rows.extend(result["rows"])

    out_csv = output_dir / "qac.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["corpus_id", "language", "question", "answer"])
        w.writeheader()
        w.writerows(qac_rows)

    print(f"Wrote {len(qac_rows)} QAC rows -> {out_csv}")
    return len(qac_rows)
