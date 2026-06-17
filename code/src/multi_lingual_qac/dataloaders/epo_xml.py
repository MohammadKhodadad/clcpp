"""
Parsing slice for EPO `<ep-patent-document>` XML files (BDDS product 14.12).

Ported from the `EPO` branch's `epo.py` (commit 15f6fac). The local-zip workflow
(`extract_epo_xml_files`, `build_epo_corpus`) is intentionally not ported here;
the streaming ingester in `epo_bdds.py` drives parsing directly off in-memory
XML blobs.
"""

from __future__ import annotations

import html
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional


CHEMISTRY_CLASSIFICATION_PREFIXES = [
    "C",
    "A01N",
    "A23L",
    "A61K",
    "A61P",
    "B01D",
    "B01F",
    "B01J",
    "B01L",
    "C25",
    "G01N",
    "H01M",
]

CHEMISTRY_KEYWORDS = [
    "adhesive",
    "antibody",
    "battery",
    "biomarker",
    "catalyst",
    "cell culture",
    "chemical",
    "chemistry",
    "coating",
    "composition",
    "compound",
    "crystal form",
    "detergent",
    "drug",
    "electrolyte",
    "excipient",
    "fermentation",
    "formulation",
    "inhibitor",
    "material",
    "molecule",
    "nanoparticle",
    "peptide",
    "pharmaceutical",
    "pharmaceutically",
    "polymer",
    "protein",
    "resin",
    "semiconductor composition",
    "slurry",
    "solvent",
    "surfactant",
    "synthesis",
    "therapeutic",
]

CLASSIFICATION_CODE_RE = re.compile(r"([A-HY]\d{2}[A-Z]?\s*\d+(?:/\d+)?)")
AUXILIARY_XML_RE = re.compile(r"__(?:TOC|SL\d+)\.xml$", re.IGNORECASE)
DESCRIPTION_MAX_CHARS = 2000
FIRST_CLAIM_MAX_CHARS = 1500
MIN_ABSTRACT_WORDS = 50
PREFERRED_TEXT_LANGUAGES = ["fr", "de", "en"]


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = html.unescape(s)
    s = s.replace("﻿", " ").replace("­", "").replace("\xa0", " ")
    s = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([(\[{])\s+", r"\1", s)
    s = re.sub(r"\s+([)\]}])", r"\1", s)
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def word_count(text: str) -> int:
    return len((text or "").split())


def _normalize_code(raw_text: str) -> str:
    raw_text = clean_text(raw_text)
    match = CLASSIFICATION_CODE_RE.search(raw_text)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def _normalized_prefix(value: str) -> str:
    return value.upper().replace(" ", "")


def _truncate_text(text: str, *, max_chars: int) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    snippet = text[:max_chars].rsplit(" ", 1)[0].strip()
    return snippet or text[:max_chars].strip()


def _extract_title_localized(root: ET.Element) -> List[Dict[str, str]]:
    titles: List[Dict[str, str]] = []
    title_block = root.find(".//B540")
    if title_block is None:
        return titles

    current_lang = ""
    for child in title_block:
        if child.tag == "B541":
            current_lang = clean_text(child.text or "").lower()
        elif child.tag == "B542":
            text = clean_text(child.text or "")
            if current_lang and text:
                titles.append({"language": current_lang, "text": text})
    return titles


def _extract_text_blocks(root: ET.Element, tag_name: str) -> List[Dict[str, str]]:
    blocks: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for node in root.findall(f".//{tag_name}"):
        language = clean_text(node.attrib.get("lang", "")).lower()
        text = clean_text(" ".join("".join(node.itertext()).split()))
        if not text:
            continue
        key = (language, text)
        if key in seen:
            continue
        seen.add(key)
        blocks.append({"language": language, "text": text})
    return blocks


def _get_text_for_language(blocks: List[Dict[str, str]], language: str) -> str:
    language = (language or "").lower()
    for block in blocks:
        if block["language"] == language and block["text"]:
            return block["text"]
    return ""


def _extract_first_claim_text(root: ET.Element) -> List[Dict[str, str]]:
    claims_by_lang: Dict[str, str] = {}
    for claims_node in root.findall(".//claims"):
        language = clean_text(claims_node.attrib.get("lang", "")).lower()
        claim_node = claims_node.find(".//claim")
        if claim_node is None:
            continue
        text = clean_text(" ".join("".join(claim_node.itertext()).split()))
        if text and language not in claims_by_lang:
            claims_by_lang[language] = _truncate_text(text, max_chars=FIRST_CLAIM_MAX_CHARS)
    return [{"language": lang, "text": text} for lang, text in claims_by_lang.items()]


def _extract_classification_codes(root: ET.Element, tag_name: str) -> List[str]:
    codes: List[str] = []
    seen: set[str] = set()
    for node in root.findall(f".//{tag_name}"):
        code = _normalize_code(node.findtext("text", default=""))
        if code and code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def _extract_party_names(root: ET.Element, xpath: str) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()
    for node in root.findall(xpath):
        name = clean_text(node.findtext("snm", default=""))
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _extract_designated_states(root: ET.Element) -> List[str]:
    states: List[str] = []
    seen: set[str] = set()
    for node in root.findall(".//B840/ctry"):
        state = clean_text(node.text or "").upper()
        if state and state not in seen:
            seen.add(state)
            states.append(state)
    return states


def _extract_priority_numbers(root: ET.Element) -> List[str]:
    numbers: List[str] = []
    seen: set[str] = set()
    for node in root.findall(".//B300/B310"):
        number = clean_text("".join(node.itertext()))
        if number and number not in seen:
            seen.add(number)
            numbers.append(number)
    return numbers


def _extract_priority_dates(root: ET.Element) -> List[str]:
    dates: List[str] = []
    seen: set[str] = set()
    for node in root.findall(".//B300/B320/date"):
        date_value = clean_text(node.text or "")
        if date_value and date_value not in seen:
            seen.add(date_value)
            dates.append(date_value)
    return dates


def _has_chemistry_classification(codes: List[str]) -> List[str]:
    matches: List[str] = []
    prefixes = [_normalized_prefix(prefix) for prefix in CHEMISTRY_CLASSIFICATION_PREFIXES]
    for code in codes:
        normalized = _normalized_prefix(code)
        if any(normalized.startswith(prefix) for prefix in prefixes):
            matches.append(code)
    return matches


def _keyword_hits(texts: List[str]) -> List[str]:
    haystack = " ".join(clean_text(text).lower() for text in texts if text)
    return [keyword for keyword in CHEMISTRY_KEYWORDS if keyword in haystack]


def analyze_epo_chemistry(record: Dict[str, Any]) -> Dict[str, Any]:
    """Score whether a parsed EPO record is chemistry-related."""
    ipc_matches = _has_chemistry_classification(record.get("ipc_codes", []))
    cpc_matches = _has_chemistry_classification(record.get("cpc_codes", []))
    keyword_hits = _keyword_hits([title["text"] for title in record.get("title_localized", [])])

    score = 0
    reasons: List[str] = []
    if ipc_matches:
        score += 2
        reasons.append(f"IPC match: {', '.join(ipc_matches[:5])}")
    if cpc_matches:
        score += 2
        reasons.append(f"CPC match: {', '.join(cpc_matches[:5])}")
    if keyword_hits:
        score += 1
        reasons.append(f"Title keywords: {', '.join(keyword_hits[:8])}")

    if ipc_matches or cpc_matches:
        label = "chemistry_core"
    elif keyword_hits:
        label = "chemistry_related"
    else:
        label = "not_chemistry"

    return {
        "score": score,
        "label": label,
        "keep": label != "not_chemistry",
        "reasons": reasons,
    }


def parse_epo_patent_root(root: ET.Element, *, xml_name: str = "") -> Dict[str, Any]:
    """Build a normalized record from an already-parsed `<ep-patent-document>` root.

    Used by the streaming ingester where the XML bytes come from a remote zip
    rather than a local file.
    """
    if root.tag != "ep-patent-document":
        raise ValueError(f"Unsupported EPO XML root tag: {root.tag}")

    title_localized = _extract_title_localized(root)
    source_language = clean_text(root.attrib.get("lang", "")).lower()
    english_title = next((item["text"] for item in title_localized if item["language"] == "en"), "")
    primary_title = english_title or (title_localized[0]["text"] if title_localized else "")

    record: Dict[str, Any] = {
        "xml_file": xml_name,
        "document_id": clean_text(root.attrib.get("id", "")),
        "publication_number": clean_text(root.attrib.get("doc-number", "")),
        "application_number": clean_text(root.findtext(".//B200/B210", default="")),
        "country_code": clean_text(root.attrib.get("country", ""))
            or clean_text(root.findtext(".//B100/B190", default="")),
        "publication_date": clean_text(root.attrib.get("date-publ", ""))
            or clean_text(root.findtext(".//B140/date", default="")),
        "filing_date": clean_text(root.findtext(".//B220/date", default="")),
        "priority_dates": _extract_priority_dates(root),
        "priority_numbers": _extract_priority_numbers(root),
        "kind": clean_text(root.attrib.get("kind", "")),
        "source_language": source_language,
        "title": primary_title,
        "title_localized": title_localized,
        "abstract_localized": _extract_text_blocks(root, "abstract"),
        "description_localized": _extract_text_blocks(root, "description"),
        "first_claim_localized": _extract_first_claim_text(root),
        "ipc_codes": _extract_classification_codes(root, "classification-ipcr"),
        "cpc_codes": _extract_classification_codes(root, "classification-cpc"),
        "applicants": _extract_party_names(root, ".//B710/B711"),
        "inventors": _extract_party_names(root, ".//B720/B721"),
        "representatives": _extract_party_names(root, ".//B740/B741"),
        "designated_states": _extract_designated_states(root),
    }

    if not record["document_id"] or not record["publication_number"]:
        raise ValueError(f"Missing essential patent identifiers in {xml_name or '<stream>'}")
    if not record["title_localized"] and not record["title"]:
        raise ValueError(f"Missing title data in {xml_name or '<stream>'}")

    record["chemistry"] = analyze_epo_chemistry(record)
    return record


def parse_epo_patent_bytes(xml_bytes: bytes, *, xml_name: str = "") -> Dict[str, Any]:
    """Parse EPO XML bytes (e.g. from a remote zip stream) into a record dict."""
    root = ET.fromstring(xml_bytes)
    return parse_epo_patent_root(root, xml_name=xml_name)


def parse_epo_patent_xml(xml_path: Path) -> Dict[str, Any]:
    """Parse a local EPO patent XML file into a record dict.

    Retained for symmetry with the EPO-branch API; the streaming ingester uses
    `parse_epo_patent_bytes` instead.
    """
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()
    return parse_epo_patent_root(root, xml_name=xml_path.name)


def language_has_substantive_text(
    record: Dict[str, Any],
    language: str,
    *,
    min_abstract_words: int = MIN_ABSTRACT_WORDS,
) -> bool:
    """True when the record has a non-trivial abstract OR first-claim in `language`.

    Title-only presence is not enough — EPO publishes titles in all three official
    languages for almost every doc, which would make the multilingual filter trivial.
    """
    abstract = _get_text_for_language(record["abstract_localized"], language)
    if word_count(abstract) >= min_abstract_words:
        return True
    first_claim = _get_text_for_language(record["first_claim_localized"], language)
    if word_count(first_claim) >= 10:
        return True
    return False


def build_row_for_language(record: Dict[str, Any], language: str) -> Optional[Dict[str, str]]:
    """Build a single-language row matching the multilingual_corpus.csv schema.

    Returns None when the language slot has no substantive text.
    """
    language = (language or "").lower()
    title = _get_text_for_language(record["title_localized"], language)
    abstract = _get_text_for_language(record["abstract_localized"], language)
    first_claim = _get_text_for_language(record["first_claim_localized"], language)
    description = _get_text_for_language(record["description_localized"], language)

    if not abstract and not first_claim:
        return None

    if description:
        description = _truncate_text(description, max_chars=DESCRIPTION_MAX_CHARS)

    context_parts: List[str] = []
    if title:
        context_parts.append(f"Title: {title}")
    if abstract:
        context_parts.append(f"Abstract: {abstract}")
    if first_claim:
        context_parts.append(f"First claim: {first_claim}")
    context = "\n\n".join(context_parts).strip()

    pub_num = record["publication_number"]
    return {
        "id": f"{pub_num}_{language}",
        "language": language,
        "title": title,
        "abstract": abstract,
        "description": description,
        "first_claim": first_claim,
        "context": context,
        "publication_number": pub_num,
        "country_code": record.get("country_code", ""),
        "publication_date": record.get("publication_date", ""),
        "source": "epo",
        "ipc_codes": "|".join(record.get("ipc_codes", [])),
    }
