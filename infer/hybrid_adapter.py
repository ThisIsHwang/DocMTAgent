from __future__ import annotations

import re
from typing import TypedDict


class TMExample(TypedDict):
    source: str
    target: str


class DocumentMemoryPacket(TypedDict, total=False):
    doc_id: str
    sentence_index: int
    document_src_summary: str
    document_tgt_summary: str
    historical_terminology: str
    previous_source_context: list[str]
    previous_target_context: list[str]
    retrieved_examples: list[TMExample]
    has_summary: bool
    has_history: bool
    has_context: bool
    has_tm_examples: bool
    previous_terminology_conflict: bool


def _clean_text(value: str | None) -> str:
    return (value or "").strip()


def _format_list(items: list[str] | None) -> str:
    if not items:
        return "N/A"
    lines = [f"{idx}. {item.strip()}" for idx, item in enumerate(items, start=1) if item.strip()]
    return "\n".join(lines) if lines else "N/A"


def _format_retrieved_examples(examples: list[TMExample] | None) -> str:
    if not examples:
        return "N/A"

    lines: list[str] = []
    for idx, example in enumerate(examples, start=1):
        src = _clean_text(example.get("source"))
        tgt = _clean_text(example.get("target"))
        if src or tgt:
            lines.append(
                f"{idx}. Source: {src or 'N/A'}\n"
                f"   Target: {tgt or 'N/A'}"
            )
    return "\n".join(lines) if lines else "N/A"


def format_tm_examples(rel_src_sents: list[str] | None, rel_tgt_sents: list[str] | None) -> str:
    examples: list[TMExample] = []
    for src, tgt in zip(rel_src_sents or [], rel_tgt_sents or []):
        examples.append({"source": src, "target": tgt})
    return _format_retrieved_examples(examples)


def format_history(hist_info: str | None) -> str:
    return _clean_text(hist_info) or "N/A"


def format_summaries(src_summary: str | None, tgt_summary: str | None) -> str:
    src_value = _clean_text(src_summary) or "N/A"
    tgt_value = _clean_text(tgt_summary) or "N/A"
    return (
        f"Document source summary: {src_value}\n"
        f"Document target summary: {tgt_value}"
    )


def format_prev_context(src_context: list[str] | None, tgt_context: list[str] | None) -> str:
    return (
        f"Previous source context:\n{_format_list(src_context)}\n\n"
        f"Previous target context:\n{_format_list(tgt_context)}"
    )


def build_thintactic_state(
    *,
    src: str,
    src_lang_name: str,
    tgt_lang_name: str,
    packet: DocumentMemoryPacket,
) -> dict:
    retrieved_examples = list(packet.get("retrieved_examples", []))
    rel_src = [example.get("source", "") for example in retrieved_examples]
    rel_tgt = [example.get("target", "") for example in retrieved_examples]

    src_summary = packet.get("document_src_summary", "")
    tgt_summary = packet.get("document_tgt_summary", "")
    historical_terminology = packet.get("historical_terminology", "")
    previous_source_context = list(packet.get("previous_source_context", []))
    previous_target_context = list(packet.get("previous_target_context", []))

    return {
        "source_text": src,
        "source_language": src_lang_name,
        "target_language": tgt_lang_name,
        "doc_id": packet.get("doc_id", "default"),
        "sentence_index": packet.get("sentence_index", 0),
        "document_mode": True,
        "context_mode": "grounded",
        "document_src_summary": src_summary,
        "document_tgt_summary": tgt_summary,
        "historical_terminology": historical_terminology,
        "previous_source_context": previous_source_context,
        "previous_target_context": previous_target_context,
        "retrieved_examples": retrieved_examples,
        "few_shot_examples": format_tm_examples(rel_src, rel_tgt),
        "pre_translation_research": format_history(historical_terminology),
        "context_analysis": format_summaries(src_summary, tgt_summary),
        "extended_context": format_prev_context(previous_source_context, previous_target_context),
        "iteration": 0,
    }


def _token_count(text: str) -> int:
    tokens = re.findall(r"\w+|[^\w\s]", text or "", flags=re.UNICODE)
    return len(tokens)


def choose_workflow(packet: DocumentMemoryPacket, src: str) -> str:
    historical_terminology = _clean_text(packet.get("historical_terminology"))
    src_summary = _clean_text(packet.get("document_src_summary"))
    tgt_summary = _clean_text(packet.get("document_tgt_summary"))
    previous_source_context = packet.get("previous_source_context") or []
    previous_target_context = packet.get("previous_target_context") or []
    retrieved_examples = packet.get("retrieved_examples") or []
    token_count = _token_count(src)

    if historical_terminology:
        return "tactic-full"
    if token_count >= 30:
        return "tactic-full"
    if not retrieved_examples and (src_summary or tgt_summary):
        return "tactic-full"
    if packet.get("previous_terminology_conflict"):
        return "tactic-full"

    if retrieved_examples:
        return "tactic-base"
    if token_count >= 12:
        return "tactic-base"
    if previous_source_context or previous_target_context:
        return "tactic-base"

    return "tactic-lite"
