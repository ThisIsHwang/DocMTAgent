from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

try:
    from delta_memory import (
        base_lang,
        init_resume_state,
        language_name,
        sep_for_lang,
    )
    from hybrid_adapter import DocumentMemoryPacket, build_thintactic_state, choose_workflow
except ImportError:
    from .delta_memory import (
        base_lang,
        init_resume_state,
        language_name,
        sep_for_lang,
    )
    from .hybrid_adapter import DocumentMemoryPacket, build_thintactic_state, choose_workflow


VALID_SETTINGS = ("summary", "long", "context", "history")
VALID_MODEL_PLATFORMS = (
    "openai",
    "deepseek",
    "qwen",
    "vertexai",
    "openai_compatible",
    "vllm",
)
VALID_WORKFLOWS = ("auto", "tactic-lite", "tactic-base", "tactic-full")
VALID_DRAFT_MODES = ("zero_shot", "few_shot", "multi_strategy")


def _parse_language_pair(language_pair: str) -> tuple[str, str]:
    if "-" not in (language_pair or ""):
        raise ValueError(f"Language pair must look like 'en-ko_KR', got: {language_pair}")
    src_lang, tgt_lang = language_pair.split("-", 1)
    return (src_lang, tgt_lang)


def _load_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file]


def _chunk_lines(lines: list[str], chunk_size: int, lang_code: str) -> tuple[list[str], list[list[int]]]:
    if chunk_size < 1:
        raise ValueError(f"chunk_size_sentences must be >= 1, got {chunk_size}")

    joiner = sep_for_lang(lang_code)
    chunks: list[str] = []
    chunk_indices: list[list[int]] = []
    for start in range(0, len(lines), chunk_size):
        block = lines[start : start + chunk_size]
        chunks.append(joiner.join(block).strip())
        chunk_indices.append(list(range(start, min(start + chunk_size, len(lines)))))
    return (chunks, chunk_indices)


def _default_prompt_path(prompt_name: str, src_lang: str, tgt_lang: str) -> Path:
    prompt_root = Path(__file__).resolve().parents[1] / "prompts"
    pair_dir = prompt_root / f"{base_lang(src_lang)}-{base_lang(tgt_lang)}"
    pair_prompt_names = {
        "src_summary_prompt",
        "tgt_summary_prompt",
        "src_merge_prompt",
        "tgt_merge_prompt",
        "history_prompt",
    }
    if prompt_name in pair_prompt_names:
        return pair_dir / f"{prompt_name}.txt"
    return prompt_root / "retrieve_prompt.txt"


def _resolve_prompt_bundle(args: argparse.Namespace, src_lang: str, tgt_lang: str) -> dict[str, str]:
    bundle: dict[str, str] = {}
    prompt_names = (
        "src_summary_prompt",
        "tgt_summary_prompt",
        "src_merge_prompt",
        "tgt_merge_prompt",
        "retrieve_prompt",
        "history_prompt",
    )
    required_by_setting = {
        "summary": ("src_summary_prompt", "tgt_summary_prompt", "src_merge_prompt", "tgt_merge_prompt"),
        "long": ("retrieve_prompt",),
        "history": ("history_prompt",),
    }
    active_settings = set(args.settings or [])
    required_prompts = {
        prompt_name
        for setting, prompt_names_for_setting in required_by_setting.items()
        if setting in active_settings and not (setting == "long" and args.retriever == "none")
        for prompt_name in prompt_names_for_setting
    }

    for prompt_name in prompt_names:
        provided = getattr(args, prompt_name, None)
        resolved = Path(provided) if provided else _default_prompt_path(prompt_name, src_lang, tgt_lang)
        if prompt_name not in required_prompts and not resolved.is_file():
            continue
        if not resolved.is_file():
            raise FileNotFoundError(f"Required prompt file not found for {prompt_name}: {resolved}")
        bundle[prompt_name] = resolved.read_text(encoding="utf-8")
    return bundle


def _parse_json_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}

    candidate = Path(value)
    raw = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("extra_config_json must be a JSON object")
    return parsed


def _save_json_array(path: str, records: list[dict[str, Any]]) -> None:
    Path(path).write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _ensure_thintactic_import_path(thin_tactic_root: str | None) -> Path:
    if thin_tactic_root:
        root = Path(thin_tactic_root).expanduser().resolve()
    else:
        root = (Path(__file__).resolve().parents[2] / "ThinTACTIC").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ThinTACTIC root not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


class AsyncPromptRunner:
    def __init__(self, configurable: dict[str, Any]) -> None:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from tactic.models.llm import get_llm_from_config

        self._AIMessage = AIMessage
        self._HumanMessage = HumanMessage
        self._SystemMessage = SystemMessage
        self._llm = get_llm_from_config({"configurable": configurable})

    async def __call__(self, messages: list[dict[str, Any]], **params: Any) -> str | None:
        invoke_kwargs: dict[str, Any] = {}
        response_format = params.get("response_format")
        if response_format:
            if getattr(self._llm, "_llm_type", "") == "vertexai_google_genai":
                schema = (response_format.get("json_schema") or {}).get("schema")
                if schema is not None:
                    invoke_kwargs["response_mime_type"] = "application/json"
                    invoke_kwargs["response_json_schema"] = schema
            else:
                invoke_kwargs["response_format"] = response_format

        langchain_messages = []
        for message in messages:
            role = str(message.get("role", "user")).lower()
            content = message.get("content", "")
            if role == "system":
                langchain_messages.append(self._SystemMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(self._AIMessage(content=content))
            else:
                langchain_messages.append(self._HumanMessage(content=content))

        response = await self._llm.ainvoke(langchain_messages, **invoke_kwargs)
        return self._message_content_to_text(response.content)

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text = item
                elif isinstance(item, dict):
                    text = item.get("text", "")
                else:
                    text = getattr(item, "text", str(item))
                if text:
                    parts.append(str(text))
            return "\n".join(parts).strip()
        return str(content).strip()


def _build_helper_configurable(args: argparse.Namespace, extra_config: dict[str, Any]) -> dict[str, Any]:
    configurable = {
        "model_platform": args.model_platform,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    configurable.update(extra_config)
    return configurable


def _build_tactic_extra_config(args: argparse.Namespace, extra_config: dict[str, Any]) -> dict[str, Any]:
    tactic_extra = dict(extra_config)
    tactic_extra["json_schema_mode"] = bool(args.json_schema_mode)
    if args.eval_model_platform:
        tactic_extra["eval_model_platform"] = args.eval_model_platform
    if args.eval_model_name:
        tactic_extra["eval_model_name"] = args.eval_model_name
    if args.eval_temperature is not None:
        tactic_extra["eval_temperature"] = args.eval_temperature
    if args.eval_max_tokens is not None:
        tactic_extra["eval_max_tokens"] = args.eval_max_tokens
    return tactic_extra


def _build_workflow_config(args: argparse.Namespace, workflow_name: str) -> dict[str, Any]:
    from tactic.graph.workflow import WORKFLOW_PRESETS

    if workflow_name not in WORKFLOW_PRESETS:
        raise ValueError(f"Unsupported workflow: {workflow_name}")

    workflow_config = dict(WORKFLOW_PRESETS[workflow_name])
    workflow_config["send_feedback"] = bool(args.send_feedback)
    workflow_config["quality_threshold"] = args.quality_threshold
    workflow_config["max_iterations"] = args.max_iterations
    if args.draft_mode:
        workflow_config["draft_mode"] = args.draft_mode
    return workflow_config


def _build_record(
    *,
    idx: int,
    sentence_index: int,
    sentence_indices: list[int],
    src: str,
    ref: str | None,
    hyp: str,
    workflow_name: str,
    rel_src: list[str] | None,
    rel_tgt: list[str] | None,
    hist_info: str | None,
    result: dict[str, Any],
    flow_logging: bool,
) -> dict[str, Any]:
    return {
        "idx": idx,
        "sentence_index": sentence_index,
        "source_line_indices": sentence_indices,
        "src": src,
        "ref": ref,
        "hyp": hyp,
        "engine": "thintactic-hybrid",
        "workflow": workflow_name,
        "rel_src": rel_src,
        "rel_tgt": rel_tgt,
        "hist_info": hist_info,
        "entity_dict": None,
        "tactic_overall_score": result.get("overall_score"),
        "tactic_best_score": result.get("best_score", result.get("overall_score")),
        "tactic_feedback": result.get("feedback"),
        "tactic_iterations": result.get("iteration"),
        "new_src_summary": None,
        "new_tgt_summary": None,
        "flow_log": result.get("flow_log") if flow_logging else None,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.retriever == "embedding":
        raise ValueError("Hybrid mode does not support --retriever embedding.")
    if args.chunk_size_sentences < 1:
        raise ValueError("--chunk_size_sentences must be >= 1.")
    if args.summary_step < 1:
        raise ValueError("--summary_step must be >= 1.")
    if "context" in (args.settings or []) and args.context_window == 0:
        raise ValueError("--context_window must be non-zero when context memory is enabled.")
    if args.eval_model_platform and not args.eval_model_name:
        raise ValueError("--eval_model_name is required when --eval_model_platform is set.")


async def main_async(args: argparse.Namespace) -> None:
    _validate_args(args)
    _ensure_thintactic_import_path(args.thin_tactic_root)

    from tactic.api import run_tactic_state

    src_lang_code, tgt_lang_code = _parse_language_pair(args.language)
    src_lang_name = language_name(src_lang_code)
    tgt_lang_name = language_name(tgt_lang_code)

    src_lines = _load_lines(args.src)
    ref_lines = _load_lines(args.ref) if args.ref else None
    if ref_lines is not None and len(ref_lines) != len(src_lines):
        raise ValueError(
            f"Source and reference line counts must match. src={len(src_lines)} ref={len(ref_lines)}"
        )

    src_chunks, chunk_indices = _chunk_lines(src_lines, args.chunk_size_sentences, src_lang_code)
    ref_chunks = None
    if ref_lines is not None:
        ref_chunks, _ = _chunk_lines(ref_lines, args.chunk_size_sentences, tgt_lang_code)

    prompt_bundle = _resolve_prompt_bundle(args, src_lang_code, tgt_lang_code)

    if "context" not in args.settings:
        args.context_window = 0

    trans_context, long_memory, doc_summary, ent_history, trans_records = init_resume_state(
        args,
        prompt_bundle,
        src_lang_code,
        tgt_lang_code,
    )

    if len(trans_records) >= len(src_chunks):
        return

    extra_config = _parse_json_dict(args.extra_config_json)
    helper_runner = AsyncPromptRunner(_build_helper_configurable(args, extra_config))
    helper_call_params = {"json_schema_mode": bool(args.json_schema_mode)}
    tactic_extra_config = _build_tactic_extra_config(args, extra_config)
    doc_id = Path(args.src).name

    for idx in range(len(trans_records), len(src_chunks)):
        src = src_chunks[idx]
        ref = ref_chunks[idx] if ref_chunks is not None else None
        sentence_indices = chunk_indices[idx]
        sentence_index = sentence_indices[0] if sentence_indices else idx

        rel_src: list[str] | None = None
        rel_tgt: list[str] | None = None
        if long_memory is not None:
            rel_src, rel_tgt = await long_memory.match(
                src,
                args.top_k,
                helper_runner,
                helper_call_params,
            )
            rel_src = list(rel_src)
            rel_tgt = list(rel_tgt)

        src_summary, tgt_summary = (doc_summary.get_summary() if doc_summary is not None else (None, None))
        hist_info = (
            ent_history.buildin_history(src, args.only_relative)
            if ent_history is not None
            else None
        )
        src_context, tgt_context = (
            trans_context.get_context()
            if trans_context is not None
            else (None, None)
        )

        retrieved_examples = [
            {"source": rel_src_item, "target": rel_tgt_item}
            for rel_src_item, rel_tgt_item in zip(rel_src or [], rel_tgt or [])
        ]
        packet: DocumentMemoryPacket = {
            "doc_id": doc_id,
            "sentence_index": sentence_index,
            "document_src_summary": src_summary or "",
            "document_tgt_summary": tgt_summary or "",
            "historical_terminology": hist_info or "",
            "previous_source_context": list(src_context or []),
            "previous_target_context": list(tgt_context or []),
            "retrieved_examples": retrieved_examples,
            "has_summary": bool(src_summary or tgt_summary),
            "has_history": bool(hist_info),
            "has_context": bool(src_context or tgt_context),
            "has_tm_examples": bool(retrieved_examples),
            "previous_terminology_conflict": bool(trans_records and trans_records[-1].get("conflict")),
        }

        workflow_name = args.workflow if args.workflow != "auto" else choose_workflow(packet, src)
        workflow_config = _build_workflow_config(args, workflow_name)
        state_input = build_thintactic_state(
            src=src,
            src_lang_name=src_lang_name,
            tgt_lang_name=tgt_lang_name,
            packet=packet,
        )
        result = await run_tactic_state(
            state_input,
            workflow_config=workflow_config,
            model_platform=args.model_platform,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            extra_config=tactic_extra_config,
        )
        hyp = result.get("translation") or result.get("best_translation", "")

        record = _build_record(
            idx=idx,
            sentence_index=sentence_index,
            sentence_indices=sentence_indices,
            src=src,
            ref=ref,
            hyp=hyp,
            workflow_name=workflow_name,
            rel_src=rel_src,
            rel_tgt=rel_tgt,
            hist_info=hist_info,
            result=result,
            flow_logging=bool(args.flow_logging),
        )

        if long_memory is not None:
            long_memory.insert(src, hyp)

        if ent_history is not None:
            conflict_list = await ent_history.extract_entity(src, hyp, helper_runner, helper_call_params)
            record["entity_dict"] = ent_history.get_history_dict()
            if conflict_list:
                record["conflict"] = conflict_list

        if trans_context is not None:
            trans_context.update(src, hyp)

        trans_records.append(record)

        if doc_summary is not None and (idx + 1) % args.summary_step == 0:
            current_window = trans_records[-args.summary_step :]
            new_src_summary, new_tgt_summary = await doc_summary.update_summary(
                current_window,
                helper_runner,
                helper_call_params,
            )
            record["new_src_summary"] = new_src_summary
            record["new_tgt_summary"] = new_tgt_summary

        _save_json_array(args.output, trans_records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--language", type=str, required=True)
    parser.add_argument("-s", "--src", type=str, required=True)
    parser.add_argument("-r", "--ref", type=str, default=None)
    parser.add_argument("-o", "--output", type=str, required=True)

    parser.add_argument("--src_summary_prompt", type=str, default=None)
    parser.add_argument("--tgt_summary_prompt", type=str, default=None)
    parser.add_argument("--src_merge_prompt", type=str, default=None)
    parser.add_argument("--tgt_merge_prompt", type=str, default=None)
    parser.add_argument("--retrieve_prompt", type=str, default=None)
    parser.add_argument("--history_prompt", type=str, default=None)

    parser.add_argument("--summary_step", type=int, default=10)
    parser.add_argument("--long_window", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--chunk_size_sentences", type=int, default=1)
    parser.add_argument("-rw", "--recency_weight", type=float, default=0.0)
    parser.add_argument("-sw", "--similarity_weight", type=float, default=10.0)
    parser.add_argument(
        "--only_relative",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=VALID_SETTINGS,
        default=list(VALID_SETTINGS),
    )
    parser.add_argument("--context_window", type=int, default=3)
    parser.add_argument("--retriever", choices=("agent", "none", "embedding"), default="agent")

    parser.add_argument("--workflow", choices=VALID_WORKFLOWS, default="auto")
    parser.add_argument("--draft_mode", choices=VALID_DRAFT_MODES, default=None)
    parser.add_argument("--model_platform", choices=VALID_MODEL_PLATFORMS, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--eval_model_platform", choices=VALID_MODEL_PLATFORMS, default=None)
    parser.add_argument("--eval_model_name", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--eval_temperature", type=float, default=None)
    parser.add_argument("--eval_max_tokens", type=int, default=None)
    parser.add_argument("--quality_threshold", type=int, default=26)
    parser.add_argument("--max_iterations", type=int, default=6)
    parser.add_argument(
        "--send_feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--flow_logging",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--json_schema_mode",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--extra_config_json", type=str, default=None)
    parser.add_argument("--thin_tactic_root", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
