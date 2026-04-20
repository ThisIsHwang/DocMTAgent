from __future__ import annotations

import asyncio
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Awaitable, Callable


ChatTextFn = Callable[..., Awaitable[str | None]]

SEP_MAP = {
    "zh": "",
    "ja": "",
    "en": " ",
    "de": " ",
    "fr": " ",
    "ar": " ",
    "ko": " ",
}

LANG_DICT = {
    "zh": "Chinese",
    "ja": "Japanese",
    "en": "English",
    "de": "German",
    "fr": "French",
    "ar": "Arabic",
    "ko": "Korean",
}


def base_lang(code: str) -> str:
    value = (code or "").strip()
    if not value:
        return ""
    return value.split("_", 1)[0].split("-", 1)[0].lower()


def language_name(code: str) -> str:
    base = base_lang(code)
    return LANG_DICT.get(base, code)


def sep_for_lang(code: str) -> str:
    return SEP_MAP.get(base_lang(code), " ")


def _load_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
    return parsed


def _string_response_schema(field_name: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            field_name: {"type": "string"},
        },
        "required": [field_name],
        "additionalProperties": False,
    }


def _int_list_response_schema(field_name: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            field_name: {
                "type": "array",
                "items": {"type": "integer"},
            }
        },
        "required": [field_name],
        "additionalProperties": False,
    }


async def _invoke_prompt(
    prompt: str,
    call_text: ChatTextFn,
    call_params: dict[str, Any],
) -> str | None:
    response = await call_text(
        [{"role": "user", "content": prompt}],
        **call_params,
    )
    if response is None:
        return None
    return str(response).strip()


async def _invoke_structured_prompt(
    prompt: str,
    call_text: ChatTextFn,
    call_params: dict[str, Any],
    *,
    response_schema: dict[str, Any],
    response_key: str,
    schema_name: str,
) -> Any:
    params = dict(call_params or {})
    params["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": response_schema,
        },
    }
    response = await call_text(
        [{"role": "user", "content": prompt}],
        **params,
    )
    if response is None:
        return None
    parsed = _load_json_object(str(response))
    if response_key not in parsed:
        raise ValueError(f"Missing key '{response_key}' in structured response")
    return parsed[response_key]


class RetrieveAgent:
    def __init__(
        self,
        total: int,
        recency_weight: float,
        similarity_weight: float,
        prompt_template: str,
        skip_context: int,
    ) -> None:
        if recency_weight + similarity_weight != 10.0:
            raise AssertionError("The weights should be added up to 10!")

        self.src_text_list: list[str] = []
        self.tgt_text_list: list[str] = []
        self.total = total
        self.recency_weight = recency_weight / 10.0
        self.similarity_weight = similarity_weight / 10.0
        self.prompt_template = prompt_template
        self.example_number: list[int] | None = None
        self.skip_context = skip_context

    def insert(self, new_src: str, new_tgt: str) -> None:
        if self.total == -1 or len(self.src_text_list) < self.total:
            self.src_text_list.append(new_src)
            self.tgt_text_list.append(new_tgt)
            return

        self.src_text_list = self.src_text_list[1:] + [new_src]
        self.tgt_text_list = self.tgt_text_list[1:] + [new_tgt]

    async def match(
        self,
        query: str,
        num: int,
        call_text: ChatTextFn,
        call_params: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        if len(self.src_text_list) <= num:
            return (self.src_text_list, self.tgt_text_list)

        sentence_list = "\n".join(
            f"<Sentence {idx + 1}> {src}"
            for idx, src in enumerate(self.src_text_list)
        )

        if self.example_number is None or len(self.example_number) != num:
            random.seed(0)
            self.example_number = random.sample(list(range(max(10, num))), num)
            self.example_number.sort()
        example_numbers = [str(i) for i in self.example_number]
        if num > 1:
            example_number = ", ".join(example_numbers[:-1]) + " and " + example_numbers[-1]
        else:
            example_number = example_numbers[0]
        example_list = str(self.example_number)

        prompt = self.prompt_template.format(
            top_num=num,
            sentence_list=sentence_list,
            example_number=example_number,
            example_list=example_list,
            query=query,
        )

        if bool(call_params.get("json_schema_mode")):
            chosen = await _invoke_structured_prompt(
                prompt,
                call_text,
                call_params,
                response_schema=_int_list_response_schema("ids"),
                response_key="ids",
                schema_name="retrieved_ids",
            )
            if chosen is None:
                return ([], [])
        else:
            chosen_ids = await _invoke_prompt(prompt, call_text, call_params)
            if chosen_ids is None:
                return ([], [])
            try:
                chosen = eval(chosen_ids)
            except Exception:
                chosen = []

        chosen = [
            item
            for item in chosen
            if type(item) is int and 1 <= item <= len(self.src_text_list)
        ]
        chosen.sort()
        return (
            [self.src_text_list[item - 1] for item in chosen],
            [self.tgt_text_list[item - 1] for item in chosen],
        )


class Summary:
    def __init__(
        self,
        src_gen_template: str,
        tgt_gen_template: str,
        src_merge_template: str,
        tgt_merge_template: str,
        src_lang: str,
        tgt_lang: str,
    ) -> None:
        self.src_summary: str | None = None
        self.tgt_summary: str | None = None
        self.src_gen_template = src_gen_template
        self.tgt_gen_template = tgt_gen_template
        self.src_merge_template = src_merge_template
        self.tgt_merge_template = tgt_merge_template
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def set_summary(self, src_summary: str | None, tgt_summary: str | None) -> None:
        self.src_summary = src_summary
        self.tgt_summary = tgt_summary

    async def gen_summary(
        self,
        record_list: list[dict[str, Any]],
        call_text: ChatTextFn,
        call_params: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        src_para = sep_for_lang(self.src_lang).join(str(item["src"]) for item in record_list)
        tgt_para = sep_for_lang(self.tgt_lang).join(str(item["hyp"]) for item in record_list)

        src_prompt = self.src_gen_template.format(src_para=src_para)
        tgt_prompt = self.tgt_gen_template.format(src_para=tgt_para)

        if bool(call_params.get("json_schema_mode")):
            src_summary, tgt_summary = await asyncio.gather(
                _invoke_structured_prompt(
                    src_prompt,
                    call_text,
                    call_params,
                    response_schema=_string_response_schema("summary"),
                    response_key="summary",
                    schema_name="source_summary",
                ),
                _invoke_structured_prompt(
                    tgt_prompt,
                    call_text,
                    call_params,
                    response_schema=_string_response_schema("summary"),
                    response_key="summary",
                    schema_name="target_summary",
                ),
            )
            src_value = None if src_summary is None else str(src_summary).strip()
            tgt_value = None if tgt_summary is None else str(tgt_summary).strip()
            return (src_value, tgt_value)

        return await asyncio.gather(
            _invoke_prompt(src_prompt, call_text, call_params),
            _invoke_prompt(tgt_prompt, call_text, call_params),
        )

    async def merge_summary(
        self,
        src_new_summary: str | None,
        tgt_new_summary: str | None,
        call_text: ChatTextFn,
        call_params: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        if self.src_summary is None:
            return (src_new_summary, tgt_new_summary)

        src_prompt = self.src_merge_template.format(
            summary_1=self.src_summary,
            summary_2=src_new_summary,
        )
        tgt_prompt = self.tgt_merge_template.format(
            summary_1=self.tgt_summary,
            summary_2=tgt_new_summary,
        )

        if bool(call_params.get("json_schema_mode")):
            src_summary, tgt_summary = await asyncio.gather(
                _invoke_structured_prompt(
                    src_prompt,
                    call_text,
                    call_params,
                    response_schema=_string_response_schema("summary"),
                    response_key="summary",
                    schema_name="merged_source_summary",
                ),
                _invoke_structured_prompt(
                    tgt_prompt,
                    call_text,
                    call_params,
                    response_schema=_string_response_schema("summary"),
                    response_key="summary",
                    schema_name="merged_target_summary",
                ),
            )
            src_value = None if src_summary is None else str(src_summary).strip()
            tgt_value = None if tgt_summary is None else str(tgt_summary).strip()
            return (src_value, tgt_value)

        return await asyncio.gather(
            _invoke_prompt(src_prompt, call_text, call_params),
            _invoke_prompt(tgt_prompt, call_text, call_params),
        )

    async def update_summary(
        self,
        record_list: list[dict[str, Any]],
        call_text: ChatTextFn,
        call_params: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        src_new_summary, tgt_new_summary = await self.gen_summary(
            record_list,
            call_text,
            call_params,
        )
        self.src_summary, self.tgt_summary = await self.merge_summary(
            src_new_summary,
            tgt_new_summary,
            call_text,
            call_params,
        )
        return (self.src_summary, self.tgt_summary)

    def get_summary(self) -> tuple[str | None, str | None]:
        return (self.src_summary, self.tgt_summary)


class History:
    def __init__(self, prompt_template: str, src_lang: str, tgt_lang: str) -> None:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.entity_dict: dict[str, str] = {}
        self.prompt_template = prompt_template

    async def extract_entity(
        self,
        src: str,
        tgt: str,
        call_text: ChatTextFn,
        call_params: dict[str, Any],
    ) -> list[str]:
        prompt = self.prompt_template.format(
            src_lang=language_name(self.src_lang),
            tgt_lang=language_name(self.tgt_lang),
            src=src,
            tgt=tgt,
        )
        if bool(call_params.get("json_schema_mode")):
            new_info = await _invoke_structured_prompt(
                prompt,
                call_text,
                call_params,
                response_schema=_string_response_schema("history"),
                response_key="history",
                schema_name="proper_noun_history",
            )
            new_info = None if new_info is None else str(new_info).strip()
        else:
            new_info = await _invoke_prompt(prompt, call_text, call_params)

        conflicts: list[str] = []
        if new_info is not None and new_info not in ["N/A", "None", "", "??"]:
            for ent_pair in new_info.split(", "):
                if len(ent_pair.split(" - ")) != 2:
                    continue

                src_ent, tgt_ent = ent_pair.split(" - ")
                src_ent = src_ent.replace('"', "").replace("'", "")
                tgt_ent = tgt_ent.replace('"', "").replace("'", "")
                if self.entity_dict.get(src_ent, "") == "":
                    self.entity_dict[src_ent] = tgt_ent if tgt_ent != "N/A" else src_ent
                elif self.entity_dict[src_ent] != tgt_ent:
                    conflicts.append(
                        f'"{src_ent}" - "{self.entity_dict[src_ent]}"/"{tgt_ent}"'
                    )
        return conflicts

    def buildin_history(self, sentence: str, only_relative: bool) -> str:
        if only_relative:
            entity_list = [ent for ent in self.entity_dict if ent in sentence]
        else:
            entity_list = list(self.entity_dict)
        return ", ".join(
            f'"{ent}" - "{self.entity_dict[ent]}"'
            for ent in entity_list
            if ent in self.entity_dict
        )

    def get_history_dict(self) -> dict[str, str]:
        return deepcopy(self.entity_dict)

    def set_history_dict(self, history_dict: dict[str, str]) -> None:
        self.entity_dict = history_dict


class Context:
    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.src_context: list[str] = []
        self.tgt_context: list[str] = []

    def update(self, src: str, tgt: str) -> None:
        if self.window_size == -1:
            self.src_context.append(src)
            self.tgt_context.append(tgt)
            return

        self.src_context = self.src_context[-(self.window_size - 1) :] + [src]
        self.tgt_context = self.tgt_context[-(self.window_size - 1) :] + [tgt]

    def get_context(self) -> tuple[list[str], list[str]]:
        return (self.src_context, self.tgt_context)


def init_memory(
    args: Any,
    prompt_bundle: dict[str, str],
    src_lang: str,
    tgt_lang: str,
) -> tuple[Context | None, RetrieveAgent | None, Summary | None, History | None]:
    settings = set(args.settings or [])

    if getattr(args, "retriever", "agent") == "embedding":
        raise ValueError("Hybrid mode does not support retriever=embedding.")

    trans_context = None
    if "context" in settings:
        trans_context = Context(args.context_window)

    long_memory = None
    if "long" in settings and getattr(args, "retriever", "agent") == "agent":
        long_memory = RetrieveAgent(
            args.long_window,
            args.recency_weight,
            args.similarity_weight,
            prompt_bundle["retrieve_prompt"],
            args.context_window,
        )

    doc_summary = None
    if "summary" in settings:
        doc_summary = Summary(
            prompt_bundle["src_summary_prompt"],
            prompt_bundle["tgt_summary_prompt"],
            prompt_bundle["src_merge_prompt"],
            prompt_bundle["tgt_merge_prompt"],
            src_lang,
            tgt_lang,
        )

    ent_history = None
    if "history" in settings:
        ent_history = History(prompt_bundle["history_prompt"], src_lang, tgt_lang)

    return (trans_context, long_memory, doc_summary, ent_history)


def init_resume_state(
    args: Any,
    prompt_bundle: dict[str, str],
    src_lang: str,
    tgt_lang: str,
) -> tuple[Context | None, RetrieveAgent | None, Summary | None, History | None, list[dict[str, Any]]]:
    trans_context, long_memory, doc_summary, ent_history = init_memory(
        args,
        prompt_bundle,
        src_lang,
        tgt_lang,
    )

    trans_records: list[dict[str, Any]] = []
    output_path = Path(args.output)
    if output_path.is_file():
        trans_records = json.loads(output_path.read_text(encoding="utf-8"))

    for record in trans_records:
        src = str(record.get("src", ""))
        hyp = str(record.get("hyp", ""))

        if doc_summary is not None and record.get("new_src_summary") is not None:
            doc_summary.set_summary(
                record.get("new_src_summary"),
                record.get("new_tgt_summary"),
            )

        if long_memory is not None:
            long_memory.insert(src, hyp)

        if ent_history is not None and isinstance(record.get("entity_dict"), dict):
            ent_history.set_history_dict(record["entity_dict"])

        if trans_context is not None:
            trans_context.update(src, hyp)

    return (trans_context, long_memory, doc_summary, ent_history, trans_records)
