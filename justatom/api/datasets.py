from __future__ import annotations

import argparse
import asyncio as asio
import json
import os
import re
import sys
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import Any

from loguru import logger

from justatom.configuring.builtins import resolve_builtin_path
from justatom.configuring.scenarios import deep_merge, load_scenario_config, parse_unknown_overrides
from justatom.running.llm import OpenAIAsyncWrapper, OpenAiTask
from justatom.storing.dataset import API as DatasetApi


_TEMPLATE_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_.-]+)\s*\}\}")


def _normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("${") and text.endswith("}"):
        return None
    return text


def _load_prompt_text(path_value: Any, field_name: str) -> str:
    path_raw = _normalize_optional_str(path_value)
    if path_raw is None:
        raise ValueError(f"prompt.{field_name} must be set")

    path = resolve_builtin_path(path_raw)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    return text


def _get_nested(mapping: dict[str, Any], key: str) -> Any:
    cursor: Any = mapping
    for part in key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def render_template(template: str, values: dict[str, Any]) -> str:
    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        value = _get_nested(values, token)
        return "" if value is None else str(value)

    return _TEMPLATE_RE.sub(_replace, template)


def _clip_text(text: str, limit: int = 6000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]


def _normalize_queries_payload(payload: Any) -> list[str]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        for key in ("queries", "rephrases", "questions", "items"):
            if key in payload:
                return _normalize_queries_payload(payload.get(key))
        return []
    if isinstance(payload, str):
        stripped = payload.strip()
        return [stripped] if stripped else []
    if isinstance(payload, Iterable):
        output: list[str] = []
        for item in payload:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                output.append(text)
        return output
    return []


def _rows_from_source(
    dataset_name_or_path: str,
    *,
    split: str | None,
    limit: int | None,
) -> Iterable[dict[str, Any]]:
    try:
        source = DatasetApi.named(str(dataset_name_or_path)).iterator(
            lazy=True,
            split=split,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load dataset source. "
            "If this is a HuggingFace dataset and you're on Python 3.14, "
            "it may be an upstream `datasets` compatibility issue for dataset card/features parsing. "
            "Try Python 3.11/3.12 environment or pin a newer compatible `datasets` package. "
            f"dataset_name_or_path={dataset_name_or_path!r}, split={split!r}"
        ) from exc
    if limit is None:
        return source
    return islice(source, int(limit))


class OpenAIQuestionGenerator(OpenAIAsyncWrapper):
    async def _call(self, task: OpenAiTask) -> dict[str, Any]:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": task.text},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.response_format is not None:
            kwargs["response_format"] = self.response_format

        raw = await self.client.chat.completions.create(**kwargs)
        text = ((raw.choices or [None])[0].message.content or "") if raw else ""
        parsed_json = None
        if text:
            try:
                parsed_json = json.loads(text)
            except Exception:
                parsed_json = None

        return {
            "id": task.id,
            "metadata": task.metadata,
            "model": self.model,
            "error": None,
            "text": text,
            "parsed_json": parsed_json,
            "raw": raw.model_dump() if raw is not None else None,
        }


def _legacy_cli_overlay(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if args.dataset_name_or_path is not None:
        cfg.setdefault("dataset", {})["name_or_path"] = args.dataset_name_or_path
    if args.dataset_split is not None:
        cfg.setdefault("dataset", {})["split"] = args.dataset_split
    if args.dataset_limit is not None:
        cfg.setdefault("dataset", {})["limit"] = args.dataset_limit
    if args.content_col is not None:
        cfg.setdefault("dataset", {})["content_field"] = args.content_col

    if args.target_count is not None:
        cfg.setdefault("generation", {})["target_count"] = args.target_count
    if args.output_language is not None:
        cfg.setdefault("generation", {})["output_language"] = args.output_language
    if args.language is not None:
        cfg.setdefault("generation", {})["language"] = args.language
    if args.style is not None:
        cfg.setdefault("generation", {})["style"] = args.style
    if args.min_words is not None:
        cfg.setdefault("generation", {})["min_words"] = args.min_words
    if args.max_words is not None:
        cfg.setdefault("generation", {})["max_words"] = args.max_words
    if args.extra_instructions is not None:
        cfg.setdefault("generation", {})["extra_instructions"] = args.extra_instructions

    if args.model is not None:
        cfg.setdefault("llm", {})["model"] = args.model
    if args.temperature is not None:
        cfg.setdefault("llm", {})["temperature"] = args.temperature
    if args.top_p is not None:
        cfg.setdefault("llm", {})["top_p"] = args.top_p
    if args.max_tokens is not None:
        cfg.setdefault("llm", {})["max_tokens"] = args.max_tokens
    if args.max_parallel_requests is not None:
        cfg.setdefault("llm", {})["max_parallel_requests"] = args.max_parallel_requests

    if args.save_path is not None:
        cfg.setdefault("output", {})["save_path"] = args.save_path
    if args.print_sample is not None:
        cfg.setdefault("runtime", {})["print_sample"] = args.print_sample
    if args.dry_run:
        cfg.setdefault("runtime", {})["dry_run"] = True

    return cfg


def load_datasets_config(
    *,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return load_scenario_config(
        "datasets",
        config=config,
        config_path=config_path,
        overrides=overrides,
    )


def _cfg_to_main_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset = cfg.get("dataset") or {}
    generation = cfg.get("generation") or {}
    llm = cfg.get("llm") or {}
    output = cfg.get("output") or {}
    prompt = cfg.get("prompt") or {}
    runtime = cfg.get("runtime") or {}
    system_prompt = _load_prompt_text(prompt.get("system_path"), "system_path")
    user_template = _load_prompt_text(prompt.get("user_template_path"), "user_template_path")

    return {
        "dataset_name_or_path": dataset.get("name_or_path"),
        "content_field": dataset.get("content_field", "content"),
        "split": dataset.get("split"),
        "limit": dataset.get("limit"),
        "target_count": int(generation.get("target_count", 3)),
        "output_language": generation.get("output_language", generation.get("language", "ru")),
        "style": generation.get("style", "diverse"),
        "min_words": int(generation.get("min_words", 4)),
        "max_words": int(generation.get("max_words", 14)),
        "extra_instructions": generation.get("extra_instructions", ""),
        "model": llm.get("model", "gpt-4o-mini"),
        "temperature": float(llm.get("temperature", 0.6)),
        "top_p": float(llm.get("top_p", 0.95)),
        "max_tokens": int(llm.get("max_tokens", 512)),
        "api_call_timeout": float(llm.get("api_call_timeout", 120.0)),
        "max_parallel_requests": int(llm.get("max_parallel_requests", llm.get("max_concurrent", 8))),
        "base_url": _normalize_optional_str(llm.get("base_url")),
        "openai_token": _normalize_optional_str(llm.get("openai_token")),
        "save_path": output.get("save_path"),
        "snapshot_dir": output.get("snapshot_dir"),
        "snapshot_prefix": output.get("snapshot_prefix", "datasets-gen-"),
        "snapshot_backend": output.get("snapshot_backend", "loguru"),
        "system_prompt": system_prompt,
        "user_template": user_template,
        "prompt_vars": prompt.get("variables", {}) or {},
        "dry_run": bool(runtime.get("dry_run", False)),
        "print_sample": int(runtime.get("print_sample", 3)),
    }


def resolve_datasets_kwargs(
    *,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = load_datasets_config(
        config=config,
        config_path=config_path,
        overrides=overrides,
    )
    return _cfg_to_main_kwargs(cfg)


def _parse_args(argv: list[str] | None = None) -> dict[str, Any]:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="justatom|datasets",
        description="Generate questions/rephrases from dataset content using OpenAI-compatible LLM",
    )
    parser.add_argument("--config")
    parser.add_argument("--dataset-name-or-path")
    parser.add_argument("--dataset-split")
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--content-col")

    parser.add_argument("--target-count", type=int)
    parser.add_argument("--output-language")
    parser.add_argument("--language")
    parser.add_argument("--style")
    parser.add_argument("--min-words", type=int)
    parser.add_argument("--max-words", type=int)
    parser.add_argument("--extra-instructions")

    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-parallel-requests", type=int)

    parser.add_argument("--save-path")
    parser.add_argument("--print-sample", type=int)
    parser.add_argument("--dry-run", action="store_true")

    args, unknown = parser.parse_known_args(argv)
    overrides = _legacy_cli_overlay(args)
    dotted_overrides = parse_unknown_overrides(unknown)
    if dotted_overrides:
        overrides = deep_merge(overrides, dotted_overrides)

    return resolve_datasets_kwargs(config_path=args.config, overrides=overrides)


async def main(
    dataset_name_or_path: str | None = None,
    content_field: str = "content",
    split: str | None = None,
    limit: int | None = None,
    target_count: int = 3,
    output_language: str = "ru",
    style: str = "diverse",
    min_words: int = 4,
    max_words: int = 14,
    extra_instructions: str = "",
    model: str = "gpt-4o-mini",
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 512,
    api_call_timeout: float = 120.0,
    max_parallel_requests: int = 8,
    base_url: str | None = None,
    openai_token: str | None = None,
    save_path: str | Path | None = None,
    snapshot_dir: str | Path | None = None,
    snapshot_prefix: str = "datasets-gen-",
    snapshot_backend: str = "loguru",
    system_prompt: str = "",
    user_template: str = "",
    prompt_vars: dict[str, Any] | None = None,
    dry_run: bool = False,
    print_sample: int = 3,
) -> list[dict[str, Any]]:
    if dataset_name_or_path is None:
        raise ValueError("dataset.name_or_path (or --dataset-name-or-path) must be set")

    if save_path is None:
        save_path = Path.cwd() / "generated" / "dataset_queries.jsonl"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _rows_from_source(
        str(dataset_name_or_path),
        split=split,
        limit=limit,
    )

    tasks: list[OpenAiTask] = []
    base_prompt_vars = prompt_vars or {}
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        if content_field not in row or row.get(content_field) is None:
            continue

        content = _clip_text(str(row.get(content_field, "")))
        if not content.strip():
            continue

        task_vars = {
            **base_prompt_vars,
            "content": content,
            "target_count": target_count,
            "output_language": output_language,
            "language": output_language,
            "style": style,
            "min_words": min_words,
            "max_words": max_words,
            "extra_instructions": extra_instructions,
        }
        prompt = render_template(user_template, task_vars)
        task = OpenAiTask(
            id=str(row.get("id", idx)),
            text=prompt,
            metadata={
                "source_index": idx,
                "dataset": str(dataset_name_or_path),
                "content_field": content_field,
                "content": content,
            },
        )
        tasks.append(task)

    if print_sample > 0:
        logger.info(
            "Prepared {} tasks. Sample ids: {}",
            len(tasks),
            [t.id for t in tasks[:print_sample]],
        )

    if dry_run:
        return [
            {
                "id": t.id,
                "metadata": t.metadata,
                "prompt": t.text,
            }
            for t in tasks[: max(1, print_sample)]
        ]

    gen = OpenAIQuestionGenerator(
        model=model,
        system_prompt=system_prompt,
        openai_token=openai_token,
        base_url=base_url,
        response_format={"type": "json_object"},
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        api_call_timeout=api_call_timeout,
        max_concurrent=max_parallel_requests,
        progress_desc="Dataset generation",
        snapshot_dir=snapshot_dir,
        snapshot_prefix=snapshot_prefix,
        snapshot_backend=snapshot_backend,
    )
    responses = await gen.run_async(tasks)

    records: list[dict[str, Any]] = []
    with save_path.open("w", encoding="utf-8") as fp:
        for res in responses:
            generated = _normalize_queries_payload(res.get("parsed_json"))
            record = {
                "id": res.get("id"),
                "dataset": str(dataset_name_or_path),
                "model": res.get("model", model),
                "generated": generated,
                "source": (res.get("metadata") or {}).get("content"),
                "error": res.get("error"),
            }
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

    logger.info("Saved {} generated rows to {}", len(records), save_path)
    return records


def _run_cli() -> None:
    kwargs = _parse_args()
    result = asio.run(main(**kwargs))
    if kwargs.get("dry_run"):
        if isinstance(result, list) and result:
            logger.info("Dry-run sample prompts: {}", json.dumps(result[:1], ensure_ascii=False))
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    _run_cli()
