from __future__ import annotations

import argparse
import asyncio
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpx

from justatom.agentic.benchmark import SearchBudget
from justatom.agentic.benchmark_data import load_cases
from justatom.agentic.inspect_harness import build_task
from justatom.agentic.openai_compatible import OpenAICompatibleChatBackend
from justatom.agentic.schemas import AgentObjective
from justatom.etc.schema import Document


class HttpSearchRetriever:
    """Read-only adapter for JustAtom's HTTP ``/searching`` endpoint."""

    def __init__(self, search_url: str, *, client: httpx.AsyncClient | None = None) -> None:
        _validate_http_url(search_url, "search_url")
        self.search_url = search_url
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    async def retrieve(self, query: str, *, top_k: int = 5, **kwargs: Any) -> list[Document]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must be non-empty")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        response = await self._client.post(
            self.search_url,
            json={"text": query, "top_k": top_k},
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as error:
            raise ValueError("invalid search response: expected a JSON object") from error
        validated = _validate_search_response(payload)
        return [Document(id=document_id, content=content, score=score) for document_id, content, score in validated]

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _validate_search_response(payload: Any) -> list[tuple[str, str, float | None]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("docs"), list):
        raise ValueError("invalid search response: docs must be a list")
    validated: list[tuple[str, str, float | None]] = []
    for index, row in enumerate(payload["docs"]):
        if not isinstance(row, dict):
            raise ValueError(f"invalid search response: docs[{index}] must be an object")
        document_id = row.get("id")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"invalid search response: docs[{index}].id must be a non-empty string")
        content = row.get("content")
        if not isinstance(content, str):
            raise ValueError(f"invalid search response: docs[{index}].content must be a string")
        if "score" not in row:
            raise ValueError(f"invalid search response: docs[{index}].score is required")
        score = row.get("score")
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)):
            raise ValueError(f"invalid search response: docs[{index}].score must be a finite number or null")
        validated.append((document_id, content, None if score is None else float(score)))
    return validated


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonempty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must be a non-empty string")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the read-only JustAtom Inspect search benchmark.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--search-url", required=True, type=_nonempty)
    parser.add_argument("--model", required=True, type=_nonempty)
    parser.add_argument("--base-url", required=True, type=_nonempty)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", type=_nonempty)
    parser.add_argument("--model-provider", choices=("auto", "openai-compatible", "openrouter"), default="auto")
    parser.add_argument("--method", choices=("react", "native", "both"), default="both")
    parser.add_argument("--top-k", type=_positive_integer, default=5)
    parser.add_argument("--max-searches", type=_positive_integer, default=4)
    parser.add_argument("--max-context-documents", type=_positive_integer, default=20)
    parser.add_argument("--max-document-chars", type=_positive_integer, default=2_000)
    parser.add_argument("--max-context-chars", type=_positive_integer, default=24_000)
    parser.add_argument("--max-query-chars", type=_positive_integer, default=4_000)
    parser.add_argument("--max-model-calls", type=_positive_integer, default=6)
    parser.add_argument("--max-output-tokens", type=_positive_integer, default=2_048)
    parser.add_argument("--token-limit", type=_positive_integer, default=8_192)
    parser.add_argument("--time-limit", type=_positive_integer, default=60)
    parser.add_argument("--limit", type=_positive_integer)
    parser.add_argument("--concurrency", type=_positive_integer, default=1)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--corpus-revision", required=True, type=_nonempty)
    parser.add_argument("--retrieval-mode", default="keyword", type=_nonempty)
    return parser


def _validate_http_url(value: str, name: str) -> SplitResult:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty URL")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{name} must be a full http:// or https:// URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{name} must not contain embedded credentials")
    if parsed.query:
        raise ValueError(f"{name} must not contain a query")
    if parsed.fragment:
        raise ValueError(f"{name} must not contain a fragment")
    try:
        parsed.port
    except ValueError as error:
        raise ValueError(f"{name} has an invalid port") from error
    return parsed


def _sanitized_url(value: str) -> str:
    parsed = _validate_http_url(value, "URL")
    hostname = parsed.hostname or ""
    try:
        if ipaddress.ip_address(hostname).version == 6:
            hostname = f"[{hostname}]"
    except ValueError:
        pass
    netloc = hostname if parsed.port is None else f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _is_local_url(value: str) -> bool:
    hostname = (_validate_http_url(value, "base_url").hostname or "").lower().rstrip(".")
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _api_key(environment_name: str, base_url: str) -> str:
    if not isinstance(environment_name, str) or not environment_name.strip():
        raise ValueError("api_key_env must be a non-empty string")
    value = os.environ.get(environment_name)
    if value and value.strip():
        return value
    if _is_local_url(base_url):
        return "unused"
    raise ValueError(f"missing API key in environment variable {environment_name}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_methods(method: str) -> list[str]:
    if method == "both":
        return ["react", "native"]
    if method in {"react", "native"}:
        return [method]
    raise ValueError("method must be react, native, or both")


async def run(args: argparse.Namespace, *, client: httpx.AsyncClient | None = None) -> dict[str, Any]:
    """Validate locally, execute the requested Inspect tasks, and export results."""

    budget = SearchBudget(
        top_k=args.top_k,
        max_searches=args.max_searches,
        max_context_documents=args.max_context_documents,
        max_document_chars=args.max_document_chars,
        max_context_chars=args.max_context_chars,
        max_query_chars=args.max_query_chars,
    )
    for name in ("max_model_calls", "max_output_tokens", "token_limit", "time_limit", "concurrency"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    methods = _selected_methods(args.method)
    _validate_http_url(args.search_url, "search_url")
    model_url = _validate_http_url(args.base_url, "base_url")
    if args.model_provider not in {"auto", "openai-compatible", "openrouter"}:
        raise ValueError("model_provider must be auto, openai-compatible, or openrouter")
    model_provider = args.model_provider
    if model_provider == "auto":
        model_provider = "openrouter" if (model_url.hostname or "").lower().rstrip(".") == "openrouter.ai" else "openai-compatible"
    if not isinstance(args.corpus_revision, str) or not args.corpus_revision.strip():
        raise ValueError("corpus_revision must be a non-empty string")
    if not isinstance(args.retrieval_mode, str) or not args.retrieval_mode.strip():
        raise ValueError("retrieval_mode must be a non-empty string")
    if not isinstance(args.model, str) or not args.model.strip():
        raise ValueError("model must be a non-empty string")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")

    dataset = Path(args.dataset)
    cases = load_cases(dataset, limit=args.limit)
    dataset_sha256 = _file_sha256(dataset)
    api_key = _api_key(args.api_key_env, args.base_url)
    provenance = {
        "corpus_revision": args.corpus_revision,
        "dataset_sha256": dataset_sha256,
        "model": args.model,
        "model_endpoint": _sanitized_url(args.base_url),
        "model_provider": model_provider,
        "retrieval_backend": "justatom-http-search",
        "retrieval_endpoint": _sanitized_url(args.search_url),
        "retrieval_mode": args.retrieval_mode,
    }

    if client is None:
        async with httpx.AsyncClient(timeout=args.time_limit) as owned_client:
            return await _execute(args, cases, methods, budget, provenance, api_key, output, owned_client)
    return await _execute(args, cases, methods, budget, provenance, api_key, output, client)


async def _execute(args, cases, methods, budget, provenance, api_key, output, client) -> dict[str, Any]:
    try:
        from inspect_ai import eval_async
        from inspect_ai.model import get_model
    except ModuleNotFoundError as error:
        if str(error.name or "").startswith("inspect_ai"):
            raise ImportError('Install the optional harness with pip install "justatom[inspect]"') from error
        raise
    from justatom.agentic.inspect_results import export_results

    retriever = HttpSearchRetriever(args.search_url, client=client)
    react_model = None
    if "react" in methods:
        model_prefix = "openrouter" if provenance["model_provider"] == "openrouter" else "openai-api/justatom"
        react_model = get_model(
            f"{model_prefix}/{args.model}",
            base_url=args.base_url,
            api_key=api_key,
            max_retries=0,
            responses_api=False,
            strict_tools=False,
        )

    tasks: list[tuple[str, object]] = []
    for method in methods:
        planner_factory = None
        if method == "native":
            planner_factory = lambda: OpenAICompatibleChatBackend(
                args.base_url,
                args.model,
                api_key=api_key,
                timeout_seconds=args.time_limit,
                temperature=0,
                max_tokens=args.max_output_tokens,
                objective=AgentObjective.CONTEXT,
            )
        task = build_task(
            cases,
            retriever,
            budget=budget,
            method=method,
            planner_factory=planner_factory,
            max_model_calls=args.max_model_calls,
            max_output_tokens=args.max_output_tokens,
            token_limit=args.token_limit,
            time_limit=args.time_limit,
            provenance=provenance,
        )
        tasks.append((method, task))

    output.mkdir(parents=True, exist_ok=False)
    logs: list[Any] = []
    evaluation_error: BaseException | None = None
    try:
        for method, task in tasks:
            model = react_model if method == "react" else "mockllm/unused"
            logs.extend(
                await eval_async(
                    task,
                    model=model,
                    max_samples=args.concurrency,
                    log_dir=str(output / "logs"),
                    # Provider usage.cost is not mapped by every Inspect adapter.
                    # Retain model response payloads for complete cost accounting.
                    log_model_api=True,
                )
            )
    except BaseException as error:
        evaluation_error = error
        raise
    finally:
        try:
            summary = export_results(logs, output, cases=cases, methods=methods)
        except BaseException:
            if evaluation_error is None:
                raise
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = asyncio.run(run(args))
    except (FileExistsError, ImportError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["HttpSearchRetriever", "build_parser", "main", "run"]
