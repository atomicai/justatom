from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import httpx
import pytest

from justatom.agentic import inspect_cli
from justatom.agentic.inspect_cli import HttpSearchRetriever, build_parser, run


def test_http_search_retriever_posts_exact_payload_and_preserves_service_ids():
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "docs": [
                    {
                        "id": " original-id ",
                        "content": "Passage text",
                        "score": 0.75,
                        "answer": "must not be forwarded",
                        "chunk_ids": ["gold-secret"],
                        "meta": {"secret": "must not be forwarded"},
                    }
                ]
            },
        )

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            retriever = HttpSearchRetriever(
                "https://search.test/searching",
                client=client,
            )
            documents = await retriever.retrieve("  exact query  ", top_k=3)

            assert len(requests) == 1
            assert requests[0].method == "POST"
            assert requests[0].url == httpx.URL("https://search.test/searching")
            assert json.loads(requests[0].content) == {
                "text": "  exact query  ",
                "top_k": 3,
            }
            assert len(documents) == 1
            assert documents[0].id == " original-id "
            assert documents[0].content == "Passage text"
            assert documents[0].score == 0.75
            assert documents[0].meta == {}

    asyncio.run(exercise())


def test_http_search_retriever_accepts_integer_json_score_as_numeric():
    async def exercise() -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"docs": [{"id": "id", "content": "text", "score": 1}]},
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            documents = await HttpSearchRetriever("https://search.test/searching", client=client).retrieve("query")
            assert documents[0].score == 1.0
            assert isinstance(documents[0].score, float)

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "docs"),
        ({"docs": {}}, "docs"),
        ({"docs": ["not-an-object"]}, r"docs\[0\]"),
        ({"docs": [{"content": "text", "score": 0.5}]}, r"docs\[0\]\.id"),
        ({"docs": [{"id": " ", "content": "text", "score": 0.5}]}, r"docs\[0\]\.id"),
        ({"docs": [{"id": "id", "content": 7, "score": 0.5}]}, r"docs\[0\]\.content"),
        ({"docs": [{"id": "id", "content": "text"}]}, r"docs\[0\]\.score"),
        ({"docs": [{"id": "id", "content": "text", "score": "0.5"}]}, r"docs\[0\]\.score"),
        ({"docs": [{"id": "id", "content": "text", "score": True}]}, r"docs\[0\]\.score"),
    ],
)
def test_http_search_retriever_rejects_malformed_response_before_returning_documents(payload, message):
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            retriever = HttpSearchRetriever("https://search.test/searching", client=client)
            with pytest.raises(ValueError, match=message):
                await retriever.retrieve("query")

    asyncio.run(exercise())


@pytest.mark.parametrize(("query", "top_k"), [("", 5), (" ", 5), (7, 5), ("query", 0), ("query", True)])
def test_http_search_retriever_rejects_invalid_request_before_http(query, top_k):
    called = False

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"docs": []})

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            retriever = HttpSearchRetriever("https://search.test/searching", client=client)
            with pytest.raises((TypeError, ValueError)):
                await retriever.retrieve(query, top_k=top_k)

    asyncio.run(exercise())
    assert called is False


def _required_cli_arguments(tmp_path: Path) -> list[str]:
    dataset = tmp_path / "benchmark.jsonl"
    dataset.write_text(
        json.dumps({"query_id": "q1", "query": "Question", "chunk_ids": ["c1"]}) + "\n",
        encoding="utf-8",
    )
    return [
        "--dataset",
        str(dataset),
        "--search-url",
        "http://127.0.0.1:8000/searching",
        "--model",
        "model-id",
        "--base-url",
        "http://127.0.0.1:9000/v1",
        "--output",
        str(tmp_path / "output"),
        "--corpus-revision",
        "corpus-v1",
    ]


def test_parser_exposes_documented_defaults(tmp_path):
    args = build_parser().parse_args(_required_cli_arguments(tmp_path))

    assert vars(args) == {
        "dataset": Path(tmp_path / "benchmark.jsonl"),
        "search_url": "http://127.0.0.1:8000/searching",
        "model": "model-id",
        "base_url": "http://127.0.0.1:9000/v1",
        "api_key_env": "OPENAI_API_KEY",
        "method": "both",
        "top_k": 5,
        "max_searches": 4,
        "max_context_documents": 20,
        "max_document_chars": 2000,
        "max_context_chars": 24000,
        "max_query_chars": 4000,
        "max_model_calls": 6,
        "token_limit": 8192,
        "time_limit": 60,
        "limit": None,
        "concurrency": 1,
        "output": Path(tmp_path / "output"),
        "corpus_revision": "corpus-v1",
        "retrieval_mode": "keyword",
    }


def test_module_help_lists_required_service_arguments():
    completed = subprocess.run(
        [sys.executable, "-m", "justatom.agentic.inspect_cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--dataset" in completed.stdout
    assert "--search-url" in completed.stdout
    assert "--corpus-revision" in completed.stdout


def test_real_inspect_openai_compatible_provider_works_with_pinned_sdk():
    pytest.importorskip("inspect_ai")
    pytest.importorskip("openai")
    from inspect_ai.model import GenerateConfig, get_model

    requests: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url == httpx.URL("http://model.test/v1/chat/completions")
        if payload.get("stream"):
            chunks = [
                {
                    "id": "completion-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "model-id",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "DONE"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "completion-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "model-id",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                },
            ]
            body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                "id": "completion-1",
                "object": "chat.completion",
                "created": 1,
                "model": "model-id",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "DONE"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            model = get_model(
                "openai-api/justatom/model-id",
                base_url="http://model.test/v1",
                api_key="unused",
                responses_api=False,
                strict_tools=False,
                http_client=client,
                memoize=False,
            )
            output = await model.generate(
                "Return DONE",
                config=GenerateConfig(temperature=0, max_tokens=32, max_retries=0),
            )
            assert output.completion == "DONE"

    asyncio.run(exercise())
    assert len(requests) == 1
    assert requests[0]["model"] == "model-id"
    assert requests[0]["temperature"] == 0.0
    assert requests[0]["max_tokens"] == 32


def test_cli_disables_openai_sdk_retries_for_failing_provider_request(tmp_path, monkeypatch):
    pytest.importorskip("inspect_ai")
    pytest.importorskip("openai")
    from inspect_ai.model import GenerateConfig
    from inspect_ai.model import get_model as real_get_model

    args = build_parser().parse_args(_required_cli_arguments(tmp_path) + ["--method", "react"])
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            500,
            json={"error": {"message": "provider failed", "type": "server_error"}},
        )

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:

            def get_model_with_transport(*model_args, **model_kwargs):
                model_kwargs.update(http_client=client, memoize=False)
                return real_get_model(*model_args, **model_kwargs)

            async def generate_once(task, *, model, **kwargs):
                await model.generate(
                    "Return DONE",
                    config=GenerateConfig(max_retries=0),
                )

            fake_results = types.ModuleType("justatom.agentic.inspect_results")
            fake_results.export_results = lambda logs, output, **kwargs: {"missing": 1}
            monkeypatch.setitem(sys.modules, "justatom.agentic.inspect_results", fake_results)
            monkeypatch.setattr(inspect_cli, "build_task", lambda cases, retriever, **kwargs: "react-task")
            monkeypatch.setattr("inspect_ai.model.get_model", get_model_with_transport)
            monkeypatch.setattr("inspect_ai.eval_async", generate_once)

            with pytest.raises(Exception):
                await run(args, client=client)

    asyncio.run(exercise())
    assert attempts == 1


def test_invalid_budget_is_rejected_before_client_or_model_use(tmp_path, monkeypatch):
    args = build_parser().parse_args(_required_cli_arguments(tmp_path))
    args.max_searches = 0

    class ForbiddenClient:
        async def __aenter__(self):
            raise AssertionError("HTTP client created before validation")

    monkeypatch.setattr(inspect_cli.httpx, "AsyncClient", ForbiddenClient)

    with pytest.raises(ValueError, match="max_searches"):
        asyncio.run(run(args))


def test_default_http_client_uses_run_time_limit(tmp_path, monkeypatch):
    args = build_parser().parse_args(_required_cli_arguments(tmp_path) + ["--time-limit", "17"])
    client_timeouts: list[int] = []

    class FakeClient:
        def __init__(self, *, timeout):
            client_timeouts.append(timeout)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    async def fake_execute(*args, **kwargs):
        return {"runs": 0}

    monkeypatch.setattr(inspect_cli.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(inspect_cli, "_execute", fake_execute)

    assert asyncio.run(run(args)) == {"runs": 0}
    assert client_timeouts == [17]


@pytest.mark.parametrize(
    ("option", "url"),
    [
        ("--search-url", "https://user:password@search.test/searching"),
        ("--search-url", "https://search.test/searching?token=secret"),
        ("--search-url", "https://search.test/searching#fragment"),
        ("--base-url", "https://user:password@models.test/v1"),
        ("--base-url", "https://models.test/v1?api_key=secret"),
        ("--base-url", "https://models.test/v1#fragment"),
    ],
)
def test_credential_bearing_urls_are_rejected_before_network(tmp_path, monkeypatch, option, url):
    arguments = _required_cli_arguments(tmp_path)
    arguments[arguments.index(option) + 1] = url
    args = build_parser().parse_args(arguments)
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")

    async def forbidden(*args, **kwargs):
        raise AssertionError("network or evaluation reached before URL validation")

    monkeypatch.setattr(inspect_cli, "_execute", forbidden)
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    try:
        with pytest.raises(ValueError, match="credentials|query|fragment"):
            asyncio.run(run(args, client=async_client))
    finally:
        asyncio.run(async_client.aclose())


def test_api_key_policy_allows_local_placeholder_but_rejects_remote_missing_key(monkeypatch):
    monkeypatch.delenv("MISSING_TEST_API_KEY", raising=False)

    assert inspect_cli._api_key("MISSING_TEST_API_KEY", "http://127.0.0.1:9000/v1") == "unused"
    assert inspect_cli._api_key("MISSING_TEST_API_KEY", "http://localhost:9000/v1") == "unused"
    with pytest.raises(ValueError, match="MISSING_TEST_API_KEY"):
        inspect_cli._api_key("MISSING_TEST_API_KEY", "https://models.test/v1")


def test_run_orchestrates_both_methods_without_leaking_credentials(tmp_path, monkeypatch):
    pytest.importorskip("inspect_ai")
    arguments = _required_cli_arguments(tmp_path)
    arguments[arguments.index("http://127.0.0.1:8000/searching")] = "https://search.test/searching"
    arguments[arguments.index("http://127.0.0.1:9000/v1")] = "https://models.test/v1"
    arguments.extend(["--concurrency", "2"])
    args = build_parser().parse_args(arguments)
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")

    built: list[tuple[str, dict]] = []
    backend_calls: list[tuple[tuple, dict]] = []
    eval_calls: list[tuple[object, dict]] = []
    model_calls: list[tuple[tuple, dict]] = []
    export_calls: list[tuple[object, Path, dict]] = []

    class FakeBackend:
        def __init__(self, *backend_args, **backend_kwargs):
            backend_calls.append((backend_args, backend_kwargs))

    def fake_build_task(cases, retriever, **kwargs):
        built.append((kwargs["method"], kwargs))
        if kwargs["method"] == "native":
            kwargs["planner_factory"]()
        return f"task-{kwargs['method']}"

    react_model = object()

    def fake_get_model(*model_args, **model_kwargs):
        model_calls.append((model_args, model_kwargs))
        return react_model

    async def fake_eval_async(task, **kwargs):
        eval_calls.append((task, kwargs))
        return [f"log-{task}"]

    def fake_export(logs, output, **kwargs):
        export_calls.append((logs, output, kwargs))
        return {"runs": 2}

    fake_results = types.ModuleType("justatom.agentic.inspect_results")
    fake_results.export_results = fake_export
    monkeypatch.setitem(sys.modules, "justatom.agentic.inspect_results", fake_results)
    monkeypatch.setattr(inspect_cli, "build_task", fake_build_task)
    monkeypatch.setattr(inspect_cli, "OpenAICompatibleChatBackend", FakeBackend)
    monkeypatch.setattr("inspect_ai.model.get_model", fake_get_model)
    monkeypatch.setattr("inspect_ai.eval_async", fake_eval_async)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(500))) as client:
            return await run(args, client=client)

    summary = asyncio.run(exercise())

    assert summary == {"runs": 2}
    assert [method for method, _ in built] == ["react", "native"]
    assert model_calls == [
        (
            ("openai-api/justatom/model-id",),
            {
                "base_url": "https://models.test/v1",
                "api_key": "environment-secret",
                "max_retries": 0,
                "responses_api": False,
                "strict_tools": False,
            },
        )
    ]
    assert backend_calls == [
        (
            ("https://models.test/v1", "model-id"),
            {
                "api_key": "environment-secret",
                "timeout_seconds": 60,
                "temperature": 0,
                "max_tokens": 512,
                "objective": "context",
            },
        )
    ]
    assert eval_calls == [
        (
            "task-react",
            {
                "model": react_model,
                "max_samples": 2,
                "log_dir": str(tmp_path / "output" / "logs"),
                "log_model_api": False,
            },
        ),
        (
            "task-native",
            {
                "model": "mockllm/unused",
                "max_samples": 2,
                "log_dir": str(tmp_path / "output" / "logs"),
                "log_model_api": False,
            },
        ),
    ]
    assert export_calls[0][0] == ["log-task-react", "log-task-native"]
    assert export_calls[0][1] == tmp_path / "output"
    assert export_calls[0][2]["methods"] == ["react", "native"]
    assert [case.query_id for case in export_calls[0][2]["cases"]] == ["q1"]

    provenance = built[0][1]["provenance"]
    assert provenance == {
        "corpus_revision": "corpus-v1",
        "dataset_sha256": hashlib.sha256((Path(args.dataset)).read_bytes()).hexdigest(),
        "model": "model-id",
        "model_endpoint": "https://models.test/v1",
        "retrieval_backend": "justatom-http-search",
        "retrieval_endpoint": "https://search.test/searching",
        "retrieval_mode": "keyword",
    }
    serialized = json.dumps(provenance)
    assert "environment-secret" not in serialized


@pytest.mark.parametrize(("failing_call", "expected_logs"), [(1, []), (2, ["log-react"])])
def test_run_exports_denominator_complete_results_when_evaluation_raises(
    tmp_path,
    monkeypatch,
    failing_call,
    expected_logs,
):
    pytest.importorskip("inspect_ai")
    args = build_parser().parse_args(_required_cli_arguments(tmp_path))
    eval_count = 0
    export_calls: list[list[object]] = []

    def fake_build_task(cases, retriever, **kwargs):
        return kwargs["method"]

    async def fake_eval_async(task, **kwargs):
        nonlocal eval_count
        eval_count += 1
        if eval_count == failing_call:
            raise RuntimeError("provider failed")
        return [f"log-{task}"]

    def fake_export(logs, output, **kwargs):
        export_calls.append(list(logs))
        return {"missing": 2}

    fake_results = types.ModuleType("justatom.agentic.inspect_results")
    fake_results.export_results = fake_export
    monkeypatch.setitem(sys.modules, "justatom.agentic.inspect_results", fake_results)
    monkeypatch.setattr(inspect_cli, "build_task", fake_build_task)
    monkeypatch.setattr("inspect_ai.model.get_model", lambda *args, **kwargs: object())
    monkeypatch.setattr("inspect_ai.eval_async", fake_eval_async)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(500))) as client:
            return await run(args, client=client)

    with pytest.raises(RuntimeError, match="provider failed"):
        asyncio.run(exercise())

    assert export_calls == [expected_logs]
    assert args.output.is_dir()


def test_export_failure_does_not_mask_evaluation_failure(tmp_path, monkeypatch):
    pytest.importorskip("inspect_ai")
    args = build_parser().parse_args(_required_cli_arguments(tmp_path))

    monkeypatch.setattr(inspect_cli, "build_task", lambda cases, retriever, **kwargs: kwargs["method"])
    monkeypatch.setattr("inspect_ai.model.get_model", lambda *args, **kwargs: object())

    async def failed_eval(*args, **kwargs):
        raise RuntimeError("original provider failure")

    def failed_export(*args, **kwargs):
        raise ValueError("secondary export failure")

    fake_results = types.ModuleType("justatom.agentic.inspect_results")
    fake_results.export_results = failed_export
    monkeypatch.setitem(sys.modules, "justatom.agentic.inspect_results", fake_results)
    monkeypatch.setattr("inspect_ai.eval_async", failed_eval)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(500))) as client:
            return await run(args, client=client)

    with pytest.raises(RuntimeError, match="original provider failure"):
        asyncio.run(exercise())


def test_existing_output_is_rejected_before_model_or_eval(tmp_path, monkeypatch):
    args = build_parser().parse_args(_required_cli_arguments(tmp_path))
    args.output.mkdir()

    async def forbidden(*args, **kwargs):
        raise AssertionError("cost-bearing dependency reached")

    monkeypatch.setattr(inspect_cli, "_execute", forbidden)

    with pytest.raises(FileExistsError, match="output"):
        asyncio.run(run(args))
