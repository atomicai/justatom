import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from justatom.agentic.openai_compatible import (
    DEFAULT_SYSTEM_PROMPT,
    OpenAICompatibleChatBackend,
    OpenAICompatibleChatError,
    OpenAICompatibleResponseError,
)
from justatom.agentic.schemas import AgentAction, CallStatus, ErrorCategory, EvidenceDocument, PlannerRequest, SearchObservation


def _planner_request(question: str = "What is the answer?") -> PlannerRequest:
    document = EvidenceDocument(
        document_id="doc-1",
        content="The supporting passage.",
        score=0.75,
        rank=1,
        retrieval_index=0,
    )
    return PlannerRequest(
        question=question,
        observations=(SearchObservation(query="answer evidence", documents=(document,)),),
        context_documents=(document,),
        remaining_retrieval_calls=2,
        remaining_steps=3,
    )


def test_chat_backend_posts_strict_schema_and_maps_success_telemetry():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        assert body["model"] == "planner-model"
        assert body["temperature"] == 0.0
        assert body["max_tokens"] == 512
        assert body["seed"] == 17
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        schema = body["response_format"]["json_schema"]["schema"]
        assert set(schema) == {"type", "oneOf"}
        assert schema["type"] == "object"
        assert schema["oneOf"][0]["additionalProperties"] is False
        assert schema["oneOf"][0]["properties"]["action"] == {"type": "string", "const": "search"}
        assert schema["oneOf"][0]["properties"]["answer"] == {"type": "null"}
        assert schema["oneOf"][0]["properties"]["query"]["pattern"] == ".*\\S.*"
        assert schema["oneOf"][0]["required"] == ["action", "query", "answer", "reason", "cited_document_ids"]
        assert schema["oneOf"][1]["additionalProperties"] is False
        assert schema["oneOf"][1]["properties"]["action"] == {"type": "string", "const": "answer"}
        assert schema["oneOf"][1]["properties"]["query"] == {"type": "null"}
        assert schema["oneOf"][1]["required"] == ["action", "query", "answer", "reason", "cited_document_ids"]
        assert schema["oneOf"][1]["properties"]["cited_document_ids"]["items"]["minLength"] == 1
        assert schema["oneOf"][1]["properties"]["answer"]["pattern"] == ".*\\S.*"

        assert "For search, query must be a non-empty string" in body["messages"][0]["content"]
        assert "For answer, query must be null" in body["messages"][0]["content"]

        prompt = json.loads(body["messages"][1]["content"])
        assert prompt["question"] == "What is the answer?"
        assert prompt["observations"][0]["documents"][0]["document_id"] == "doc-1"
        assert prompt["context_documents"][0]["content"] == "The supporting passage."
        assert prompt["remaining_retrieval_calls"] == 2

        return httpx.Response(
            200,
            headers={"x-request-id": "request-123"},
            json={
                "id": "chatcmpl-ignored-when-header-present",
                "model": "served-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "action": "search",
                                    "query": "a narrower query",
                                    "answer": None,
                                    "reason": "more evidence is needed",
                                    "cited_document_ids": [],
                                }
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 7,
                    "total_tokens": 37,
                    "cost_usd": 0.0012,
                    "pricing_id": "provider-2026-01",
                    "prompt_tokens_details": {"cached_tokens": 8},
                    "completion_tokens_details": {"reasoning_tokens": 2},
                },
            },
        )

    backend = OpenAICompatibleChatBackend(
        "http://chat.test/v1",
        "planner-model",
        api_key="test-key",
        seed=17,
        transport=httpx.MockTransport(handler),
    )
    reply = asyncio.run(backend.plan(_planner_request()))
    asyncio.run(backend.close())

    assert len(requests) == 1
    assert reply.decision.action is AgentAction.SEARCH
    assert reply.decision.query == "a narrower query"
    assert reply.model == "served-model"
    assert reply.provider_request_id == "request-123"
    assert reply.finish_reason == "stop"
    assert reply.cache_hit is True
    assert reply.usage is not None
    assert reply.usage.input_tokens == 30
    assert reply.usage.output_tokens == 7
    assert reply.usage.total_tokens == 37
    assert reply.usage.cached_input_tokens == 8
    assert reply.usage.reasoning_tokens == 2
    assert reply.usage.source == "provider"
    assert reply.cost is not None
    assert reply.cost.usd == 0.0012
    assert reply.cost.source == "provider"
    assert reply.cost.pricing_id == "provider-2026-01"
    assert len(reply.attempts) == 1
    assert reply.attempts[0].attempt_index == 0
    assert reply.attempts[0].status is CallStatus.OK
    assert reply.attempts[0].provider_request_id == "request-123"


def test_chat_backend_accepts_nullable_usage_fields_and_completion_id_fallback():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-123",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "action": "answer",
                                    "query": None,
                                    "answer": "The answer.",
                                    "reason": None,
                                    "cited_document_ids": ["doc-1"],
                                }
                            )
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "cost": "unknown-provider-format",
                    "prompt_tokens_details": {"cached_tokens": None},
                    "completion_tokens_details": {"reasoning_tokens": None},
                    "cache_hit": None,
                },
            },
        )

    backend = OpenAICompatibleChatBackend(
        "http://chat.test/v1/",
        "planner-model",
        transport=httpx.MockTransport(handler),
    )
    reply = asyncio.run(backend.plan(_planner_request()))
    asyncio.run(backend.close())

    assert reply.decision.action is AgentAction.ANSWER
    assert reply.decision.cited_document_ids == ("doc-1",)
    assert reply.provider_request_id == "chatcmpl-123"
    assert reply.finish_reason is None
    assert reply.cache_hit is None
    assert reply.usage is not None
    assert reply.usage.input_tokens is None
    assert reply.usage.output_tokens is None
    assert reply.usage.total_tokens is None
    assert reply.usage.cached_input_tokens is None
    assert reply.usage.reasoning_tokens is None
    assert reply.cost is None


def test_chat_backend_keeps_absent_usage_absent():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": ('{"action":"search","query":"next","answer":null,' '"reason":null,"cited_document_ids":[]}')
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    backend = OpenAICompatibleChatBackend(
        "http://chat.test",
        "planner-model",
        transport=httpx.MockTransport(handler),
    )
    reply = asyncio.run(backend.plan(_planner_request()))
    asyncio.run(backend.close())

    assert reply.usage is None
    assert reply.cache_hit is None


@pytest.mark.parametrize(
    "content",
    [
        '{"action":"lookup","query":"x"}',
        '{"action":"search","query":"x","answer":"must not coexist"}',
        '{"action":"answer","answer":"ok","unexpected":"private response body"}',
        '{"action":"answer","query":null,"answer":"ok","reason":"  ","cited_document_ids":[]}',
    ],
)
def test_chat_backend_rejects_invalid_or_non_strict_decisions(content: str):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"x-request-id": "request-invalid"},
            json={"choices": [{"message": {"content": content}}]},
        )

    backend = OpenAICompatibleChatBackend(
        "http://chat.test",
        "planner-model",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(OpenAICompatibleResponseError) as exc_info:
        asyncio.run(backend.plan(_planner_request()))
    asyncio.run(backend.close())

    assert content not in str(exc_info.value)
    assert len(exc_info.value.attempts) == 1
    attempt = exc_info.value.attempts[0]
    assert attempt.status is CallStatus.ERROR
    assert attempt.provider_request_id == "request-invalid"
    assert attempt.error is not None
    assert attempt.error.category is ErrorCategory.VALIDATION


def test_chat_backend_does_not_retry_or_leak_secrets_on_http_failure():
    request_count = 0
    secret_key = "top-secret-api-key"
    secret_question = "private prompt text"
    secret_body = "private provider response body"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        assert secret_question in request.content.decode()
        assert secret_key in request.headers["authorization"]
        return httpx.Response(503, headers={"x-request-id": "failed-request"}, text=secret_body)

    backend = OpenAICompatibleChatBackend(
        "http://chat.test/v1",
        "planner-model",
        api_key=secret_key,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(OpenAICompatibleChatError, match="HTTP 503") as exc_info:
        asyncio.run(backend.plan(_planner_request(secret_question)))
    asyncio.run(backend.close())

    rendered = f"{exc_info.value!r} {exc_info.value}"
    assert request_count == 1
    assert secret_key not in rendered
    assert secret_question not in rendered
    assert secret_body not in rendered
    assert len(exc_info.value.attempts) == 1
    attempt = exc_info.value.attempts[0]
    assert attempt.provider_request_id == "failed-request"
    assert attempt.error is not None and attempt.error.retryable is True


def test_chat_backend_sanitizes_malformed_json_content():
    secret_content = "{not json private response body"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": secret_content}}]})

    backend = OpenAICompatibleChatBackend(
        "http://chat.test",
        "planner-model",
        api_key="top-secret-api-key",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(OpenAICompatibleResponseError) as exc_info:
        asyncio.run(backend.plan(_planner_request("private prompt text")))
    asyncio.run(backend.close())

    rendered = f"{exc_info.value!r} {exc_info.value}"
    assert "top-secret-api-key" not in rendered
    assert "private prompt text" not in rendered
    assert secret_content not in rendered


def test_chat_backend_rejects_oversized_streamed_response_before_parsing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"x-request-id": "oversized-request"},
            content=b"x" * 257,
        )

    backend = OpenAICompatibleChatBackend(
        "http://chat.test",
        "planner-model",
        max_response_bytes=256,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(OpenAICompatibleResponseError, match="max_response_bytes") as exc_info:
        asyncio.run(backend.plan(_planner_request()))
    asyncio.run(backend.close())

    attempt = exc_info.value.attempts[-1]
    assert attempt.provider_request_id == "oversized-request"
    assert attempt.error is not None
    assert attempt.error.code == "response_too_large"


def test_chat_backend_close_is_idempotent_for_an_injected_client():
    class CountingClient(httpx.AsyncClient):
        close_count = 0

        async def aclose(self) -> None:
            self.close_count += 1
            await super().aclose()

    client = CountingClient(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    backend = OpenAICompatibleChatBackend("http://chat.test", "planner-model", client=client)

    async def close_twice() -> None:
        await backend.close()
        await backend.close()

    asyncio.run(close_twice())
    assert client.close_count == 1
    assert client.is_closed


def test_chat_backend_prompt_fingerprint_is_stable_and_prompt_specific():
    first = OpenAICompatibleChatBackend("http://chat.test", "model-a")
    second = OpenAICompatibleChatBackend("http://other.test", "model-b")
    custom = OpenAICompatibleChatBackend("http://chat.test", "model-a", system_prompt="A custom planner prompt")

    assert first.backend_name == "openai-compatible"
    assert first.model_name == "model-a"
    assert first.prompt_fingerprint == hashlib.sha256(DEFAULT_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert second.prompt_fingerprint == first.prompt_fingerprint
    assert custom.prompt_fingerprint != first.prompt_fingerprint

    asyncio.run(first.close())
    asyncio.run(second.close())
    asyncio.run(custom.close())


def test_backend_config_fingerprint_excludes_api_key_and_covers_sampling_config():
    first = OpenAICompatibleChatBackend("http://chat.test", "model-a", api_key="secret-one", seed=7)
    same = OpenAICompatibleChatBackend("http://chat.test", "model-a", api_key="secret-two", seed=7)
    hotter = OpenAICompatibleChatBackend("http://chat.test", "model-a", temperature=0.5, seed=7)
    other_seed = OpenAICompatibleChatBackend("http://chat.test", "model-a", seed=8)
    smaller_response = OpenAICompatibleChatBackend("http://chat.test", "model-a", seed=7, max_response_bytes=1024)

    assert first.config_fingerprint == same.config_fingerprint
    assert hotter.config_fingerprint != first.config_fingerprint
    assert other_seed.config_fingerprint != first.config_fingerprint
    assert smaller_response.config_fingerprint != first.config_fingerprint
    assert "secret-one" not in first.config_fingerprint

    async def close_all() -> None:
        await asyncio.gather(first.close(), same.close(), hotter.close(), other_seed.close(), smaller_response.close())

    asyncio.run(close_all())


def test_usage_total_is_inferred_when_provider_reports_both_components():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": ('{"action":"search","query":"next","answer":null,' '"reason":null,"cited_document_ids":[]}')
                        }
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4},
            },
        )

    backend = OpenAICompatibleChatBackend(
        "http://chat.test",
        "planner-model",
        transport=httpx.MockTransport(handler),
    )
    reply = asyncio.run(backend.plan(_planner_request()))
    asyncio.run(backend.close())

    assert reply.usage is not None
    assert reply.usage.total_tokens == 15


def test_agentic_chat_backend_import_does_not_require_openai_or_torch():
    repo_root = Path(__file__).resolve().parents[2]
    script = """
import builtins

original_import = builtins.__import__

def import_without_sdk_or_torch(name, *args, **kwargs):
    if name == "openai" or name.startswith("openai.") or name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError(f"{name} intentionally unavailable")
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_sdk_or_torch
from justatom.agentic.openai_compatible import OpenAICompatibleChatBackend

assert OpenAICompatibleChatBackend.__name__ == "OpenAICompatibleChatBackend"
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
