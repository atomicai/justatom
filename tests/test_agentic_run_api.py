from __future__ import annotations

import asyncio
from typing import Any

import pytest

from justatom.agentic.contracts import TracePersistenceError
from justatom.agentic.runtime import AgenticCapacityError, AgenticConfigurationError, AgenticRunResult
from justatom.agentic.schemas import EvidenceDocument, RunLimits, RunStatus, RunTrace, TerminationReason, TextCapturePolicy
from justatom.api.run import create_app


class FakeRetrievalRuntime:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        self.events.append("retrieval.close")


class FakeAgenticRuntime:
    def __init__(self, result: AgenticRunResult | BaseException, events: list[str] | None = None) -> None:
        self.result = result
        self.events = events if events is not None else []
        self.calls: list[dict[str, Any]] = []
        self.close_calls = 0

    async def run(self, question: str, **kwargs: Any) -> AgenticRunResult:
        self.calls.append({"question": question, **kwargs})
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    async def close(self) -> None:
        self.close_calls += 1
        self.events.append("agent.close")


def _result(
    status: RunStatus = RunStatus.COMPLETED,
    reason: TerminationReason = TerminationReason.ANSWERED,
) -> AgenticRunResult:
    answer = "API answer" if reason is TerminationReason.ANSWERED else None
    evidence = (
        EvidenceDocument(
            document_id="doc-a",
            content="private supporting text",
            score=0.91,
            rank=1,
            retrieval_index=0,
        ),
    )
    trace = RunTrace(
        schema_version=1,
        run_id="run-123",
        request_id="request-123",
        experiment_id=None,
        variant=None,
        seed=None,
        config_fingerprint="fingerprint",
        planner_config_fingerprint="planner-fingerprint",
        retrieval_config_fingerprint="retrieval-fingerprint",
        capture_text=TextCapturePolicy.NONE,
        query_sha256=None,
        normalized_query_sha256=None,
        query_text=None,
        started_at="2026-01-01T00:00:00Z",
        duration_ms=12.0,
        queue_latency_ms=1.0,
        execution_ms=11.0,
        status=status,
        termination_reason=reason,
        budget_dimension=None,
        limits=RunLimits(
            max_steps=4,
            max_retrieval_calls=3,
            max_llm_calls=3,
            max_tokens=None,
            max_duration_ms=60_000.0,
        ),
        steps=(),
        final_context_document_ids=("doc-a",),
        answer_sha256=None,
        answer_text=None,
        final_context_chars=23,
        final_cited_document_ids=("doc-a",),
    )
    return AgenticRunResult(
        run_id="run-123",
        answer=answer,
        evidence=evidence,
        trace=trace,
        metrics={
            "duration_ms": 12.0,
            "retrieval_call_count": 1,
            "final_context_chars": 23,
            "citation_count": 1,
            "citation_context_coverage": {"numerator": 1, "denominator": 1, "rate": 1.0},
        },
    )


def test_agentic_endpoint_is_explicitly_unavailable_when_disabled() -> None:
    async def scenario() -> None:
        retrieval = FakeRetrievalRuntime()
        app = create_app(runtime=retrieval, start_mq=False)

        async with app.test_app() as test_app:
            response = await test_app.test_client().post("/searching/agentic", json={"text": "question"})

        assert response.status_code == 503
        assert await response.get_json() == {"error": "agentic runtime is disabled"}
        assert retrieval.close_calls == 1

    asyncio.run(scenario())


def test_app_builds_agentic_runtime_during_lifecycle_and_closes_it_before_retrieval(monkeypatch) -> None:
    async def scenario() -> None:
        events: list[str] = []
        retrieval = FakeRetrievalRuntime(events)
        agent = FakeAgenticRuntime(_result(), events)
        build_calls: list[tuple[dict[str, Any], Any]] = []

        async def build(config, retriever):
            build_calls.append((config, retriever))
            return agent

        monkeypatch.setattr("justatom.api.run.build_agentic_runtime", build)
        app = create_app(
            config={"agentic": {"enabled": True, "variant": "lifecycle-test"}},
            runtime=retrieval,
            start_mq=False,
        )
        assert build_calls == []

        async with app.test_app():
            assert app.extensions["agentic_runtime"] is agent
            assert len(build_calls) == 1
            assert build_calls[0][0]["enabled"] is True
            assert build_calls[0][0]["variant"] == "lifecycle-test"
            assert build_calls[0][1] is retrieval

        assert events == ["agent.close", "retrieval.close"]
        assert agent.close_calls == 1
        assert retrieval.close_calls == 1

    asyncio.run(scenario())


def test_agentic_builder_failure_rolls_back_shared_retrieval_runtime(monkeypatch) -> None:
    async def scenario() -> None:
        retrieval = FakeRetrievalRuntime()

        async def fail_to_build(config, retriever):
            del config
            assert retriever is retrieval
            raise RuntimeError("agent startup failed")

        monkeypatch.setattr("justatom.api.run.build_agentic_runtime", fail_to_build)
        app = create_app(
            config={"agentic": {"enabled": True}},
            runtime=retrieval,
            start_mq=False,
        )

        with pytest.raises(RuntimeError, match="agent startup failed"):
            await app.before_serving_funcs[0]()

        assert retrieval.close_calls == 1
        assert "retrieval_runtime" not in app.extensions
        assert "agentic_runtime" not in app.extensions

    asyncio.run(scenario())


def test_falsy_non_mapping_agentic_config_is_not_silently_disabled() -> None:
    async def scenario() -> None:
        retrieval = FakeRetrievalRuntime()
        app = create_app(config={"agentic": []}, runtime=retrieval, start_mq=False)

        with pytest.raises(AgenticConfigurationError, match="must be a mapping"):
            await app.before_serving_funcs[0]()

        assert retrieval.close_calls == 1

    asyncio.run(scenario())


def test_agentic_endpoint_forwards_request_and_returns_safe_evidence_and_metrics() -> None:
    async def scenario() -> None:
        events: list[str] = []
        retrieval = FakeRetrievalRuntime(events)
        agent = FakeAgenticRuntime(_result(), events)
        app = create_app(runtime=retrieval, agentic_runtime=agent, start_mq=False)

        async with app.test_app() as test_app:
            response = await test_app.test_client().post(
                "/searching/agentic",
                json={
                    "text": "  answer this  ",
                    "request_id": "request-123",
                    "filter_by": {"language": "en"},
                    "metadata": {"experiment": "smoke"},
                },
            )
            body = await response.get_json()

        assert response.status_code == 200
        assert agent.calls == [
            {
                "question": "answer this",
                "request_id": "request-123",
                "filters": {"language": "en"},
                "metadata": {"experiment": "smoke"},
            }
        ]
        assert body == {
            "run_id": "run-123",
            "answer": "API answer",
            "status": "completed",
            "termination_reason": "answered",
            "cited_document_ids": ["doc-a"],
            "evidence": [{"id": "doc-a", "rank": 1, "score": 0.91}],
            "metrics": {
                "duration_ms": 12.0,
                "retrieval_call_count": 1,
                "final_context_chars": 23,
                "citation_count": 1,
                "citation_context_coverage": {"numerator": 1, "denominator": 1, "rate": 1.0},
            },
        }
        assert "private supporting text" not in str(body)
        assert events == ["agent.close", "retrieval.close"]
        assert agent.close_calls == 1
        assert retrieval.close_calls == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("status", "reason", "expected_status"),
    [
        (RunStatus.COMPLETED, TerminationReason.MAX_STEPS, 200),
        (RunStatus.TIMED_OUT, TerminationReason.MAX_DURATION, 504),
        (RunStatus.FAILED, TerminationReason.PLANNER_ERROR, 502),
        (RunStatus.FAILED, TerminationReason.ERROR, 500),
    ],
)
def test_agentic_endpoint_maps_runtime_status(status: RunStatus, reason: TerminationReason, expected_status: int) -> None:
    async def scenario() -> None:
        app = create_app(
            runtime=FakeRetrievalRuntime(),
            agentic_runtime=FakeAgenticRuntime(_result(status, reason)),
            start_mq=False,
        )
        async with app.test_app() as test_app:
            response = await test_app.test_client().post("/searching/agentic", json={"text": "question"})
            body = await response.get_json()

        assert response.status_code == expected_status
        assert body["status"] == status.value
        assert body["termination_reason"] == reason.value

    asyncio.run(scenario())


def test_agentic_endpoint_rejects_invalid_payloads_without_running_agent() -> None:
    async def scenario() -> None:
        agent = FakeAgenticRuntime(_result())
        app = create_app(runtime=FakeRetrievalRuntime(), agentic_runtime=agent, start_mq=False)

        async with app.test_app() as test_app:
            client = test_app.test_client()
            responses = [
                await client.post("/searching/agentic", json=[]),
                await client.post("/searching/agentic", json={}),
                await client.post("/searching/agentic", json={"text": "  "}),
                await client.post("/searching/agentic", json={"text": "q", "request_id": 7}),
                await client.post("/searching/agentic", json={"text": "q", "filter_by": "all"}),
                await client.post("/searching/agentic", json={"text": "q", "metadata": []}),
                await client.post("/searching/agentic", json={"text": "q", "top_k": 10}),
            ]

        assert all(response.status_code == 400 for response in responses)
        assert agent.calls == []

    asyncio.run(scenario())


def test_agentic_endpoint_rejects_body_before_unbounded_json_buffering() -> None:
    async def scenario() -> None:
        agent = FakeAgenticRuntime(_result())
        app = create_app(
            config={"agentic": {"max_request_bytes": 32}},
            runtime=FakeRetrievalRuntime(),
            agentic_runtime=agent,
            start_mq=False,
        )

        async with app.test_app() as test_app:
            response = await test_app.test_client().post(
                "/searching/agentic",
                data=b'{"text":"' + b"x" * 64 + b'"}',
                headers={"Content-Type": "application/json"},
            )

        assert response.status_code == 413
        assert await response.get_json() == {"error": "request body exceeds max_request_bytes"}
        assert agent.calls == []

    asyncio.run(scenario())


def test_agentic_runtime_validation_error_is_a_bad_request() -> None:
    async def scenario() -> None:
        agent = FakeAgenticRuntime(ValueError("metadata must contain only finite JSON values"))
        app = create_app(runtime=FakeRetrievalRuntime(), agentic_runtime=agent, start_mq=False)

        async with app.test_app() as test_app:
            response = await test_app.test_client().post(
                "/searching/agentic",
                json={"text": "question", "metadata": {"value": "invalid downstream"}},
            )

        assert response.status_code == 400
        assert await response.get_json() == {"error": "metadata must contain only finite JSON values"}

    asyncio.run(scenario())


def test_required_trace_persistence_failure_is_service_unavailable() -> None:
    async def scenario() -> None:
        agent = FakeAgenticRuntime(TracePersistenceError("trace_capacity_exhausted"))
        app = create_app(runtime=FakeRetrievalRuntime(), agentic_runtime=agent, start_mq=False)

        async with app.test_app() as test_app:
            response = await test_app.test_client().post("/searching/agentic", json={"text": "question"})

        assert response.status_code == 503
        assert await response.get_json() == {"error": "required trace persistence unavailable"}

    asyncio.run(scenario())


def test_agentic_capacity_exhaustion_is_retryable_too_many_requests() -> None:
    async def scenario() -> None:
        agent = FakeAgenticRuntime(AgenticCapacityError("agentic run capacity exhausted"))
        app = create_app(runtime=FakeRetrievalRuntime(), agentic_runtime=agent, start_mq=False)

        async with app.test_app() as test_app:
            response = await test_app.test_client().post("/searching/agentic", json={"text": "question"})

        assert response.status_code == 429
        assert response.headers["Retry-After"] == "1"
        assert await response.get_json() == {"error": "agentic runtime capacity exhausted"}

    asyncio.run(scenario())
