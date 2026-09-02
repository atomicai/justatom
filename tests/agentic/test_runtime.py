from __future__ import annotations

import asyncio
import copy
import itertools
import json
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import httpx
import pytest

from justatom.agentic.contracts import TracePersistenceError
from justatom.agentic.runtime import (
    AgenticCapacityError,
    AgenticConfigurationError,
    AgenticRAGRuntime,
    AgenticRuntimeConfig,
    build_agentic_runtime,
)
from justatom.agentic.schemas import (
    AgentAction,
    AttemptTrace,
    CallKind,
    CallStatus,
    ErrorCategory,
    ErrorTrace,
    PlannerDecision,
    PlannerReply,
    RunStatus,
    TerminationReason,
    TextCapturePolicy,
    TokenUsage,
)
from justatom.agentic.telemetry import JsonlTraceSink, NullTraceSink


@dataclass
class FakeDocument:
    id: str
    content: str
    score: float | None = None


class ScriptedRetriever:
    mode = "test"

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, int, dict[str, Any] | None]] = []

    async def retrieve(self, query: str, *, top_k: int = 5, **kwargs: Any) -> list[FakeDocument]:
        self.calls.append((query, top_k, kwargs.get("filters")))
        if not self.responses:
            raise AssertionError("unexpected retrieval call")
        response = self.responses.pop(0)
        if callable(response):
            response = response(query, top_k, kwargs.get("filters"))
        if asyncio.iscoroutine(response) or isinstance(response, asyncio.Future):
            response = await response
        if isinstance(response, BaseException):
            raise response
        return response


class ScriptedBackend:
    backend_name = "fake-planner"
    model_name = "fake-model"
    prompt_fingerprint = "fake-prompt-fingerprint"

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.requests = []
        self.close_calls = 0

    async def plan(self, request):
        self.requests.append(request)
        if not self.replies:
            raise AssertionError("unexpected planner call")
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(request)
        if asyncio.iscoroutine(reply) or isinstance(reply, asyncio.Future):
            reply = await reply
        if isinstance(reply, BaseException):
            raise reply
        return reply

    async def close(self) -> None:
        self.close_calls += 1


class RecordingSink:
    def __init__(self) -> None:
        self.traces = []
        self.close_calls = 0

    async def write(self, trace) -> None:
        self.traces.append(trace)

    async def close(self) -> None:
        self.close_calls += 1


def _answer(text: str = "grounded answer", *, usage: TokenUsage | None = None) -> PlannerReply:
    return PlannerReply(
        decision=PlannerDecision(action=AgentAction.ANSWER, answer=text, cited_document_ids=("doc-a",)),
        usage=usage,
        model="served-model",
        finish_reason="stop",
    )


def _search(query: str, *, usage: TokenUsage | None = None) -> PlannerReply:
    return PlannerReply(decision=PlannerDecision(action=AgentAction.SEARCH, query=query), usage=usage)


def _runtime(
    retriever: ScriptedRetriever,
    backend: ScriptedBackend,
    *,
    config: AgenticRuntimeConfig | None = None,
    sink: RecordingSink | None = None,
) -> tuple[AgenticRAGRuntime, RecordingSink]:
    trace_sink = sink or RecordingSink()
    ids = (f"id-{index}" for index in itertools.count())
    return (
        AgenticRAGRuntime(
            retriever,
            backend,
            config=config,
            trace_sink=trace_sink,
            id_factory=lambda: next(ids),
        ),
        trace_sink,
    )


def test_initial_retrieval_then_answer_records_complete_trace_and_nullable_usage() -> None:
    async def scenario() -> None:
        retriever = ScriptedRetriever([[FakeDocument("doc-a", "support", 0.9)]])
        backend = ScriptedBackend([_answer()])
        runtime, sink = _runtime(retriever, backend)

        result = await runtime.run("What is supported?", request_id="request-1", metadata={"case": "happy"})

        assert result.answer == "grounded answer"
        assert [document.document_id for document in result.evidence] == ["doc-a"]
        assert result.trace.status is RunStatus.COMPLETED
        assert result.trace.termination_reason is TerminationReason.ANSWERED
        assert result.trace.request_id == "request-1"
        assert result.trace.metadata == {"case": "happy"}
        assert result.trace.query_text is None
        assert result.trace.query_sha256 is not None
        assert result.trace.capture_text is TextCapturePolicy.HASH
        assert result.trace.planner_config_fingerprint
        assert result.trace.retrieval_config_fingerprint
        assert result.trace.final_context_chars == len("support")
        assert result.trace.final_cited_document_ids == ("doc-a",)
        assert [step.action for step in result.trace.steps] == [AgentAction.SEARCH, AgentAction.ANSWER]
        assert result.trace.steps[-1].decision is not None
        assert result.trace.steps[-1].decision.action is AgentAction.ANSWER
        assert result.trace.steps[-1].decision.answer_sha256 == result.trace.answer_sha256
        calls = [call for step in result.trace.steps for call in step.calls]
        assert [call.kind for call in calls] == [
            CallKind.RETRIEVAL,
            CallKind.PLANNER,
        ]
        assert all(call.queue_latency_ms is not None for call in calls)
        assert result.metrics["call_queue_latency_coverage_by_kind"]["retrieval"] == {
            "numerator": 1,
            "denominator": 1,
            "rate": 1.0,
        }
        assert result.metrics["call_execution_latency_ms_by_kind"]["retrieval"]["count"] == 1
        assert result.metrics["answered"] is True
        assert result.metrics["retrieval_call_count"] == 1
        assert result.metrics["llm_call_count"] == 1
        assert result.metrics["final_context_chars"] == len("support")
        assert result.metrics["citation_count"] == 1
        assert result.metrics["unique_citation_count"] == 1
        assert result.metrics["citations_in_context_count"] == 1
        assert result.metrics["citation_context_coverage"] == {"numerator": 1, "denominator": 1, "rate": 1.0}
        assert result.metrics["token_totals"]["total_tokens"] is None
        assert result.metrics["token_usage_coverage"] == {"numerator": 0, "denominator": 1, "rate": 0.0}
        assert sink.traces == [result.trace]

        await runtime.close()
        assert backend.close_calls == 1
        assert sink.close_calls == 1

    asyncio.run(scenario())


def test_search_then_answer_deduplicates_and_caps_context_and_forwards_filters() -> None:
    async def scenario() -> None:
        retriever = ScriptedRetriever(
            [
                [FakeDocument("doc-a", "abcdef", 0.9), FakeDocument("doc-b", "wxyz", 0.8)],
                [FakeDocument("doc-a", "changed duplicate", 0.7), FakeDocument("doc-c", "cccc", 0.6)],
            ]
        )
        backend = ScriptedBackend([_search("narrower query"), _answer("final")])
        config = AgenticRuntimeConfig(
            max_steps=4,
            max_retrieval_calls=2,
            max_llm_calls=2,
            top_k=7,
            max_document_chars=4,
            max_context_chars=6,
            max_context_documents=2,
        )
        runtime, _ = _runtime(retriever, backend, config=config)
        filters = {"language": "en", "year": {"gte": 2020}}

        result = await runtime.run("broad query", filters=filters)

        assert result.answer == "final"
        assert retriever.calls == [
            ("broad query", 7, filters),
            ("narrower query", 7, filters),
        ]
        assert len(backend.requests) == 2
        assert all(document.content == "" for document in backend.requests[0].observations[0].documents)
        assert [(document.document_id, document.content) for document in backend.requests[0].context_documents] == [
            ("doc-a", "abcd"),
            ("doc-b", "wx"),
        ]
        assert backend.requests[1].context_documents == backend.requests[0].context_documents
        assert [(document.document_id, document.content) for document in result.evidence] == [
            ("doc-a", "abcd"),
            ("doc-b", "wx"),
        ]
        assert result.trace.final_context_document_ids == ("doc-a", "doc-b")
        assert result.trace.final_context_chars == 6
        assert result.trace.final_cited_document_ids == ("doc-a",)
        assert result.metrics["final_context_chars"] == 6
        assert result.metrics["citation_context_coverage"]["rate"] == 1.0
        assert result.metrics["retrieval_document_occurrence_count"] == 4
        assert result.metrics["retrieval_unique_document_count"] == 3

        await runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("capture_text", "expects_hash", "expects_raw"),
    [
        (TextCapturePolicy.NONE, False, False),
        (TextCapturePolicy.HASH, True, False),
        (TextCapturePolicy.FULL, True, True),
    ],
)
def test_filter_trace_respects_text_capture_policy(
    capture_text: TextCapturePolicy,
    expects_hash: bool,
    expects_raw: bool,
) -> None:
    async def scenario() -> None:
        retriever = ScriptedRetriever([[FakeDocument("doc-a", "support")]])
        backend = ScriptedBackend([_answer()])
        runtime, _ = _runtime(retriever, backend, config=AgenticRuntimeConfig(capture_text=capture_text))
        filters = {"tenant": "private-tenant", "year": {"gte": 2024}}

        result = await runtime.run("question", filters=filters)

        assert (result.trace.filters_sha256 is not None) is expects_hash
        assert (result.trace.filters is not None) is expects_raw
        if expects_raw:
            assert result.trace.filters == filters
        else:
            assert "private-tenant" not in str(result.trace.to_dict())
        await runtime.close()

    asyncio.run(scenario())


def test_repeated_query_is_detected_after_normalization() -> None:
    async def scenario() -> None:
        retriever = ScriptedRetriever([[FakeDocument("doc-a", "support")]])
        backend = ScriptedBackend([_search("  original   QUESTION  ")])
        runtime, _ = _runtime(retriever, backend)

        result = await runtime.run("Original Question")

        assert result.trace.status is RunStatus.COMPLETED
        assert result.trace.termination_reason is TerminationReason.REPEATED_QUERY
        assert len(retriever.calls) == 1
        await runtime.close()

    asyncio.run(scenario())


def test_no_progress_stops_before_another_empty_retrieval() -> None:
    async def scenario() -> None:
        retriever = ScriptedRetriever([[]])
        backend = ScriptedBackend([_search("different query")])
        runtime, _ = _runtime(retriever, backend, config=AgenticRuntimeConfig(no_progress_limit=1))

        result = await runtime.run("initial query")

        assert result.trace.status is RunStatus.COMPLETED
        assert result.trace.termination_reason is TerminationReason.NO_PROGRESS
        assert len(retriever.calls) == 1
        await runtime.close()

    asyncio.run(scenario())


def test_retrieval_budget_stops_before_planned_search() -> None:
    async def scenario() -> None:
        retriever = ScriptedRetriever([[FakeDocument("doc-a", "support")]])
        backend = ScriptedBackend([_search("another query")])
        config = AgenticRuntimeConfig(max_retrieval_calls=1)
        runtime, _ = _runtime(retriever, backend, config=config)

        result = await runtime.run("initial query")

        assert result.trace.termination_reason is TerminationReason.MAX_RETRIEVAL_CALLS
        assert result.trace.budget_dimension == "max_retrieval_calls"
        assert len(retriever.calls) == 1
        decision = result.trace.steps[-1].decision
        assert decision is not None
        assert decision.action is AgentAction.SEARCH
        assert decision.query_sha256 is not None
        assert decision.query_text is None
        await runtime.close()

    asyncio.run(scenario())


def test_llm_budget_is_checked_before_a_second_planner_call() -> None:
    async def scenario() -> None:
        retriever = ScriptedRetriever([[FakeDocument("doc-a", "one")], [FakeDocument("doc-b", "two")]])
        backend = ScriptedBackend([_search("another query")])
        config = AgenticRuntimeConfig(max_steps=4, max_retrieval_calls=2, max_llm_calls=1)
        runtime, _ = _runtime(retriever, backend, config=config)

        result = await runtime.run("initial query")

        assert result.trace.termination_reason is TerminationReason.MAX_LLM_CALLS
        assert result.trace.budget_dimension == "max_llm_calls"
        assert len(retriever.calls) == 2
        assert len(backend.requests) == 1
        await runtime.close()

    asyncio.run(scenario())


def test_step_budget_is_a_hard_bound_on_executed_actions() -> None:
    async def scenario() -> None:
        retriever = ScriptedRetriever([[FakeDocument("doc-a", "one")]])
        backend = ScriptedBackend([_search("would exceed the step budget")])
        config = AgenticRuntimeConfig(max_steps=2, max_retrieval_calls=2)
        runtime, _ = _runtime(retriever, backend, config=config)

        result = await runtime.run("initial query")

        assert result.trace.termination_reason is TerminationReason.MAX_STEPS
        assert result.trace.budget_dimension == "max_steps"
        assert len(result.trace.steps) == config.max_steps
        assert len(retriever.calls) == 1
        await runtime.close()

    asyncio.run(scenario())


def test_token_budget_stops_before_executing_planned_search() -> None:
    async def scenario() -> None:
        retriever = ScriptedRetriever([[FakeDocument("doc-a", "support")]])
        usage = TokenUsage(input_tokens=8, output_tokens=2, total_tokens=10, source="provider")
        backend = ScriptedBackend([_search("another query", usage=usage)])
        config = AgenticRuntimeConfig(max_tokens=10, max_retrieval_calls=2)
        runtime, _ = _runtime(retriever, backend, config=config)

        result = await runtime.run("initial query")

        assert result.trace.termination_reason is TerminationReason.MAX_TOKENS
        assert result.trace.budget_dimension == "max_tokens"
        assert len(retriever.calls) == 1
        assert result.metrics["token_totals"]["total_tokens"] == 10
        assert result.metrics["token_budget"] == {
            "limit": 10,
            "observed_total": 10,
            "coverage": {"numerator": 1, "denominator": 1, "rate": 1.0},
            "reached": True,
            "overrun": 0,
        }
        await runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("slow_component", "expected_kind"),
    [("retrieval", CallKind.RETRIEVAL), ("planner", CallKind.PLANNER)],
)
def test_per_call_deadlines_are_traced(slow_component: str, expected_kind: CallKind) -> None:
    async def scenario() -> None:
        never = asyncio.Event()

        async def wait_forever(*args: Any) -> Any:
            del args
            await never.wait()

        retriever = ScriptedRetriever([wait_forever if slow_component == "retrieval" else [FakeDocument("doc-a", "support")]])
        backend = ScriptedBackend([wait_forever if slow_component == "planner" else _answer()])
        config = AgenticRuntimeConfig(
            total_timeout_seconds=1.0,
            retrieval_timeout_seconds=0.02,
            planner_timeout_seconds=0.02,
        )
        runtime, sink = _runtime(retriever, backend, config=config)

        result = await runtime.run("question")

        assert result.trace.status is RunStatus.TIMED_OUT
        assert result.trace.termination_reason is TerminationReason.TIMEOUT
        calls = [call for step in result.trace.steps for call in step.calls]
        assert calls[-1].kind is expected_kind
        assert calls[-1].status is CallStatus.TIMEOUT
        assert sink.traces == [result.trace]
        never.set()
        await runtime.close()

    asyncio.run(scenario())


def test_component_capacity_timeout_is_distinct_from_upstream_timeout() -> None:
    async def scenario() -> None:
        runtime, _ = _runtime(
            ScriptedRetriever([]),
            ScriptedBackend([]),
            config=AgenticRuntimeConfig(
                total_timeout_seconds=1.0,
                retrieval_timeout_seconds=0.02,
                max_concurrency=1,
            ),
        )
        await runtime._component_semaphore.acquire()
        try:
            result = await runtime.run("question")
        finally:
            runtime._component_semaphore.release()

        call = result.trace.steps[0].calls[0]
        assert call.status is CallStatus.TIMEOUT
        assert call.error is not None
        assert call.error.component == "runtime"
        assert call.error.code == "retrieval_capacity_timeout"
        assert call.attempts == ()
        assert result.metrics["attempt_count"] == 0
        assert call.queue_latency_ms is not None
        assert call.queue_latency_ms == pytest.approx(call.latency_ms, abs=1.0)
        await runtime.close()

    asyncio.run(scenario())


def test_total_execution_deadline_cancels_call_and_retains_partial_trace() -> None:
    async def scenario() -> None:
        never = asyncio.Event()

        async def blocked(*args: Any) -> Any:
            del args
            await never.wait()

        retriever = ScriptedRetriever([blocked])
        backend = ScriptedBackend([])
        config = AgenticRuntimeConfig(total_timeout_seconds=0.02, retrieval_timeout_seconds=1.0)
        runtime, sink = _runtime(retriever, backend, config=config)

        result = await runtime.run("question")

        assert result.trace.status is RunStatus.TIMED_OUT
        assert result.trace.termination_reason is TerminationReason.MAX_DURATION
        assert result.trace.budget_dimension == "max_duration"
        calls = [call for step in result.trace.steps for call in step.calls]
        assert len(calls) == 1
        assert calls[0].kind is CallKind.RETRIEVAL
        assert calls[0].status is CallStatus.CANCELLED
        assert sink.traces == [result.trace]
        never.set()
        await runtime.close()

    asyncio.run(scenario())


def test_total_deadline_returns_while_cancellation_resistant_retrieval_drains() -> None:
    async def scenario() -> None:
        drained = asyncio.Event()

        async def cancellation_resistant(*args: Any) -> Any:
            del args
            worker = asyncio.create_task(asyncio.sleep(0.2))
            cancellation: asyncio.CancelledError | None = None
            while True:
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError as error:
                    cancellation = error
                    continue
                break
            drained.set()
            if cancellation is not None:
                raise cancellation
            return []

        retriever = ScriptedRetriever([cancellation_resistant, []])
        runtime, _ = _runtime(
            retriever,
            ScriptedBackend([]),
            config=AgenticRuntimeConfig(
                total_timeout_seconds=0.02,
                retrieval_timeout_seconds=1.0,
                max_concurrency=1,
            ),
        )

        first = await runtime.run("first question")
        second = await runtime.run("second question")

        assert not drained.is_set()
        assert first.trace.termination_reason is TerminationReason.MAX_DURATION
        assert second.trace.termination_reason is TerminationReason.MAX_DURATION
        assert len(retriever.calls) == 1

        await runtime.close()
        assert drained.is_set()

    asyncio.run(scenario())


def test_component_timeout_retains_permit_until_real_to_thread_work_finishes() -> None:
    async def scenario() -> None:
        worker_started = threading.Event()
        release_worker = threading.Event()
        worker_finished = threading.Event()

        def blocking_worker() -> list[FakeDocument]:
            worker_started.set()
            release_worker.wait()
            worker_finished.set()
            return [FakeDocument("late-doc", "late support")]

        async def threaded_retrieval(*_args: Any) -> list[FakeDocument]:
            return await asyncio.to_thread(blocking_worker)

        retriever = ScriptedRetriever(
            [
                threaded_retrieval,
                [FakeDocument("doc-a", "support")],
            ]
        )
        runtime, _ = _runtime(
            retriever,
            ScriptedBackend([_answer()]),
            config=AgenticRuntimeConfig(
                total_timeout_seconds=1.0,
                retrieval_timeout_seconds=0.02,
                max_concurrency=1,
            ),
        )

        try:
            first = await runtime.run("first question")
            assert first.trace.termination_reason is TerminationReason.TIMEOUT
            assert worker_started.is_set()
            assert not worker_finished.is_set()

            second = await runtime.run("second question")
            second_call = second.trace.steps[0].calls[0]
            assert second_call.status is CallStatus.TIMEOUT
            assert second_call.error is not None
            assert second_call.error.code == "retrieval_capacity_timeout"
            assert len(retriever.calls) == 1

            release_worker.set()
            await asyncio.wait_for(runtime._operations_idle.wait(), timeout=1.0)
            assert worker_finished.is_set()

            third = await runtime.run("third question")
            assert third.trace.termination_reason is TerminationReason.ANSWERED
            assert len(retriever.calls) == 2
        finally:
            release_worker.set()
            await runtime.close()

    asyncio.run(scenario())


def test_run_cancellation_retains_permit_until_real_to_thread_work_finishes() -> None:
    async def scenario() -> None:
        worker_started = threading.Event()
        release_worker = threading.Event()
        worker_finished = threading.Event()

        def blocking_worker() -> list[FakeDocument]:
            worker_started.set()
            release_worker.wait()
            worker_finished.set()
            return [FakeDocument("late-doc", "late support")]

        async def threaded_retrieval(*_args: Any) -> list[FakeDocument]:
            return await asyncio.to_thread(blocking_worker)

        retriever = ScriptedRetriever(
            [
                threaded_retrieval,
                [FakeDocument("doc-a", "support")],
            ]
        )
        runtime, _ = _runtime(
            retriever,
            ScriptedBackend([_answer()]),
            config=AgenticRuntimeConfig(
                total_timeout_seconds=1.0,
                retrieval_timeout_seconds=1.0,
                max_concurrency=1,
            ),
        )

        first_task = asyncio.create_task(runtime.run("cancelled question"))
        second_task: asyncio.Task[Any] | None = None
        try:
            assert await asyncio.to_thread(worker_started.wait, 1.0)
            first_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first_task
            assert not worker_finished.is_set()

            second_task = asyncio.create_task(runtime.run("queued question"))
            await asyncio.sleep(0.02)
            assert not second_task.done()
            assert len(retriever.calls) == 1

            second_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await second_task
            release_worker.set()
            await asyncio.wait_for(runtime._operations_idle.wait(), timeout=1.0)
            assert worker_finished.is_set()

            third = await runtime.run("third question")
            assert third.trace.termination_reason is TerminationReason.ANSWERED
            assert len(retriever.calls) == 2
        finally:
            release_worker.set()
            if not first_task.done():
                first_task.cancel()
                await asyncio.gather(first_task, return_exceptions=True)
            if second_task is not None and not second_task.done():
                second_task.cancel()
                await asyncio.gather(second_task, return_exceptions=True)
            await runtime.close()

    asyncio.run(scenario())


def test_total_deadline_includes_semaphore_queue_time() -> None:
    async def scenario() -> None:
        retriever = ScriptedRetriever([])
        backend = ScriptedBackend([])
        config = AgenticRuntimeConfig(total_timeout_seconds=0.02, max_concurrency=1)
        runtime, sink = _runtime(retriever, backend, config=config)

        await runtime._semaphore.acquire()  # Simulate a saturated worker without timing races between two runs.
        try:
            result = await runtime.run("queued question")
        finally:
            runtime._semaphore.release()

        assert result.trace.status is RunStatus.TIMED_OUT
        assert result.trace.termination_reason is TerminationReason.MAX_DURATION
        assert result.trace.steps == ()
        assert result.trace.queue_latency_ms > 0
        assert result.trace.execution_ms == 0
        assert sink.traces == [result.trace]
        await runtime.close()

    asyncio.run(scenario())


def test_run_admission_rejects_burst_beyond_bounded_queue() -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def blocking_retrieval(*_args: Any) -> list[FakeDocument]:
            entered.set()
            await release.wait()
            return []

        runtime, sink = _runtime(
            ScriptedRetriever([blocking_retrieval, []]),
            ScriptedBackend([_answer(), _answer()]),
            config=AgenticRuntimeConfig(max_concurrency=1, max_queued_runs=1),
        )
        first = asyncio.create_task(runtime.run("first question"))
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        second = asyncio.create_task(runtime.run("second question"))
        for _ in range(10):
            if runtime._accepted_runs == 2:
                break
            await asyncio.sleep(0)
        assert runtime._accepted_runs == 2

        with pytest.raises(AgenticCapacityError, match="capacity exhausted"):
            await runtime.run("rejected question")
        assert await runtime.admission_metrics() == {
            "current_admitted_run_count": 2,
            "rejected_run_count": 1,
            "max_concurrency": 1,
            "max_queued_runs": 1,
            "admission_capacity": 2,
        }

        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=1.0)
        assert len(sink.traces) == 2
        assert (await runtime.admission_metrics())["current_admitted_run_count"] == 0
        await runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_run_queue_capacity_must_be_a_nonnegative_integer(value: Any) -> None:
    with pytest.raises(AgenticConfigurationError, match="max_queued_runs"):
        AgenticRuntimeConfig(max_queued_runs=value)


def test_invalid_planner_reply_is_classified_and_traced() -> None:
    class InvalidDecision:
        action = "execute_arbitrary_tool"
        query = None
        answer = None

    class InvalidReply:
        decision = InvalidDecision()
        attempts = ()
        usage = None
        cost = None
        model = None
        provider_request_id = None
        cache_hit = None
        time_to_first_token_ms = None
        finish_reason = None

    async def scenario() -> None:
        retriever = ScriptedRetriever([[FakeDocument("doc-a", "support")]])
        backend = ScriptedBackend([InvalidReply()])
        runtime, sink = _runtime(retriever, backend)

        result = await runtime.run("question")

        assert result.trace.status is RunStatus.FAILED
        assert result.trace.termination_reason is TerminationReason.INVALID_ACTION
        planner_calls = [call for step in result.trace.steps for call in step.calls if call.kind is CallKind.PLANNER]
        assert len(planner_calls) == 1
        assert planner_calls[0].status is CallStatus.ERROR
        assert sink.traces == [result.trace]
        await runtime.close()

    asyncio.run(scenario())


def test_oversized_planner_answer_is_an_invalid_action() -> None:
    async def scenario() -> None:
        retriever = ScriptedRetriever([[FakeDocument("doc-a", "support")]])
        provider_attempt = AttemptTrace(attempt_index=0, status=CallStatus.OK, latency_ms=2.0)
        reply = PlannerReply(
            decision=PlannerDecision(action=AgentAction.ANSWER, answer="answer is too long"),
            attempts=(provider_attempt,),
        )
        backend = ScriptedBackend([reply])
        runtime, _ = _runtime(retriever, backend, config=AgenticRuntimeConfig(max_answer_chars=5))

        result = await runtime.run("question")

        assert result.trace.status is RunStatus.FAILED
        assert result.trace.termination_reason is TerminationReason.INVALID_ACTION
        planner_call = result.trace.steps[-1].calls[0]
        assert planner_call.kind is CallKind.PLANNER
        assert planner_call.status is CallStatus.ERROR
        assert planner_call.attempts == (provider_attempt,)
        assert planner_call.error is not None
        assert planner_call.error.category is ErrorCategory.VALIDATION
        await runtime.close()

    asyncio.run(scenario())


def test_planner_failure_uses_terminal_attempt_instead_of_any_earlier_timeout() -> None:
    async def scenario() -> None:
        timeout_error = ErrorTrace(
            component="planner",
            category=ErrorCategory.TIMEOUT,
            code="provider_timeout",
            retryable=True,
        )
        terminal_error = ErrorTrace(
            component="planner",
            category=ErrorCategory.BACKEND,
            code="provider_rejected",
        )
        failure = RuntimeError("sanitized backend failure")
        failure.attempts = (
            AttemptTrace(attempt_index=0, status=CallStatus.TIMEOUT, latency_ms=1.0, error=timeout_error),
            AttemptTrace(attempt_index=1, status=CallStatus.ERROR, latency_ms=1.0, error=terminal_error),
        )
        runtime, _ = _runtime(
            ScriptedRetriever([[FakeDocument("doc-a", "support")]]),
            ScriptedBackend([failure]),
        )

        result = await runtime.run("question")

        planner_call = result.trace.steps[-1].calls[0]
        assert result.trace.status is RunStatus.FAILED
        assert result.trace.termination_reason is TerminationReason.PLANNER_ERROR
        assert planner_call.status is CallStatus.ERROR
        assert planner_call.error == terminal_error
        assert planner_call.attempts == failure.attempts
        await runtime.close()

    asyncio.run(scenario())


def test_oversized_planner_reason_is_an_invalid_action() -> None:
    async def scenario() -> None:
        retriever = ScriptedRetriever([[FakeDocument("doc-a", "support")]])
        reply = PlannerReply(
            decision=PlannerDecision(
                action=AgentAction.ANSWER,
                answer="answer",
                reason="reason is too long",
            )
        )
        runtime, _ = _runtime(
            retriever,
            ScriptedBackend([reply]),
            config=AgenticRuntimeConfig(max_reason_chars=5),
        )

        result = await runtime.run("question")

        assert result.trace.status is RunStatus.FAILED
        assert result.trace.termination_reason is TerminationReason.INVALID_ACTION
        await runtime.close()

    asyncio.run(scenario())


def test_request_filter_and_metadata_size_limits_apply_before_execution() -> None:
    async def scenario() -> None:
        retriever = ScriptedRetriever([])
        backend = ScriptedBackend([])
        runtime, _ = _runtime(
            retriever,
            backend,
            config=AgenticRuntimeConfig(max_filter_chars=8, max_metadata_chars=8),
        )

        with pytest.raises(ValueError, match="max_filter_chars"):
            await runtime.run("question", filters={"private": "too long"})
        with pytest.raises(ValueError, match="max_metadata_chars"):
            await runtime.run("question", metadata={"private": "too long"})
        assert retriever.calls == []
        assert backend.requests == []
        await runtime.close()

    asyncio.run(scenario())


def test_identifier_bounds_apply_to_request_retrieval_and_citations() -> None:
    async def scenario() -> None:
        config = AgenticRuntimeConfig(max_identifier_chars=5)
        request_runtime, _ = _runtime(ScriptedRetriever([]), ScriptedBackend([]), config=config)
        with pytest.raises(ValueError, match="max_identifier_chars"):
            await request_runtime.run("question", request_id="too-long")
        await request_runtime.close()

        retrieval_runtime, _ = _runtime(
            ScriptedRetriever([[FakeDocument("document-id-too-long", "support")]]),
            ScriptedBackend([]),
            config=config,
        )
        retrieval_result = await retrieval_runtime.run("question")
        assert retrieval_result.trace.termination_reason is TerminationReason.RETRIEVAL_ERROR
        assert retrieval_result.trace.steps[0].calls[0].retrieval is not None
        assert retrieval_result.trace.steps[0].calls[0].retrieval.documents == ()
        await retrieval_runtime.close()

        citation_runtime, _ = _runtime(
            ScriptedRetriever([[FakeDocument("doc-a", "support")]]),
            ScriptedBackend(
                [
                    PlannerReply(
                        decision=PlannerDecision(
                            action=AgentAction.ANSWER,
                            answer="answer",
                            cited_document_ids=("too-long",),
                        )
                    )
                ]
            ),
            config=config,
        )
        citation_result = await citation_runtime.run("question")
        assert citation_result.trace.termination_reason is TerminationReason.INVALID_ACTION
        await citation_runtime.close()

    asyncio.run(scenario())


def test_runtime_enforces_top_k_even_when_retriever_over_returns() -> None:
    async def scenario() -> None:
        documents = [FakeDocument(f"doc-{index}", f"content-{index}") for index in range(5)]
        retriever = ScriptedRetriever([documents])
        backend = ScriptedBackend([_answer()])
        runtime, _ = _runtime(retriever, backend, config=AgenticRuntimeConfig(top_k=2))

        result = await runtime.run("question")

        assert [document.document_id for document in result.evidence] == ["doc-0", "doc-1"]
        retrieval = result.trace.steps[0].calls[0].retrieval
        assert retrieval is not None
        assert len(retrieval.documents) == 2
        assert retrieval.backend_document_count == 5
        assert retrieval.truncated_document_count == 3
        assert result.metrics["retrieval_backend_document_count"] == 5
        assert result.metrics["retrieval_locally_truncated_document_count"] == 3
        await runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("capture_text", [TextCapturePolicy.NONE, TextCapturePolicy.FULL])
def test_planner_decision_trace_respects_capture_policy(capture_text: TextCapturePolicy) -> None:
    async def scenario() -> None:
        retriever = ScriptedRetriever([[FakeDocument("doc-a", "support")]])
        reply = PlannerReply(
            decision=PlannerDecision(
                action=AgentAction.ANSWER,
                answer="private answer",
                reason="private reason",
                cited_document_ids=("doc-a",),
            )
        )
        runtime, _ = _runtime(retriever, ScriptedBackend([reply]), config=AgenticRuntimeConfig(capture_text=capture_text))

        result = await runtime.run("private question")
        decision = result.trace.steps[-1].decision

        assert decision is not None
        if capture_text is TextCapturePolicy.FULL:
            assert decision.answer_text == "private answer"
            assert decision.reason_text == "private reason"
            assert decision.answer_sha256 is not None
        else:
            assert decision.answer_text is None
            assert decision.reason_text is None
            assert decision.answer_sha256 is None
            assert "private reason" not in str(result.trace.to_dict())
        await runtime.close()

    asyncio.run(scenario())


def test_backend_error_returns_failed_result_and_retains_trace() -> None:
    async def scenario() -> None:
        retriever = ScriptedRetriever([RuntimeError("retriever secret")])
        backend = ScriptedBackend([])
        runtime, sink = _runtime(retriever, backend)

        result = await runtime.run("private question")

        assert result.answer is None
        assert result.trace.status is RunStatus.FAILED
        assert result.trace.termination_reason is TerminationReason.RETRIEVAL_ERROR
        assert result.trace.error is not None
        assert result.trace.error.code == "retrieval_backend_error"
        assert result.metrics["call_error_category_counts"] == {"backend": 1}
        assert result.metrics["attempts_by_status"]["error"] == 1
        assert "retriever secret" not in str(result.trace.to_dict())
        assert sink.traces == [result.trace]
        await runtime.close()

    asyncio.run(scenario())


def test_cancellation_is_reraised_after_cancelled_trace_is_written() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        never = asyncio.Event()

        async def blocked(*args: Any) -> Any:
            del args
            started.set()
            await never.wait()

        retriever = ScriptedRetriever([blocked])
        backend = ScriptedBackend([])
        runtime, sink = _runtime(retriever, backend)
        task = asyncio.create_task(runtime.run("question"))
        await started.wait()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert len(sink.traces) == 1
        trace = sink.traces[0]
        assert trace.status is RunStatus.CANCELLED
        assert trace.termination_reason is TerminationReason.CANCELLED
        call = trace.steps[0].calls[0]
        assert call.status is CallStatus.CANCELLED
        never.set()
        await runtime.close()

    asyncio.run(scenario())


def test_cancellation_while_queued_records_queue_latency() -> None:
    async def scenario() -> None:
        runtime, sink = _runtime(
            ScriptedRetriever([]),
            ScriptedBackend([]),
            config=AgenticRuntimeConfig(max_concurrency=1),
        )
        await runtime._semaphore.acquire()
        task = asyncio.create_task(runtime.run("queued question"))
        try:
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            runtime._semaphore.release()

        assert len(sink.traces) == 1
        assert sink.traces[0].status is RunStatus.CANCELLED
        assert sink.traces[0].queue_latency_ms > 0
        assert sink.traces[0].execution_ms == 0
        await runtime.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("required", [True, False])
def test_trace_persistence_has_separate_timeout(required: bool, caplog) -> None:
    class SlowSink(RecordingSink):
        def __init__(self) -> None:
            super().__init__()
            self.cancelled = False

        async def write(self, trace) -> None:
            del trace
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    async def scenario() -> None:
        sink = SlowSink()
        runtime, _ = _runtime(
            ScriptedRetriever([[FakeDocument("doc-a", "support")]]),
            ScriptedBackend([_answer()]),
            config=AgenticRuntimeConfig(
                trace_required=required,
                trace_timeout_seconds=0.01,
            ),
            sink=sink,
        )

        if required:
            with pytest.raises(TracePersistenceError, match="trace_confirmation_timeout"):
                await runtime.run("question")
        else:
            result = await runtime.run("question")
            assert result.trace.termination_reason is TerminationReason.ANSWERED
            assert "agentic trace confirmation deadline exceeded; delivery outcome pending" in caplog.text
        assert sink.cancelled is True
        await runtime.close()

    asyncio.run(scenario())


def test_trace_deadline_detaches_cancellation_resistant_sink_and_bounds_admission(caplog) -> None:
    class DrainingSink(RecordingSink):
        def __init__(self) -> None:
            super().__init__()
            self.write_calls = 0
            self.drained = asyncio.Event()

        async def write(self, trace) -> None:
            del trace
            self.write_calls += 1
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0.2)
                self.drained.set()
                raise

    async def scenario() -> None:
        sink = DrainingSink()
        runtime, _ = _runtime(
            ScriptedRetriever(
                [
                    [FakeDocument("doc-a", "support")],
                    [FakeDocument("doc-b", "support")],
                ]
            ),
            ScriptedBackend([_answer(), _answer()]),
            config=AgenticRuntimeConfig(
                trace_required=False,
                trace_timeout_seconds=0.01,
                max_concurrency=1,
            ),
            sink=sink,
        )

        await runtime.run("first")
        await runtime.run("second")

        assert sink.write_calls == 1
        assert not sink.drained.is_set()
        assert "delivery outcome pending" in caplog.text
        assert "sink capacity remained saturated" in caplog.text

        await runtime.close()
        assert sink.drained.is_set()

    asyncio.run(scenario())


def test_close_waits_for_in_flight_trajectory_and_is_concurrent_and_idempotent() -> None:
    async def scenario() -> None:
        retrieval_started = asyncio.Event()
        release_retrieval = asyncio.Event()

        async def blocked(*args: Any) -> list[FakeDocument]:
            del args
            retrieval_started.set()
            await release_retrieval.wait()
            return [FakeDocument("doc-a", "support")]

        retriever = ScriptedRetriever([blocked])
        backend = ScriptedBackend([_answer()])
        sink = RecordingSink()
        runtime, _ = _runtime(retriever, backend, sink=sink)
        run_task = asyncio.create_task(runtime.run("question"))
        await retrieval_started.wait()

        close_tasks = [asyncio.create_task(runtime.close()), asyncio.create_task(runtime.close())]
        await asyncio.sleep(0)
        assert all(not task.done() for task in close_tasks)
        assert backend.close_calls == 0
        assert sink.close_calls == 0

        release_retrieval.set()
        result = await run_task
        await asyncio.gather(*close_tasks)
        await runtime.close()

        assert result.trace.termination_reason is TerminationReason.ANSWERED
        assert backend.close_calls == 1
        assert sink.close_calls == 1
        with pytest.raises(RuntimeError, match="closed"):
            await runtime.run("too late")

    asyncio.run(scenario())


def test_builder_disabled_does_not_construct_a_backend() -> None:
    async def scenario() -> None:
        retriever = ScriptedRetriever([])
        runtime = await build_agentic_runtime({"enabled": False}, retriever)
        assert runtime is None

    asyncio.run(scenario())


def test_direct_runtime_requires_sink_for_required_tracing() -> None:
    with pytest.raises(AgenticConfigurationError, match="trace_sink is required"):
        AgenticRAGRuntime(
            ScriptedRetriever([]),
            ScriptedBackend([]),
            config=AgenticRuntimeConfig(trace_required=True),
        )


def test_direct_runtime_rejects_discard_sink_for_required_tracing() -> None:
    with pytest.raises(AgenticConfigurationError, match="trace_sink.required must match"):
        AgenticRAGRuntime(
            ScriptedRetriever([]),
            ScriptedBackend([]),
            config=AgenticRuntimeConfig(trace_required=True),
            trace_sink=NullTraceSink(required=False),
        )


def test_builder_constructs_enabled_runtime_without_contacting_provider() -> None:
    async def scenario() -> None:
        provider_calls = 0

        def provider(request: httpx.Request) -> httpx.Response:
            nonlocal provider_calls
            provider_calls += 1
            return httpx.Response(500)

        config = {
            "enabled": True,
            "max_steps": 2,
            "planner": {
                "backend": "openai-compatible",
                "base_url": "http://planner.test/v1",
                "model": "planner-model",
            },
            "trace": {"path": None, "capture_text": "none", "required": False, "timeout_seconds": 1.5},
        }
        runtime = await build_agentic_runtime(config, ScriptedRetriever([]), transport=httpx.MockTransport(provider))

        assert runtime is not None
        assert runtime.chat_backend.backend_name == "openai-compatible"
        assert runtime.chat_backend.model_name == "planner-model"
        assert runtime.config.max_steps == 2
        assert runtime.config.capture_text.value == "none"
        assert runtime.config.trace_timeout_seconds == 1.5
        assert provider_calls == 0
        await runtime.close()

    asyncio.run(scenario())


def test_builder_requires_a_persistent_sink_when_tracing_is_required() -> None:
    async def scenario() -> None:
        config = {
            "enabled": True,
            "planner": {
                "backend": "openai-compatible",
                "base_url": "http://planner.test/v1",
                "model": "planner-model",
            },
            "trace": {"path": None, "required": True},
        }
        with pytest.raises(AgenticConfigurationError, match="trace.path is required"):
            await build_agentic_runtime(config, ScriptedRetriever([]))

    asyncio.run(scenario())


def test_builder_forwards_experiment_seed_to_planner_by_default() -> None:
    async def scenario() -> None:
        def provider(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["seed"] == 23
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "action": "answer",
                                        "query": None,
                                        "answer": "answer",
                                        "reason": None,
                                        "cited_document_ids": ["doc-a"],
                                    }
                                )
                            }
                        }
                    ]
                },
            )

        config = {
            "enabled": True,
            "seed": 23,
            "planner": {
                "backend": "openai-compatible",
                "base_url": "http://planner.test/v1",
                "model": "planner-model",
            },
            "trace": {"path": None, "required": False},
        }
        runtime = await build_agentic_runtime(
            config,
            ScriptedRetriever([[FakeDocument("doc-a", "support")]]),
            transport=httpx.MockTransport(provider),
        )
        assert runtime is not None

        result = await runtime.run("question")

        assert result.trace.seed == 23
        assert result.trace.termination_reason is TerminationReason.ANSWERED
        await runtime.close()

    asyncio.run(scenario())


def test_config_fingerprints_cover_planner_and_retrieval_configuration() -> None:
    first_retriever = ScriptedRetriever([])
    second_retriever = ScriptedRetriever([])
    second_retriever.mode = "different-test-mode"
    first, _ = _runtime(first_retriever, ScriptedBackend([]))
    second, _ = _runtime(second_retriever, ScriptedBackend([]))

    assert first.planner_config_fingerprint == second.planner_config_fingerprint
    assert first.retrieval_config_fingerprint != second.retrieval_config_fingerprint


def test_retrieval_fingerprint_covers_endpoint_transport_device_and_revision() -> None:
    def configured_retriever(**overrides: Any) -> ScriptedRetriever:
        values = {
            "url": "http://store-a.test:8080",
            "grpc_port": 50051,
            "grpc_secure": False,
            "device": "cuda:0",
            "timeout": 30.0,
            "index_revision": "corpus-v1",
        }
        values.update(overrides)
        retriever = ScriptedRetriever([])
        retriever.store = SimpleNamespace(
            collection_name="Docs",
            _url=values["url"],
            _grpc_port=values["grpc_port"],
            _grpc_secure=values["grpc_secure"],
        )
        retriever.embedder = SimpleNamespace(
            model="encoder",
            base_url="http://embedder.test/v1",
            device=values["device"],
            timeout=values["timeout"],
        )
        retriever.index_revision = values["index_revision"]
        return retriever

    baseline, _ = _runtime(configured_retriever(), ScriptedBackend([]))
    mutations = (
        {"url": "http://store-b.test:8080"},
        {"grpc_port": 50052},
        {"grpc_secure": True},
        {"device": "cpu"},
        {"timeout": 60.0},
        {"index_revision": "corpus-v2"},
    )

    for mutation in mutations:
        changed, _ = _runtime(configured_retriever(**mutation), ScriptedBackend([]))
        assert changed.retrieval_config_fingerprint != baseline.retrieval_config_fingerprint


def test_overall_fingerprint_covers_trace_sink_capacity(tmp_path) -> None:
    first = AgenticRAGRuntime(
        ScriptedRetriever([]),
        ScriptedBackend([]),
        trace_sink=JsonlTraceSink(tmp_path / "first.jsonl", max_pending_writes=1),
    )
    second = AgenticRAGRuntime(
        ScriptedRetriever([]),
        ScriptedBackend([]),
        trace_sink=JsonlTraceSink(tmp_path / "second.jsonl", max_pending_writes=2),
    )

    assert first.retrieval_config_fingerprint == second.retrieval_config_fingerprint
    assert first.planner_config_fingerprint == second.planner_config_fingerprint
    assert first.config_fingerprint != second.config_fingerprint

    asyncio.run(first.close())
    asyncio.run(second.close())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config: config.update({"mystery": True}), "unknown agentic keys"),
        (lambda config: config["planner"].update({"mystery": True}), "unknown agentic.planner keys"),
        (lambda config: config["trace"].update({"mystery": True}), "unknown agentic.trace keys"),
    ],
)
def test_builder_rejects_unknown_configuration_fields(mutation: Callable[[dict[str, Any]], None], message: str) -> None:
    async def scenario() -> None:
        config: dict[str, Any] = {
            "enabled": True,
            "planner": {
                "backend": "openai-compatible",
                "base_url": "http://planner.test/v1",
                "model": "planner-model",
            },
            "trace": {},
        }
        mutation(config)
        with pytest.raises(AgenticConfigurationError, match=message):
            await build_agentic_runtime(copy.deepcopy(config), ScriptedRetriever([]))

    asyncio.run(scenario())
