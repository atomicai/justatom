from __future__ import annotations

import asyncio
import json
import os
import stat
import threading
from dataclasses import replace

import pytest

from justatom.agentic.schemas import (
    TRACE_SCHEMA_VERSION,
    AgentAction,
    CallKind,
    CallStatus,
    CallTrace,
    CostUsage,
    DecisionTrace,
    DocumentTrace,
    RetrievalPayload,
    RunLimits,
    RunStatus,
    RunTrace,
    StepTrace,
    TerminationReason,
    TextCapturePolicy,
    TokenUsage,
)
from justatom.agentic.telemetry import (
    InMemoryTraceSink,
    JsonlTraceSink,
    NullTraceSink,
    TraceSinkOverloadedError,
    aggregate_run_metrics,
    derive_run_metrics,
    iter_jsonl_traces,
    load_jsonl_traces,
)


def _document(document_id: str, rank: int) -> DocumentTrace:
    return DocumentTrace(
        document_id=document_id,
        rank=rank,
        score=1.0 / rank,
        content_chars=1,
        content_sha256=f"hash-{document_id}",
    )


def _retrieval_call(index: int, query_hash: str, document_ids: tuple[str, ...]) -> CallTrace:
    return CallTrace(
        call_id=f"retrieval-{index}",
        call_index=index,
        kind=CallKind.RETRIEVAL,
        backend="test",
        model=None,
        started_offset_ms=float(index),
        latency_ms=5.0,
        status=CallStatus.OK,
        retrieval=RetrievalPayload(
            retrieval_index=index,
            query_sha256=f"raw-{index}",
            normalized_query_sha256=query_hash,
            query_text=None,
            mode="dense",
            collection="docs",
            index_revision=None,
            top_k_requested=3,
            documents=tuple(_document(document_id, rank) for rank, document_id in enumerate(document_ids, 1)),
            backend_document_count=len(document_ids),
        ),
    )


def _trace(*, run_id: str = "run-1", duration_ms: float = 100.0) -> RunTrace:
    planner_with_usage = CallTrace(
        call_id="planner-0",
        call_index=0,
        kind=CallKind.PLANNER,
        backend="openai-compatible",
        model="model",
        started_offset_ms=0.0,
        latency_ms=10.0,
        status=CallStatus.OK,
        tokens=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15, source="provider"),
        cost=CostUsage(usd=0.004, source="provider"),
        cache_hit=True,
        time_to_first_token_ms=2.5,
    )
    planner_without_usage = CallTrace(
        call_id="planner-1",
        call_index=3,
        kind=CallKind.PLANNER,
        backend="openai-compatible",
        model="model",
        started_offset_ms=20.0,
        latency_ms=10.0,
        status=CallStatus.OK,
    )
    steps = (
        StepTrace(
            step_index=0,
            started_offset_ms=0.0,
            latency_ms=20.0,
            action=AgentAction.SEARCH,
            calls=(planner_with_usage, _retrieval_call(0, "same-query", ("d1", "d2"))),
            context_document_ids=("d1", "d2"),
            decision=DecisionTrace(action=AgentAction.SEARCH, query_sha256="planned-query-0"),
        ),
        StepTrace(
            step_index=1,
            started_offset_ms=20.0,
            latency_ms=20.0,
            action=AgentAction.SEARCH,
            calls=(planner_without_usage, _retrieval_call(1, "same-query", ("d2", "d3"))),
            context_document_ids=("d1", "d2", "d3"),
            decision=DecisionTrace(action=AgentAction.SEARCH, query_sha256="planned-query-1"),
        ),
    )
    return RunTrace(
        schema_version=TRACE_SCHEMA_VERSION,
        run_id=run_id,
        request_id=None,
        experiment_id=None,
        variant=None,
        seed=7,
        config_fingerprint="config",
        planner_config_fingerprint="planner-config",
        retrieval_config_fingerprint="retrieval-config",
        capture_text=TextCapturePolicy.HASH,
        query_sha256="query",
        normalized_query_sha256="normalized-query",
        query_text=None,
        started_at="2026-01-01T00:00:00Z",
        duration_ms=duration_ms,
        queue_latency_ms=2.0,
        execution_ms=duration_ms - 2.0,
        status=RunStatus.COMPLETED,
        termination_reason=TerminationReason.ANSWERED,
        budget_dimension=None,
        limits=RunLimits(
            max_steps=5,
            max_retrieval_calls=5,
            max_llm_calls=5,
            max_tokens=1000,
            max_duration_ms=1000.0,
        ),
        steps=steps,
        final_context_document_ids=("d1", "d2", "d3"),
        answer_sha256="answer",
        answer_text=None,
    )


def test_null_and_memory_sinks_are_concurrency_safe_and_close_idempotently():
    async def exercise() -> None:
        trace = _trace()
        memory = InMemoryTraceSink()
        null = NullTraceSink()

        await asyncio.gather(*(memory.write(trace) for _ in range(25)))
        await asyncio.gather(*(null.write(trace) for _ in range(25)))
        assert len(await memory.snapshot()) == 25
        assert len(memory.traces) == 25

        await asyncio.gather(memory.close(), memory.close(), null.close(), null.close())
        assert memory.closed and null.closed
        with pytest.raises(RuntimeError, match="closed"):
            await memory.write(trace)
        with pytest.raises(RuntimeError, match="closed"):
            await null.write(trace)

    asyncio.run(exercise())


def test_null_sink_cannot_claim_required_delivery():
    with pytest.raises(ValueError, match="cannot be required"):
        NullTraceSink(required=True)


def test_jsonl_sink_writes_one_strict_json_object_per_concurrent_call(tmp_path):
    async def exercise() -> None:
        path = tmp_path / "nested" / "traces.jsonl"
        sink = JsonlTraceSink(path)
        await asyncio.gather(*(sink.write(_trace(run_id=f"run-{index}")) for index in range(20)))
        await asyncio.gather(sink.close(), sink.close())

        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 20
        assert {record["run_id"] for record in records} == {f"run-{index}" for index in range(20)}
        loaded = load_jsonl_traces(path)
        assert {trace.run_id for trace in loaded} == {f"run-{index}" for index in range(20)}
        assert all(RunTrace.from_dict(trace.to_dict()) == trace for trace in loaded)
        if os.name != "nt":
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    asyncio.run(exercise())


def test_jsonl_sink_timeout_does_not_block_event_loop_and_close_drains_write(tmp_path):
    class SlowJsonlTraceSink(JsonlTraceSink):
        def _append(self, line: str) -> None:
            release_write.wait()
            super()._append(line)

    async def exercise() -> None:
        nonlocal safety_timer
        path = tmp_path / "traces.jsonl"
        sink = SlowJsonlTraceSink(path, max_pending_writes=1)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(sink.write(_trace()), timeout=0.01)

        with pytest.raises(TraceSinkOverloadedError, match="capacity"):
            await sink.write(_trace(run_id="overflow"))
        release_write.set()
        safety_timer.cancel()
        await sink.close()
        assert load_jsonl_traces(path) == (_trace(),)

    release_write = threading.Event()
    safety_timer = threading.Timer(1.0, release_write.set)
    safety_timer.daemon = True
    safety_timer.start()
    try:
        asyncio.run(exercise())
    finally:
        release_write.set()
        safety_timer.cancel()


def test_jsonl_sink_cancelled_close_drains_accepted_write_before_reraising(tmp_path):
    class SlowJsonlTraceSink(JsonlTraceSink):
        def _append(self, line: str) -> None:
            write_started.set()
            release_write.wait()
            super()._append(line)

    async def exercise() -> None:
        path = tmp_path / "traces.jsonl"
        sink = SlowJsonlTraceSink(path)
        write_task = asyncio.create_task(sink.write(_trace()))
        assert await asyncio.to_thread(write_started.wait, 1.0)

        close_task = asyncio.create_task(sink.close())
        await asyncio.sleep(0)
        close_task.cancel()
        await asyncio.sleep(0)
        assert not close_task.done()

        release_write.set()
        with pytest.raises(asyncio.CancelledError):
            await close_task
        await write_task

        assert sink.closed
        assert load_jsonl_traces(path) == (_trace(),)

    write_started = threading.Event()
    release_write = threading.Event()
    safety_timer = threading.Timer(2.0, release_write.set)
    safety_timer.daemon = True
    safety_timer.start()
    try:
        asyncio.run(exercise())
    finally:
        release_write.set()
        safety_timer.cancel()


def test_jsonl_sink_cancelled_close_preserves_cancellation_when_cleanup_fails(tmp_path):
    class FailingCloseSink(JsonlTraceSink):
        async def _drain_and_close(self) -> None:
            close_started.set()
            await release_close.wait()
            raise RuntimeError("close failed")

    async def exercise() -> None:
        sink = FailingCloseSink(tmp_path / "traces.jsonl")
        close_task = asyncio.create_task(sink.close())
        await close_started.wait()

        close_task.cancel()
        await asyncio.sleep(0)
        assert not close_task.done()

        release_close.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await close_task
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert close_task.cancelled()

    close_started = asyncio.Event()
    release_close = asyncio.Event()
    asyncio.run(exercise())


@pytest.mark.skipif(not hasattr(os, "fchmod"), reason="descriptor chmod is unavailable")
def test_jsonl_sink_sets_file_permissions_through_descriptor(tmp_path, monkeypatch):
    original_fchmod = os.fchmod
    calls: list[tuple[int, int]] = []

    def record_fchmod(descriptor: int, mode: int) -> None:
        calls.append((descriptor, mode))
        original_fchmod(descriptor, mode)

    def reject_path_chmod(*args, **kwargs) -> None:
        raise AssertionError("path-based chmod must not be used for the trace file")

    monkeypatch.setattr("justatom.agentic.telemetry.os.fchmod", record_fchmod)
    monkeypatch.setattr("justatom.agentic.telemetry.os.chmod", reject_path_chmod)

    async def exercise() -> None:
        sink = JsonlTraceSink(tmp_path / "traces.jsonl")
        await sink.write(_trace())
        await sink.close()

    asyncio.run(exercise())
    assert len(calls) == 1
    assert calls[0][1] == 0o600


def test_jsonl_loader_reports_corrupt_line_without_silently_skipping_it(tmp_path):
    path = tmp_path / "traces.jsonl"
    valid = json.dumps(_trace().to_dict(), allow_nan=False)
    path.write_text(f"{valid}\n{{not-json}}\n", encoding="utf-8")

    iterator = iter_jsonl_traces(path)
    assert next(iterator).run_id == "run-1"
    with pytest.raises(ValueError, match=r"traces\.jsonl:2"):
        next(iterator)


def test_jsonl_loader_rejects_schema_type_corruption(tmp_path):
    path = tmp_path / "traces.jsonl"
    payload = _trace().to_dict()
    payload["steps"][0]["calls"][1]["retrieval"]["documents"][0]["document_id"] = 17
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"traces\.jsonl:1"):
        load_jsonl_traces(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.update(status="completed", termination_reason="planner_error"),
            "status and termination_reason",
        ),
        (
            lambda payload: payload["steps"][0]["decision"].update(reason_text="leaked planner reason"),
            "raw trace text",
        ),
        (
            lambda payload: payload["steps"][0]["calls"][1]["retrieval"]["documents"][0].update(rank=2),
            "ranks must be consecutive",
        ),
        (
            lambda payload: payload["steps"][0]["calls"][0].update(queue_latency_ms=11.0),
            "queue_latency_ms must not exceed latency_ms",
        ),
        (
            lambda payload: payload.update(duration_ms=10**1000),
            "duration_ms must be a finite",
        ),
    ],
)
def test_trace_loader_rejects_semantically_incoherent_or_privacy_violating_payloads(mutate, message):
    payload = _trace().to_dict()
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        RunTrace.from_dict(payload)


def test_trace_loader_rejects_hashes_under_none_capture_policy():
    payload = _trace().to_dict()
    payload["capture_text"] = "none"

    with pytest.raises(ValueError, match="trace hashes"):
        RunTrace.from_dict(payload)


def test_jsonl_sink_rejects_nan_before_writing(tmp_path):
    async def exercise() -> None:
        path = tmp_path / "traces.jsonl"
        trace = replace(_trace(), metadata={"bad": float("nan")})
        sink = JsonlTraceSink(path)

        with pytest.raises(ValueError, match="JSON compliant"):
            await sink.write(trace)
        await sink.close()
        assert not path.exists()

    asyncio.run(exercise())


def test_run_trace_rejects_inconsistent_terminal_state():
    trace = _trace()

    with pytest.raises(ValueError, match="answered traces"):
        replace(trace, status=RunStatus.FAILED)
    with pytest.raises(ValueError, match="cancelled status"):
        replace(trace, status=RunStatus.CANCELLED, termination_reason=TerminationReason.ERROR)


def test_derive_run_metrics_preserves_exact_token_coverage_and_diversity():
    metrics = derive_run_metrics(_trace())

    assert metrics["schema_version"] == TRACE_SCHEMA_VERSION
    assert metrics["experiment_id"] is None
    assert metrics["variant"] is None
    assert metrics["seed"] == 7
    assert metrics["config_fingerprint"] == "config"
    assert metrics["planner_config_fingerprint"] == "planner-config"
    assert metrics["retrieval_config_fingerprint"] == "retrieval-config"
    assert metrics["filters_sha256"] is None
    assert metrics["operational_success"] is True
    assert metrics["answered"] is True
    assert metrics["call_count"] == 4
    assert metrics["calls_by_kind"] == {"planner": 2, "retrieval": 2, "reranker": 0, "answer": 0}
    assert metrics["call_latency_ms_by_kind"]["retrieval"] == {
        "count": 2,
        "sum": 10.0,
        "sum_overflow": False,
        "mean": 5.0,
    }
    assert metrics["retrieval_total_ms"] == 10.0
    assert metrics["planner_total_ms"] == 20.0
    assert metrics["token_totals"]["input_tokens"] == 10
    assert metrics["token_coverage"]["input_tokens"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert metrics["token_coverage"]["cached_input_tokens"] == {
        "numerator": 0,
        "denominator": 2,
        "rate": 0.0,
    }
    assert metrics["token_budget"] == {
        "limit": 1000,
        "observed_total": 15,
        "coverage": {"numerator": 1, "denominator": 2, "rate": 0.5},
        "reached": None,
        "overrun": None,
    }
    assert metrics["cost_total_usd"] == 0.004
    assert metrics["cost_total_overflow"] is False
    assert metrics["cost_coverage"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert metrics["cache_hit_count"] == 1
    assert metrics["cache_hit_coverage"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert metrics["cache_hit_rate"] == 1.0
    assert metrics["time_to_first_token_ms"] == {
        "count": 1,
        "sum": 2.5,
        "sum_overflow": False,
        "mean": 2.5,
    }
    assert metrics["retrieval_query_hash_coverage"] == {"numerator": 2, "denominator": 2, "rate": 1.0}
    assert metrics["retrieval_query_diversity"] == 0.5
    assert metrics["retrieval_document_occurrence_count"] == 4
    assert metrics["retrieval_unique_document_count"] == 3
    assert metrics["retrieval_document_diversity"] == 0.75
    assert metrics["retrieval_document_redundancy"] == 0.25
    assert metrics["retrieval_repeated_document_occurrence_count"] == 1
    assert metrics["successful_empty_retrieval_count"] == 0
    assert metrics["retrieval_backend_document_count"] == 4
    assert metrics["retrieval_backend_count_coverage"] == {"numerator": 2, "denominator": 2, "rate": 1.0}
    assert metrics["retrieval_locally_truncated_document_count"] == 0
    assert metrics["retrieval_hops"] == [
        {
            "retrieval_index": 0,
            "document_occurrence_count": 2,
            "unique_document_count": 2,
            "new_unique_document_count": 2,
            "duplicate_occurrence_count": 0,
            "cross_hop_repeat_count": 0,
            "novelty_rate": 1.0,
            "jaccard_with_previous": None,
            "jaccard_with_seen_before": None,
        },
        {
            "retrieval_index": 1,
            "document_occurrence_count": 2,
            "unique_document_count": 2,
            "new_unique_document_count": 1,
            "duplicate_occurrence_count": 0,
            "cross_hop_repeat_count": 1,
            "novelty_rate": 0.5,
            "jaccard_with_previous": 1 / 3,
            "jaccard_with_seen_before": 1 / 3,
        },
    ]


def test_cost_sum_overflow_is_explicit_and_json_safe():
    trace = _trace()
    steps = tuple(
        replace(
            step,
            calls=tuple(
                replace(call, cost=CostUsage(usd=1e308, source="provider")) if call.kind is CallKind.PLANNER else call
                for call in step.calls
            ),
        )
        for step in trace.steps
    )
    trace = replace(trace, steps=steps)

    derived = derive_run_metrics(trace)
    aggregate = aggregate_run_metrics([trace])

    assert derived["cost_total_usd"] is None
    assert derived["cost_total_overflow"] is True
    assert derived["cost_coverage"] == {"numerator": 2, "denominator": 2, "rate": 1.0}
    assert aggregate["cost_total_usd"] is None
    assert aggregate["cost_total_overflow"] is True
    json.dumps({"derived": derived, "aggregate": aggregate}, allow_nan=False)


def test_derive_run_metrics_reports_unobserved_query_hashes_without_guessing():
    trace = _trace()
    steps = tuple(
        replace(
            step,
            calls=tuple(
                (
                    replace(call, retrieval=replace(call.retrieval, normalized_query_sha256=None))
                    if call.retrieval is not None
                    else call
                )
                for call in step.calls
            ),
        )
        for step in trace.steps
    )

    metrics = derive_run_metrics(replace(trace, steps=steps))

    assert metrics["retrieval_query_hash_coverage"] == {"numerator": 0, "denominator": 2, "rate": 0.0}
    assert metrics["retrieval_query_occurrence_count"] == 2
    assert metrics["retrieval_observed_query_hash_count"] == 0
    assert metrics["retrieval_unique_query_count"] == 0
    assert metrics["retrieval_query_diversity"] is None


def test_failed_retrieval_is_counted_as_a_call_but_not_as_an_empty_evidence_hop():
    trace = _trace()
    failed_call = replace(_retrieval_call(2, "failed-query", ()), status=CallStatus.ERROR)
    failed_step = StepTrace(
        step_index=2,
        started_offset_ms=40.0,
        latency_ms=5.0,
        action=AgentAction.SEARCH,
        calls=(failed_call,),
        context_document_ids=("d1", "d2", "d3"),
    )

    metrics = derive_run_metrics(replace(trace, steps=(*trace.steps, failed_step)))

    assert metrics["retrieval_call_count"] == 3
    assert metrics["successful_retrieval_call_count"] == 2
    assert metrics["calls_by_status"]["error"] == 1
    assert metrics["successful_empty_retrieval_count"] == 0
    assert len(metrics["retrieval_hops"]) == 2


def test_aggregate_run_metrics_reports_denominators_and_all_percentiles():
    completed = _trace(duration_ms=100.0)
    failed = replace(
        _trace(run_id="failed", duration_ms=200.0),
        status=RunStatus.FAILED,
        termination_reason=TerminationReason.ERROR,
    )
    metrics = aggregate_run_metrics([completed, failed])

    assert metrics["schema_version_counts"] == [{"value": TRACE_SCHEMA_VERSION, "count": 2}]
    assert metrics["experiment_id_counts"] == [{"value": None, "count": 2}]
    assert metrics["variant_counts"] == [{"value": None, "count": 2}]
    assert metrics["seed_counts"] == [{"value": 7, "count": 2}]
    assert metrics["config_fingerprint_counts"] == {"config": 2}
    assert metrics["homogeneous_config_fingerprint"] is True
    assert metrics["operational_success"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert metrics["answered"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert metrics["latency_ms"]["all"] == {
        "numerator": 2,
        "denominator": 2,
        "rate": 1.0,
        "sum": 300.0,
        "sum_overflow": False,
        "mean": 150.0,
        "min": 100.0,
        "max": 200.0,
        "p50": 150.0,
        "p90": 190.0,
        "p95": 195.0,
        "p99": 199.0,
    }
    assert metrics["latency_ms"]["operational_success"]["p99"] == 100.0
    assert metrics["call_latency_ms_by_kind"]["retrieval"]["p50"] == 5.0
    assert metrics["call_latency_ms_by_kind"]["planner"]["sum"] == 40.0
    assert metrics["cost_total_usd"] == 0.008
    assert metrics["retrieval_hop_novelty"]["rate"] == 0.75
    assert metrics["retrieval_previous_hop_jaccard"]["mean"] == pytest.approx(1 / 3)
    assert metrics["retrieval_within_run_unique_query_count"] == 2
    assert metrics["workload_unique_query_count"] == 1
    assert metrics["retrieval_query_diversity"] == 0.5
    assert metrics["retrieval_within_run_unique_document_count"] == 6
    assert metrics["corpus_unique_document_count"] == 3
    assert metrics["final_context_document_count"] == 6
    assert metrics["final_context_within_run_unique_document_count"] == 6
    assert metrics["corpus_unique_final_context_document_count"] == 3
    assert metrics["retrieval_document_diversity"] == 0.75
    assert metrics["retrieval_document_redundancy"] == 0.25
    assert metrics["token_budget"]["run_count"] == 2
    assert metrics["token_budget"]["observation_coverage"] == {"numerator": 0, "denominator": 2, "rate": 0.0}


def test_aggregate_run_metrics_has_no_nan_for_empty_input():
    metrics = aggregate_run_metrics([])

    assert metrics["operational_success"] == {"numerator": 0, "denominator": 0, "rate": None}
    assert metrics["homogeneous_config_fingerprint"] is None
    assert metrics["homogeneous_filters_sha256"] is None
    assert metrics["latency_ms"]["all"]["p99"] is None
    json.dumps(metrics, allow_nan=False)


def test_aggregate_latency_sum_overflow_is_explicit_and_json_safe():
    traces = [_trace(run_id=f"run-{index}", duration_ms=1e308) for index in range(2)]

    metrics = aggregate_run_metrics(traces)
    latency = metrics["latency_ms"]["all"]

    assert latency["sum"] is None
    assert latency["sum_overflow"] is True
    assert latency["mean"] is None
    assert latency["min"] == 1e308
    assert latency["max"] == 1e308
    assert latency["p99"] == 1e308
    json.dumps(metrics, allow_nan=False)


def test_aggregate_run_metrics_keeps_unknown_filter_homogeneity_unknown():
    metrics = aggregate_run_metrics([_trace(), _trace(run_id="run-2")])

    assert metrics["filters_sha256_coverage"] == {"numerator": 0, "denominator": 2, "rate": 0.0}
    assert metrics["homogeneous_filters_sha256"] is None


def test_aggregate_run_metrics_exposes_mixed_experiment_composition():
    first = replace(
        _trace(),
        experiment_id="exp",
        variant="base",
        seed=1,
        filters_sha256="filter-a",
    )
    second = replace(
        _trace(run_id="run-2"),
        experiment_id="exp",
        variant="agentic",
        seed=2,
        config_fingerprint="config-2",
        filters_sha256="filter-b",
    )

    metrics = aggregate_run_metrics([first, second])

    assert metrics["experiment_id_counts"] == [{"value": "exp", "count": 2}]
    assert metrics["variant_counts"] == [
        {"value": "agentic", "count": 1},
        {"value": "base", "count": 1},
    ]
    assert metrics["seed_counts"] == [{"value": 1, "count": 1}, {"value": 2, "count": 1}]
    assert metrics["config_fingerprint_counts"] == {"config": 1, "config-2": 1}
    assert metrics["homogeneous_config_fingerprint"] is False
    assert metrics["filters_sha256_counts"] == [
        {"value": "filter-a", "count": 1},
        {"value": "filter-b", "count": 1},
    ]
    assert metrics["filters_sha256_coverage"] == {"numerator": 2, "denominator": 2, "rate": 1.0}
    assert metrics["homogeneous_filters_sha256"] is False
