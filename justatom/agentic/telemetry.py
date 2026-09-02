from __future__ import annotations

import asyncio
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

from justatom.agentic.contracts import TraceDeliveryPendingError
from justatom.agentic.schemas import CallKind, CallStatus, RunStatus, RunTrace, TerminationReason

_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
)
_TOKEN_BEARING_CALL_KINDS = frozenset({CallKind.PLANNER, CallKind.RERANKER, CallKind.ANSWER})
_PERCENTILES = (50, 90, 95, 99)


def _validate_required(required: bool) -> bool:
    if not isinstance(required, bool):
        raise TypeError("required must be a bool")
    return required


def _validate_trace(trace: RunTrace) -> None:
    if not isinstance(trace, RunTrace):
        raise TypeError("trace must be a RunTrace")


class TraceSinkOverloadedError(RuntimeError):
    """Raised when accepting another trace would exceed the sink backlog."""


class NullTraceSink:
    """A trace sink that intentionally discards every trace."""

    def __init__(self, *, required: bool = False) -> None:
        if required is True:
            raise ValueError("NullTraceSink cannot be required")
        self.required = _validate_required(required)
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def closed(self) -> bool:
        return self._closed

    async def write(self, trace: RunTrace) -> None:
        _validate_trace(trace)
        async with self._lock:
            if self._closed:
                raise RuntimeError("trace sink is closed")

    async def close(self) -> None:
        async with self._lock:
            self._closed = True

    async def __aenter__(self) -> NullTraceSink:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()


class InMemoryTraceSink:
    """A small concurrency-safe sink useful for tests and local experiments."""

    def __init__(self, *, required: bool = True) -> None:
        self.required = _validate_required(required)
        self._traces: list[RunTrace] = []
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def traces(self) -> list[RunTrace]:
        """Return a point-in-time copy; callers cannot mutate the sink."""

        return list(self._traces)

    def __len__(self) -> int:
        return len(self._traces)

    async def snapshot(self) -> tuple[RunTrace, ...]:
        async with self._lock:
            return tuple(self._traces)

    async def write(self, trace: RunTrace) -> None:
        _validate_trace(trace)
        async with self._lock:
            if self._closed:
                raise RuntimeError("trace sink is closed")
            self._traces.append(trace)

    async def close(self) -> None:
        async with self._lock:
            self._closed = True

    async def __aenter__(self) -> InMemoryTraceSink:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()


class JsonlTraceSink:
    """Append complete traces to an UTF-8 JSON Lines file.

    Encoding happens before the file is touched and uses ``allow_nan=False``.  A
    trace containing a non-finite number in free-form metadata therefore fails
    instead of producing non-standard JSON or a partial line.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        required: bool = True,
        max_pending_writes: int = 64,
    ) -> None:
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or pathlib.Path")
        if isinstance(max_pending_writes, bool) or not isinstance(max_pending_writes, int):
            raise TypeError("max_pending_writes must be a positive integer")
        if max_pending_writes <= 0:
            raise ValueError("max_pending_writes must be a positive integer")
        self.path = Path(path)
        self.required = _validate_required(required)
        self.max_pending_writes = max_pending_writes
        self._file: TextIO | None = None
        self._closing = False
        self._closed = False
        self._lock = asyncio.Lock()
        self._pending: set[asyncio.Task[None]] = set()
        self._close_task: asyncio.Task[None] | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    def _append(self, line: str) -> None:
        if self._file is None:
            parent_created = not self.path.parent.exists()
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if parent_created:
                self.path.parent.chmod(0o700)
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self.path, flags, 0o600)
            try:
                fchmod = getattr(os, "fchmod", None)
                if fchmod is not None:
                    fchmod(descriptor, 0o600)
                self._file = os.fdopen(descriptor, "a", encoding="utf-8", newline="\n")
            except BaseException:
                os.close(descriptor)
                raise
        self._file.write(line)
        self._file.write("\n")
        self._file.flush()

    async def _write_serialized(self, line: str) -> None:
        async with self._lock:
            if self._closed:
                raise RuntimeError("trace sink is closed")
            await asyncio.to_thread(self._append, line)

    def _write_finished(self, task: asyncio.Task[None]) -> None:
        self._pending.discard(task)
        if not task.cancelled():
            # Retrieve detached-task failures after a caller-side timeout. A
            # caller still awaiting ``write`` receives the same exception.
            task.exception()

    def _prune_finished_writes(self) -> None:
        for task in tuple(self._pending):
            if task.done():
                self._write_finished(task)

    async def write(self, trace: RunTrace) -> None:
        _validate_trace(trace)
        line = json.dumps(
            trace.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        if self._closing or self._closed:
            raise RuntimeError("trace sink is closed")
        self._prune_finished_writes()
        if len(self._pending) >= self.max_pending_writes:
            raise TraceSinkOverloadedError("trace sink pending-write capacity exceeded")
        task = asyncio.create_task(self._write_serialized(line))
        self._pending.add(task)
        task.add_done_callback(self._write_finished)
        # File I/O runs off the event loop. If the caller enforces a timeout,
        # the accepted write finishes in the background; ``close`` waits for
        # it before touching the file handle.
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            raise TraceDeliveryPendingError("trace confirmation timed out while accepted delivery remains pending") from error

    async def _drain_and_close(self) -> None:
        while self._pending:
            await asyncio.gather(*tuple(self._pending), return_exceptions=True)
        async with self._lock:
            if self._closed:
                return
            try:
                if self._file is not None:
                    self._file.close()
                    self._file = None
            finally:
                self._closed = True

    async def close(self) -> None:
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._drain_and_close())

        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                await asyncio.shield(self._close_task)
                break
            except asyncio.CancelledError as error:
                if self._close_task.cancelled():
                    if cancellation is not None:
                        raise cancellation from error
                    raise
                if cancellation is None:
                    cancellation = error
            except BaseException as cleanup_error:
                if cancellation is not None:
                    raise cancellation from cleanup_error
                raise
        if cancellation is not None:
            raise cancellation

    async def __aenter__(self) -> JsonlTraceSink:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()


# Keep the conventional initialism spelling available without making it the
# primary public name used by the rest of the package.
JSONLTraceSink = JsonlTraceSink


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def iter_jsonl_traces(path: str | Path) -> Iterator[RunTrace]:
    """Stream strict current-schema traces from a JSON Lines artifact."""

    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or pathlib.Path")
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line, parse_constant=_reject_json_constant)
                yield RunTrace.from_dict(payload)
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid agentic trace at {resolved}:{line_number}") from error


def load_jsonl_traces(path: str | Path) -> tuple[RunTrace, ...]:
    """Load a complete JSONL artifact; use :func:`iter_jsonl_traces` for large runs."""

    return tuple(iter_jsonl_traces(path))


def _coverage(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _sum_known(values: Iterable[int | float | None]) -> int | float | None:
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _finite_float_sum(values: Iterable[int | float | None]) -> tuple[float | None, bool]:
    known = [value for value in values if value is not None]
    if not known:
        return None, False
    try:
        total = math.fsum(float(value) for value in known)
    except (OverflowError, ValueError):
        return None, True
    return (total, False) if math.isfinite(total) else (None, True)


def _effective_total_tokens(call: Any) -> int | None:
    if call.tokens is None:
        return None
    if call.tokens.total_tokens is not None:
        return call.tokens.total_tokens
    if call.tokens.input_tokens is not None and call.tokens.output_tokens is not None:
        return call.tokens.input_tokens + call.tokens.output_tokens
    return None


def _token_budget_metrics(token_calls: list[Any], limit: int | None) -> dict[str, Any]:
    values = [_effective_total_tokens(call) for call in token_calls]
    total = _sum_known(values)
    observed_total = int(total) if total is not None else None
    known_count = sum(value is not None for value in values)
    complete = bool(token_calls) and known_count == len(token_calls)
    reached: bool | None = None
    if limit is not None and observed_total is not None:
        if observed_total >= limit:
            reached = True
        elif complete:
            reached = False
    return {
        "limit": limit,
        "observed_total": observed_total,
        "coverage": _coverage(known_count, len(token_calls)),
        "reached": reached,
        "overrun": (max(observed_total - limit, 0) if limit is not None and observed_total is not None and complete else None),
    }


def _all_calls(trace: RunTrace) -> list[Any]:
    return [call for step in trace.steps for call in step.calls]


def _call_execution_latency(call: Any) -> float | None:
    if call.queue_latency_ms is None:
        return None
    return call.latency_ms - call.queue_latency_ms


def _latency_summary(values: Iterable[float]) -> dict[str, int | float | None]:
    materialized = list(values)
    total, sum_overflow = _finite_float_sum(materialized)
    if not materialized:
        total = 0.0
    return {
        "count": len(materialized),
        "sum": total,
        "sum_overflow": sum_overflow,
        "mean": total / len(materialized) if materialized and total is not None else None,
    }


def _value_counts(values: Iterable[Any]) -> list[dict[str, Any]]:
    counts = Counter(values)
    ordered = sorted(
        counts.items(),
        key=lambda item: json.dumps(item[0], ensure_ascii=False, sort_keys=True),
    )
    return [{"value": value, "count": count} for value, count in ordered]


def _retrieval_hop_metrics(retrieval_calls: Iterable[Any]) -> list[dict[str, Any]]:
    ordered_calls = sorted(
        (call for call in retrieval_calls if call.status is CallStatus.OK and call.retrieval is not None),
        key=lambda call: (
            call.retrieval.retrieval_index if call.retrieval is not None else call.call_index,
            call.call_index,
        ),
    )
    seen: set[str] = set()
    previous: set[str] | None = None
    hops: list[dict[str, Any]] = []
    for call in ordered_calls:
        documents = call.retrieval.documents if call.retrieval is not None else ()
        document_ids = [document.document_id for document in documents]
        unique_ids = set(document_ids)
        new_ids = unique_ids - seen
        previous_union = unique_ids | previous if previous is not None else set()
        seen_union = unique_ids | seen
        hops.append(
            {
                "retrieval_index": call.retrieval.retrieval_index if call.retrieval is not None else call.call_index,
                "document_occurrence_count": len(document_ids),
                "unique_document_count": len(unique_ids),
                "new_unique_document_count": len(new_ids),
                "duplicate_occurrence_count": len(document_ids) - len(unique_ids),
                "cross_hop_repeat_count": len(unique_ids & seen),
                "novelty_rate": len(new_ids) / len(unique_ids) if unique_ids else None,
                "jaccard_with_previous": (
                    len(unique_ids & previous) / len(previous_union) if previous is not None and previous_union else None
                ),
                "jaccard_with_seen_before": len(unique_ids & seen) / len(seen_union) if seen and seen_union else None,
            }
        )
        seen.update(unique_ids)
        previous = unique_ids
    return hops


def derive_run_metrics(trace: RunTrace) -> dict[str, Any]:
    """Derive JSON-safe operational metrics without mutating a run trace.

    Token totals contain only explicitly reported values.  Every total is paired
    with call-level coverage, so missing provider accounting is never silently
    treated as zero.
    """

    _validate_trace(trace)
    calls = _all_calls(trace)
    calls_by_kind = Counter(call.kind.value for call in calls)
    calls_by_status = Counter(call.status.value for call in calls)
    attempts = [attempt for call in calls for attempt in call.attempts]
    attempts_by_status = Counter(attempt.status.value for attempt in attempts)
    call_error_categories = Counter(call.error.category.value for call in calls if call.error is not None)
    token_calls = [call for call in calls if call.kind in _TOKEN_BEARING_CALL_KINDS]
    retrieval_calls = [call for call in calls if call.kind is CallKind.RETRIEVAL]
    successful_retrieval_calls = [call for call in retrieval_calls if call.status is CallStatus.OK]

    token_totals: dict[str, int | None] = {}
    token_coverage: dict[str, dict[str, int | float | None]] = {}
    for field in _TOKEN_FIELDS:
        values = [getattr(call.tokens, field) if call.tokens is not None else None for call in token_calls]
        total = _sum_known(values)
        token_totals[field] = int(total) if total is not None else None
        token_coverage[field] = _coverage(sum(value is not None for value in values), len(token_calls))

    usage_observed = sum(
        call.tokens is not None and any(getattr(call.tokens, field) is not None for field in _TOKEN_FIELDS) for call in token_calls
    )
    cost_values = [call.cost.usd if call.cost is not None else None for call in token_calls]
    cost_total, cost_total_overflow = _finite_float_sum(cost_values)
    cost_sources = Counter(call.cost.source for call in token_calls if call.cost is not None)
    known_cache_hits = [call.cache_hit for call in token_calls if call.cache_hit is not None]
    time_to_first_token_values = [call.time_to_first_token_ms for call in token_calls if call.time_to_first_token_ms is not None]

    normalized_query_hashes = [
        call.retrieval.normalized_query_sha256
        for call in retrieval_calls
        if call.retrieval is not None and call.retrieval.normalized_query_sha256 is not None
    ]
    document_ids = [
        document.document_id
        for call in successful_retrieval_calls
        if call.retrieval is not None
        for document in call.retrieval.documents
    ]
    unique_query_count = len(set(normalized_query_hashes))
    unique_document_count = len(set(document_ids))
    query_occurrence_count = len(retrieval_calls)
    observed_query_hash_count = len(normalized_query_hashes)
    document_occurrence_count = len(document_ids)
    call_latency_ms_by_kind = {
        kind.value: _latency_summary(call.latency_ms for call in calls if call.kind is kind) for kind in CallKind
    }
    call_queue_latency_ms_by_kind = {
        kind.value: _latency_summary(
            call.queue_latency_ms for call in calls if call.kind is kind and call.queue_latency_ms is not None
        )
        for kind in CallKind
    }
    call_execution_latency_ms_by_kind = {
        kind.value: _latency_summary(
            execution_latency
            for call in calls
            if call.kind is kind
            for execution_latency in (_call_execution_latency(call),)
            if execution_latency is not None
        )
        for kind in CallKind
    }
    call_queue_latency_coverage_by_kind = {
        kind.value: _coverage(
            sum(call.queue_latency_ms is not None for call in calls if call.kind is kind),
            sum(call.kind is kind for call in calls),
        )
        for kind in CallKind
    }
    attempt_count = sum(len(call.attempts) for call in calls)
    retrieval_hops = _retrieval_hop_metrics(successful_retrieval_calls)
    backend_document_counts = [
        call.retrieval.backend_document_count for call in successful_retrieval_calls if call.retrieval is not None
    ]
    truncated_document_count = sum(
        call.retrieval.truncated_document_count for call in successful_retrieval_calls if call.retrieval is not None
    )
    retrieval_requested_slot_count = sum(call.retrieval.top_k_requested for call in retrieval_calls if call.retrieval is not None)
    retrieval_slot_budget = trace.limits.max_retrieval_calls * trace.limits.top_k
    final_context_ids = set(trace.final_context_document_ids)
    cited_ids = trace.final_cited_document_ids
    cited_in_context_count = sum(document_id in final_context_ids for document_id in cited_ids)

    return {
        "run_id": trace.run_id,
        "schema_version": trace.schema_version,
        "experiment_id": trace.experiment_id,
        "variant": trace.variant,
        "seed": trace.seed,
        "objective": trace.objective.value,
        "config_fingerprint": trace.config_fingerprint,
        "planner_config_fingerprint": trace.planner_config_fingerprint,
        "retrieval_config_fingerprint": trace.retrieval_config_fingerprint,
        "filters_sha256": trace.filters_sha256,
        "status": trace.status.value,
        "termination_reason": trace.termination_reason.value,
        "operational_success": trace.status is RunStatus.COMPLETED,
        "answered": trace.status is RunStatus.COMPLETED and trace.termination_reason is TerminationReason.ANSWERED,
        "answered_with_citations": (
            trace.status is RunStatus.COMPLETED and trace.termination_reason is TerminationReason.ANSWERED and bool(cited_ids)
        ),
        "agent_stopped": trace.status is RunStatus.COMPLETED and trace.termination_reason is TerminationReason.AGENT_STOP,
        "duration_ms": trace.duration_ms,
        "queue_latency_ms": trace.queue_latency_ms,
        "execution_ms": trace.execution_ms,
        "budget_dimension": trace.budget_dimension,
        "error_category": trace.error.category.value if trace.error is not None else None,
        "error_code": trace.error.code if trace.error is not None else None,
        "step_count": len(trace.steps),
        "step_latency_ms": _latency_summary(step.latency_ms for step in trace.steps),
        "call_count": len(calls),
        "calls_by_kind": {kind.value: calls_by_kind[kind.value] for kind in CallKind},
        "calls_by_status": {status.value: calls_by_status[status.value] for status in CallStatus},
        "call_error_category_counts": dict(sorted(call_error_categories.items())),
        "call_latency_ms_by_kind": call_latency_ms_by_kind,
        "call_queue_latency_ms_by_kind": call_queue_latency_ms_by_kind,
        "call_execution_latency_ms_by_kind": call_execution_latency_ms_by_kind,
        "call_queue_latency_coverage_by_kind": call_queue_latency_coverage_by_kind,
        "retrieval_total_ms": call_latency_ms_by_kind[CallKind.RETRIEVAL.value]["sum"],
        "planner_total_ms": call_latency_ms_by_kind[CallKind.PLANNER.value]["sum"],
        "attempt_count": attempt_count,
        "attempts_by_status": {status.value: attempts_by_status[status.value] for status in CallStatus},
        "attempt_latency_ms": _latency_summary(attempt.latency_ms for attempt in attempts),
        "retry_count": sum(max(len(call.attempts) - 1, 0) for call in calls),
        "llm_call_count": len(token_calls),
        "retrieval_call_count": len(retrieval_calls),
        "successful_retrieval_call_count": len(successful_retrieval_calls),
        "retrieval_requested_slot_count": retrieval_requested_slot_count,
        "retrieval_slot_budget": retrieval_slot_budget,
        "retrieval_slot_budget_utilization": _coverage(retrieval_requested_slot_count, retrieval_slot_budget),
        "token_totals": token_totals,
        "token_usage_coverage": _coverage(usage_observed, len(token_calls)),
        "token_coverage": token_coverage,
        "token_budget": _token_budget_metrics(token_calls, trace.limits.max_tokens),
        "cost_total_usd": float(cost_total) if cost_total is not None else None,
        "cost_total_overflow": cost_total_overflow,
        "cost_coverage": _coverage(sum(value is not None for value in cost_values), len(token_calls)),
        "cost_sources": dict(sorted(cost_sources.items())),
        "cache_hit_count": sum(known_cache_hits),
        "cache_hit_coverage": _coverage(len(known_cache_hits), len(token_calls)),
        "cache_hit_rate": sum(known_cache_hits) / len(known_cache_hits) if known_cache_hits else None,
        "time_to_first_token_ms": _latency_summary(time_to_first_token_values),
        "time_to_first_token_coverage": _coverage(len(time_to_first_token_values), len(token_calls)),
        "retrieval_query_occurrence_count": query_occurrence_count,
        "retrieval_observed_query_hash_count": observed_query_hash_count,
        "retrieval_unique_query_count": unique_query_count,
        "retrieval_query_hash_coverage": _coverage(observed_query_hash_count, query_occurrence_count),
        "retrieval_query_diversity": unique_query_count / observed_query_hash_count if observed_query_hash_count else None,
        "retrieval_document_occurrence_count": document_occurrence_count,
        "retrieval_unique_document_count": unique_document_count,
        "retrieval_document_diversity": unique_document_count / document_occurrence_count if document_occurrence_count else None,
        "retrieval_document_redundancy": (
            1.0 - unique_document_count / document_occurrence_count if document_occurrence_count else None
        ),
        "retrieval_repeated_document_occurrence_count": document_occurrence_count - unique_document_count,
        "successful_empty_retrieval_count": sum(hop["document_occurrence_count"] == 0 for hop in retrieval_hops),
        "retrieval_backend_document_count": _sum_known(backend_document_counts),
        "retrieval_backend_count_coverage": _coverage(
            sum(value is not None for value in backend_document_counts),
            len(successful_retrieval_calls),
        ),
        "retrieval_locally_truncated_document_count": truncated_document_count,
        "retrieval_hops": retrieval_hops,
        "final_context_document_count": len(trace.final_context_document_ids),
        "final_context_unique_document_count": len(set(trace.final_context_document_ids)),
        "final_context_chars": trace.final_context_chars,
        "final_context_document_budget": _coverage(
            len(trace.final_context_document_ids),
            trace.limits.max_context_documents,
        ),
        "final_context_char_budget": _coverage(trace.final_context_chars, trace.limits.max_context_chars),
        "citation_count": len(cited_ids),
        "unique_citation_count": len(set(cited_ids)),
        "citations_in_context_count": cited_in_context_count,
        "citations_out_of_context_count": len(cited_ids) - cited_in_context_count,
        "citation_context_coverage": _coverage(cited_in_context_count, len(cited_ids)),
    }


def _percentile(sorted_values: list[float], percentile: int) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile / 100
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    lower = sorted_values[lower_index]
    return lower + (sorted_values[upper_index] - lower) * weight


def _distribution(values: Iterable[float], *, denominator: int) -> dict[str, int | float | None]:
    ordered = sorted(values)
    total, sum_overflow = _finite_float_sum(ordered)
    if not ordered:
        total = 0.0
    result: dict[str, int | float | None] = {
        "numerator": len(ordered),
        "denominator": denominator,
        "rate": len(ordered) / denominator if denominator else None,
        "sum": total,
        "sum_overflow": sum_overflow,
        "mean": total / len(ordered) if ordered and total is not None else None,
        "min": ordered[0] if ordered else None,
        "max": ordered[-1] if ordered else None,
    }
    result.update({f"p{percentile}": _percentile(ordered, percentile) for percentile in _PERCENTILES})
    return result


def _call_latency_distribution(values: Iterable[float]) -> dict[str, int | float | None]:
    materialized = list(values)
    result = _distribution(materialized, denominator=len(materialized))
    result.update(_latency_summary(materialized))
    return result


def aggregate_run_metrics(traces: Iterable[RunTrace]) -> dict[str, Any]:
    """Aggregate traces with explicit denominators and deterministic quantiles."""

    materialized = list(traces)
    for trace in materialized:
        _validate_trace(trace)

    derived = [derive_run_metrics(trace) for trace in materialized]
    run_count = len(materialized)
    operational_success_count = sum(metric["operational_success"] for metric in derived)
    answered_count = sum(metric["answered"] for metric in derived)
    answered_with_citations_count = sum(metric["answered_with_citations"] for metric in derived)
    agent_stopped_count = sum(metric["agent_stopped"] for metric in derived)
    status_counts = Counter(metric["status"] for metric in derived)
    termination_counts = Counter(metric["termination_reason"] for metric in derived)
    token_budgets = [metric["token_budget"] for metric in derived if metric["token_budget"]["limit"] is not None]
    known_token_budget_reached = [budget["reached"] for budget in token_budgets if budget["reached"] is not None]
    known_token_budget_overruns = [budget["overrun"] for budget in token_budgets if budget["overrun"] is not None]

    all_calls = [call for trace in materialized for call in _all_calls(trace)]
    calls_by_kind = Counter(call.kind.value for call in all_calls)
    calls_by_status = Counter(call.status.value for call in all_calls)
    all_attempts = [attempt for call in all_calls for attempt in call.attempts]
    attempts_by_status = Counter(attempt.status.value for attempt in all_attempts)
    call_error_categories = Counter(call.error.category.value for call in all_calls if call.error is not None)
    token_calls = [call for call in all_calls if call.kind in _TOKEN_BEARING_CALL_KINDS]
    retrieval_calls = [call for call in all_calls if call.kind is CallKind.RETRIEVAL]
    successful_retrieval_calls = [call for call in retrieval_calls if call.status is CallStatus.OK]

    token_totals: dict[str, int | None] = {}
    token_coverage: dict[str, dict[str, int | float | None]] = {}
    for field in _TOKEN_FIELDS:
        values = [getattr(call.tokens, field) if call.tokens is not None else None for call in token_calls]
        total = _sum_known(values)
        token_totals[field] = int(total) if total is not None else None
        token_coverage[field] = _coverage(sum(value is not None for value in values), len(token_calls))

    usage_observed = sum(
        call.tokens is not None and any(getattr(call.tokens, field) is not None for field in _TOKEN_FIELDS) for call in token_calls
    )
    cost_values = [call.cost.usd if call.cost is not None else None for call in token_calls]
    cost_total, cost_total_overflow = _finite_float_sum(cost_values)
    cost_sources = Counter(call.cost.source for call in token_calls if call.cost is not None)
    known_cache_hits = [call.cache_hit for call in token_calls if call.cache_hit is not None]
    time_to_first_token_values = [call.time_to_first_token_ms for call in token_calls if call.time_to_first_token_ms is not None]
    normalized_query_hashes = [
        call.retrieval.normalized_query_sha256
        for call in retrieval_calls
        if call.retrieval is not None and call.retrieval.normalized_query_sha256 is not None
    ]
    document_ids = [
        document.document_id
        for call in successful_retrieval_calls
        if call.retrieval is not None
        for document in call.retrieval.documents
    ]
    query_occurrence_count = len(retrieval_calls)
    observed_query_hash_count = len(normalized_query_hashes)
    document_occurrence_count = len(document_ids)
    workload_unique_query_count = len(set(normalized_query_hashes))
    corpus_unique_document_count = len(set(document_ids))
    within_run_unique_query_count = sum(metric["retrieval_unique_query_count"] for metric in derived)
    within_run_unique_document_count = sum(metric["retrieval_unique_document_count"] for metric in derived)
    repeated_document_occurrence_count = sum(metric["retrieval_repeated_document_occurrence_count"] for metric in derived)
    call_latency_ms_by_kind = {
        kind.value: _call_latency_distribution(call.latency_ms for call in all_calls if call.kind is kind) for kind in CallKind
    }
    call_queue_latency_ms_by_kind = {
        kind.value: _distribution(
            (call.queue_latency_ms for call in all_calls if call.kind is kind and call.queue_latency_ms is not None),
            denominator=sum(call.kind is kind for call in all_calls),
        )
        for kind in CallKind
    }
    call_execution_latency_ms_by_kind = {
        kind.value: _distribution(
            (
                execution_latency
                for call in all_calls
                if call.kind is kind
                for execution_latency in (_call_execution_latency(call),)
                if execution_latency is not None
            ),
            denominator=sum(call.kind is kind for call in all_calls),
        )
        for kind in CallKind
    }
    attempt_count = sum(len(call.attempts) for call in all_calls)
    all_cited_ids = [document_id for trace in materialized for document_id in trace.final_cited_document_ids]
    cited_in_context_count = sum(
        document_id in set(trace.final_context_document_ids)
        for trace in materialized
        for document_id in trace.final_cited_document_ids
    )
    error_categories = Counter(trace.error.category.value for trace in materialized if trace.error is not None)
    budget_dimensions = Counter(trace.budget_dimension for trace in materialized if trace.budget_dimension is not None)
    retrieval_hops = [hop for metric in derived for hop in metric["retrieval_hops"]]
    hop_novelty_values = [hop["novelty_rate"] for hop in retrieval_hops if hop["novelty_rate"] is not None]
    previous_jaccard_values = [hop["jaccard_with_previous"] for hop in retrieval_hops if hop["jaccard_with_previous"] is not None]
    hop_new_unique_total = sum(hop["new_unique_document_count"] for hop in retrieval_hops)
    hop_unique_total = sum(hop["unique_document_count"] for hop in retrieval_hops)
    hop_pair_count = sum(max(len(metric["retrieval_hops"]) - 1, 0) for metric in derived)
    config_fingerprint_counts = Counter(trace.config_fingerprint for trace in materialized)
    planner_fingerprint_counts = Counter(trace.planner_config_fingerprint for trace in materialized)
    retrieval_fingerprint_counts = Counter(trace.retrieval_config_fingerprint for trace in materialized)
    filter_hashes = [trace.filters_sha256 for trace in materialized]
    known_filter_hash_count = sum(value is not None for value in filter_hashes)

    return {
        "run_count": run_count,
        "schema_version_counts": _value_counts(trace.schema_version for trace in materialized),
        "experiment_id_counts": _value_counts(trace.experiment_id for trace in materialized),
        "variant_counts": _value_counts(trace.variant for trace in materialized),
        "seed_counts": _value_counts(trace.seed for trace in materialized),
        "objective_counts": _value_counts(trace.objective.value for trace in materialized),
        "config_fingerprint_counts": dict(sorted(config_fingerprint_counts.items())),
        "planner_config_fingerprint_counts": dict(sorted(planner_fingerprint_counts.items())),
        "retrieval_config_fingerprint_counts": dict(sorted(retrieval_fingerprint_counts.items())),
        "filters_sha256_counts": _value_counts(filter_hashes),
        "filters_sha256_coverage": _coverage(known_filter_hash_count, run_count),
        "homogeneous_config_fingerprint": len(config_fingerprint_counts) <= 1 if run_count else None,
        "homogeneous_planner_config_fingerprint": len(planner_fingerprint_counts) <= 1 if run_count else None,
        "homogeneous_retrieval_config_fingerprint": len(retrieval_fingerprint_counts) <= 1 if run_count else None,
        "homogeneous_objective": len({trace.objective for trace in materialized}) <= 1 if run_count else None,
        "homogeneous_filters_sha256": (
            len(set(filter_hashes)) <= 1 if run_count and known_filter_hash_count == run_count else None
        ),
        "operational_success": _coverage(operational_success_count, run_count),
        "answered": _coverage(answered_count, run_count),
        "answered_with_citations": _coverage(answered_with_citations_count, answered_count),
        "agent_stopped": _coverage(agent_stopped_count, run_count),
        "status_counts": {status.value: status_counts[status.value] for status in RunStatus},
        "termination_reason_counts": {reason.value: termination_counts[reason.value] for reason in TerminationReason},
        "latency_ms": {
            "all": _distribution((trace.duration_ms for trace in materialized), denominator=run_count),
            "operational_success": _distribution(
                (trace.duration_ms for trace in materialized if trace.status is RunStatus.COMPLETED),
                denominator=run_count,
            ),
            "queue": _distribution((trace.queue_latency_ms for trace in materialized), denominator=run_count),
            "execution": _distribution((trace.execution_ms for trace in materialized), denominator=run_count),
        },
        "budget_dimension_counts": dict(sorted(budget_dimensions.items())),
        "run_error_category_counts": dict(sorted(error_categories.items())),
        "step_count": sum(len(trace.steps) for trace in materialized),
        "step_latency_ms": _call_latency_distribution(step.latency_ms for trace in materialized for step in trace.steps),
        "call_count": len(all_calls),
        "calls_by_kind": {kind.value: calls_by_kind[kind.value] for kind in CallKind},
        "calls_by_status": {status.value: calls_by_status[status.value] for status in CallStatus},
        "call_error_category_counts": dict(sorted(call_error_categories.items())),
        "call_latency_ms_by_kind": call_latency_ms_by_kind,
        "call_queue_latency_ms_by_kind": call_queue_latency_ms_by_kind,
        "call_execution_latency_ms_by_kind": call_execution_latency_ms_by_kind,
        "retrieval_total_ms": call_latency_ms_by_kind[CallKind.RETRIEVAL.value]["sum"],
        "planner_total_ms": call_latency_ms_by_kind[CallKind.PLANNER.value]["sum"],
        "attempt_count": attempt_count,
        "attempts_by_status": {status.value: attempts_by_status[status.value] for status in CallStatus},
        "attempt_latency_ms": _call_latency_distribution(attempt.latency_ms for attempt in all_attempts),
        "retry_count": sum(max(len(call.attempts) - 1, 0) for call in all_calls),
        "llm_call_count": len(token_calls),
        "retrieval_call_count": len(retrieval_calls),
        "successful_retrieval_call_count": len(successful_retrieval_calls),
        "retrieval_requested_slot_count": sum(metric["retrieval_requested_slot_count"] for metric in derived),
        "retrieval_slot_budget": sum(metric["retrieval_slot_budget"] for metric in derived),
        "retrieval_slot_budget_utilization": _coverage(
            sum(metric["retrieval_requested_slot_count"] for metric in derived),
            sum(metric["retrieval_slot_budget"] for metric in derived),
        ),
        "token_totals": token_totals,
        "token_usage_coverage": _coverage(usage_observed, len(token_calls)),
        "token_coverage": token_coverage,
        "token_budget": {
            "run_count": len(token_budgets),
            "observation_coverage": _coverage(len(known_token_budget_reached), len(token_budgets)),
            "reached": _coverage(sum(known_token_budget_reached), len(known_token_budget_reached)),
            "known_overrun_total": _sum_known(known_token_budget_overruns),
        },
        "cost_total_usd": float(cost_total) if cost_total is not None else None,
        "cost_total_overflow": cost_total_overflow,
        "cost_coverage": _coverage(sum(value is not None for value in cost_values), len(token_calls)),
        "cost_sources": dict(sorted(cost_sources.items())),
        "cache_hit_count": sum(known_cache_hits),
        "cache_hit_coverage": _coverage(len(known_cache_hits), len(token_calls)),
        "cache_hit_rate": sum(known_cache_hits) / len(known_cache_hits) if known_cache_hits else None,
        "time_to_first_token_ms": _distribution(
            time_to_first_token_values,
            denominator=len(token_calls),
        ),
        "retrieval_query_occurrence_count": query_occurrence_count,
        "retrieval_observed_query_hash_count": observed_query_hash_count,
        "retrieval_within_run_unique_query_count": within_run_unique_query_count,
        "workload_unique_query_count": workload_unique_query_count,
        "retrieval_query_hash_coverage": _coverage(observed_query_hash_count, query_occurrence_count),
        "retrieval_query_diversity": (
            within_run_unique_query_count / observed_query_hash_count if observed_query_hash_count else None
        ),
        "retrieval_document_occurrence_count": document_occurrence_count,
        "retrieval_within_run_unique_document_count": within_run_unique_document_count,
        "corpus_unique_document_count": corpus_unique_document_count,
        "retrieval_document_diversity": (
            within_run_unique_document_count / document_occurrence_count if document_occurrence_count else None
        ),
        "retrieval_document_redundancy": (
            repeated_document_occurrence_count / document_occurrence_count if document_occurrence_count else None
        ),
        "retrieval_repeated_document_occurrence_count": repeated_document_occurrence_count,
        "successful_empty_retrieval_count": sum(metric["successful_empty_retrieval_count"] for metric in derived),
        "retrieval_backend_document_count": _sum_known(metric["retrieval_backend_document_count"] for metric in derived),
        "retrieval_backend_count_coverage": _coverage(
            sum(metric["retrieval_backend_count_coverage"]["numerator"] for metric in derived),
            len(successful_retrieval_calls),
        ),
        "retrieval_locally_truncated_document_count": sum(
            metric["retrieval_locally_truncated_document_count"] for metric in derived
        ),
        "retrieval_hop_novelty": {
            "new_unique_document_count": hop_new_unique_total,
            "unique_document_count": hop_unique_total,
            "rate": hop_new_unique_total / hop_unique_total if hop_unique_total else None,
            "distribution": _distribution(hop_novelty_values, denominator=len(retrieval_hops)),
        },
        "retrieval_previous_hop_jaccard": _distribution(
            previous_jaccard_values,
            denominator=hop_pair_count,
        ),
        "final_context_document_count": sum(len(trace.final_context_document_ids) for trace in materialized),
        "final_context_within_run_unique_document_count": sum(len(set(trace.final_context_document_ids)) for trace in materialized),
        "corpus_unique_final_context_document_count": len(
            {document_id for trace in materialized for document_id in trace.final_context_document_ids}
        ),
        "final_context_chars": {
            "total": sum(trace.final_context_chars for trace in materialized),
            "distribution": _distribution(
                (float(trace.final_context_chars) for trace in materialized),
                denominator=run_count,
            ),
        },
        "final_context_document_budget": _coverage(
            sum(len(trace.final_context_document_ids) for trace in materialized),
            sum(trace.limits.max_context_documents for trace in materialized),
        ),
        "final_context_char_budget": _coverage(
            sum(trace.final_context_chars for trace in materialized),
            sum(trace.limits.max_context_chars for trace in materialized),
        ),
        "citation_count": len(all_cited_ids),
        "within_run_unique_citation_count": sum(len(set(trace.final_cited_document_ids)) for trace in materialized),
        "corpus_unique_cited_document_count": len(set(all_cited_ids)),
        "citations_in_context_count": cited_in_context_count,
        "citations_out_of_context_count": len(all_cited_ids) - cited_in_context_count,
        "citation_context_coverage": _coverage(cited_in_context_count, len(all_cited_ids)),
    }


__all__ = [
    "InMemoryTraceSink",
    "JSONLTraceSink",
    "JsonlTraceSink",
    "NullTraceSink",
    "TraceSinkOverloadedError",
    "aggregate_run_metrics",
    "derive_run_metrics",
    "iter_jsonl_traces",
    "load_jsonl_traces",
]
