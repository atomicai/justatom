from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from justatom.agentic.contracts import AgentRetriever, ChatBackend, TraceDeliveryPendingError, TracePersistenceError, TraceSink
from justatom.agentic.schemas import (
    TRACE_SCHEMA_VERSION,
    AgentAction,
    AttemptTrace,
    CallKind,
    CallStatus,
    CallTrace,
    DecisionTrace,
    DocumentTrace,
    ErrorCategory,
    ErrorTrace,
    EvidenceDocument,
    PlannerDecision,
    PlannerReply,
    PlannerRequest,
    RetrievalPayload,
    RunLimits,
    RunStatus,
    RunTrace,
    SearchObservation,
    StepTrace,
    TerminationReason,
    TextCapturePolicy,
    TokenUsage,
    normalize_query,
    normalized_query_sha256,
    sha256_text,
)

_ROOT_CONFIG_KEYS = {
    "enabled",
    "max_steps",
    "max_retrieval_calls",
    "max_llm_calls",
    "max_tokens",
    "total_timeout_seconds",
    "retrieval_timeout_seconds",
    "planner_timeout_seconds",
    "max_concurrency",
    "max_queued_runs",
    "top_k",
    "max_request_bytes",
    "max_query_chars",
    "max_answer_chars",
    "max_reason_chars",
    "max_identifier_chars",
    "max_filter_chars",
    "max_metadata_chars",
    "max_document_chars",
    "max_context_chars",
    "max_context_documents",
    "no_progress_limit",
    "planner",
    "trace",
    "experiment_id",
    "variant",
    "seed",
    "metadata",
}
_PLANNER_CONFIG_KEYS = {
    "backend",
    "base_url",
    "model",
    "api_key",
    "timeout_seconds",
    "temperature",
    "max_tokens",
    "max_response_bytes",
    "seed",
    "system_prompt",
}
_TRACE_CONFIG_KEYS = {
    "path",
    "capture_text",
    "required",
    "timeout_seconds",
    "max_pending_writes",
}
_LOGGER = logging.getLogger(__name__)


class AgenticConfigurationError(ValueError):
    pass


class AgenticCapacityError(RuntimeError):
    """Raised before admission when the bounded run queue is full."""

    retry_after_seconds = 1


@dataclass(frozen=True, slots=True)
class AgenticRuntimeConfig:
    max_steps: int = 4
    max_retrieval_calls: int = 3
    max_llm_calls: int = 3
    max_tokens: int | None = None
    total_timeout_seconds: float = 60.0
    retrieval_timeout_seconds: float = 30.0
    planner_timeout_seconds: float = 30.0
    trace_timeout_seconds: float = 5.0
    max_concurrency: int = 2
    max_queued_runs: int = 8
    top_k: int = 10
    max_request_bytes: int = 65_536
    max_query_chars: int = 512
    max_answer_chars: int = 16_000
    max_reason_chars: int = 4_000
    max_identifier_chars: int = 512
    max_filter_chars: int = 16_000
    max_metadata_chars: int = 8_000
    max_document_chars: int = 2_000
    max_context_chars: int = 24_000
    max_context_documents: int = 50
    no_progress_limit: int = 2
    capture_text: TextCapturePolicy = TextCapturePolicy.HASH
    trace_required: bool = True
    experiment_id: str | None = None
    variant: str | None = None
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "max_steps",
            "max_retrieval_calls",
            "max_llm_calls",
            "max_concurrency",
            "top_k",
            "max_request_bytes",
            "max_query_chars",
            "max_answer_chars",
            "max_reason_chars",
            "max_identifier_chars",
            "max_filter_chars",
            "max_metadata_chars",
            "max_document_chars",
            "max_context_chars",
            "max_context_documents",
            "no_progress_limit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise AgenticConfigurationError(f"{name} must be a positive integer")
        if isinstance(self.max_queued_runs, bool) or not isinstance(self.max_queued_runs, int) or self.max_queued_runs < 0:
            raise AgenticConfigurationError("max_queued_runs must be a non-negative integer")
        if self.max_steps < 2:
            raise AgenticConfigurationError("max_steps must be at least 2 (initial retrieval plus a planner decision)")
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens <= 0
        ):
            raise AgenticConfigurationError("max_tokens must be a positive integer or null")
        for name in (
            "total_timeout_seconds",
            "retrieval_timeout_seconds",
            "planner_timeout_seconds",
            "trace_timeout_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise AgenticConfigurationError(f"{name} must be a positive finite number")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise AgenticConfigurationError("seed must be an integer or null")
        if not isinstance(self.trace_required, bool):
            raise AgenticConfigurationError("trace_required must be a boolean")
        if not isinstance(self.capture_text, TextCapturePolicy):
            try:
                object.__setattr__(self, "capture_text", TextCapturePolicy(self.capture_text))
            except (TypeError, ValueError) as error:
                raise AgenticConfigurationError("capture_text must be none, hash, or full") from error
        for name in ("experiment_id", "variant"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise AgenticConfigurationError(f"{name} must be a non-empty string or null")
            value = value.strip()
            if len(value) > self.max_identifier_chars:
                raise AgenticConfigurationError(f"{name} exceeds max_identifier_chars={self.max_identifier_chars}")
            object.__setattr__(self, name, value)
        if not isinstance(self.metadata, Mapping):
            raise AgenticConfigurationError("metadata must be a mapping")
        normalized_metadata, canonical_metadata = _canonical_json_mapping(self.metadata, "metadata")
        if len(canonical_metadata) > self.max_metadata_chars:
            raise AgenticConfigurationError(f"metadata exceeds max_metadata_chars={self.max_metadata_chars}")
        object.__setattr__(self, "metadata", normalized_metadata)

    @property
    def limits(self) -> RunLimits:
        return RunLimits(
            max_steps=self.max_steps,
            max_retrieval_calls=self.max_retrieval_calls,
            max_llm_calls=self.max_llm_calls,
            max_tokens=self.max_tokens,
            max_duration_ms=self.total_timeout_seconds * 1_000.0,
        )


@dataclass(frozen=True, slots=True)
class AgenticRunResult:
    run_id: str
    answer: str | None
    evidence: tuple[EvidenceDocument, ...]
    trace: RunTrace
    metrics: dict[str, Any]


@dataclass(slots=True)
class _StepBuilder:
    step_index: int
    start_ns: int
    action: AgentAction = AgentAction.STOP
    calls: list[CallTrace] = field(default_factory=list)
    decision: DecisionTrace | None = None
    error: ErrorTrace | None = None
    finished: bool = False


@dataclass(slots=True)
class _RunState:
    run_id: str
    request_id: str | None
    question: str
    start_ns: int
    started_at: str
    queue_latency_ms: float = 0.0
    execution_start_ns: int | None = None
    call_index: int = 0
    retrieval_index: int = 0
    llm_calls: int = 0
    observed_total_tokens: int = 0
    steps: list[StepTrace] = field(default_factory=list)
    observations: list[SearchObservation] = field(default_factory=list)
    context: dict[str, EvidenceDocument] = field(default_factory=dict)
    context_chars: int = 0
    seen_queries: set[str] = field(default_factory=set)
    no_progress_streak: int = 0
    answer: str | None = None
    final_cited_document_ids: tuple[str, ...] = ()
    status: RunStatus = RunStatus.COMPLETED
    termination_reason: TerminationReason = TerminationReason.ERROR
    budget_dimension: str | None = None
    error: ErrorTrace | None = None


@dataclass(slots=True)
class _ComponentTiming:
    queue_start_ns: int
    execution_start_ns: int | None = None


class _ExecutionFailure(Exception):
    def __init__(self, reason: TerminationReason, error: ErrorTrace, *, timed_out: bool = False) -> None:
        super().__init__(error.code)
        self.reason = reason
        self.error = error
        self.timed_out = timed_out


class _InvalidPlannerReply(ValueError):
    pass


class _TraceCapacityExhausted(RuntimeError):
    pass


class _TraceDeliveryPending(TimeoutError):
    pass


class AgenticRAGRuntime:
    """A bounded, observable search/answer loop over an existing retriever.

    The runtime owns the chat backend and trace sink. It deliberately does not
    own or close the retriever because a retrieval service may share it with
    ordinary one-shot search endpoints.
    """

    def __init__(
        self,
        retriever: AgentRetriever,
        chat_backend: ChatBackend,
        *,
        config: AgenticRuntimeConfig | None = None,
        trace_sink: TraceSink | None = None,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
        utc_now: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.config = config or AgenticRuntimeConfig()
        if trace_sink is None:
            if self.config.trace_required:
                raise AgenticConfigurationError("trace_sink is required when trace_required is true")
            from justatom.agentic.telemetry import NullTraceSink

            trace_sink = NullTraceSink()
        sink_required = getattr(trace_sink, "required", None)
        if sink_required is not None:
            if not isinstance(sink_required, bool):
                raise AgenticConfigurationError("trace_sink.required must be a boolean")
            if sink_required is not self.config.trace_required:
                raise AgenticConfigurationError("trace_sink.required must match trace_required")
        self.retriever = retriever
        self.chat_backend = chat_backend
        self.trace_sink = trace_sink
        self._base_metadata, _ = _canonical_json_mapping(self.config.metadata, "metadata")
        self._clock = monotonic_ns
        self._utc_now = utc_now or (lambda: datetime.now(timezone.utc))
        self._id_factory = id_factory or (lambda: str(uuid.uuid4()))
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        self._component_semaphore = asyncio.Semaphore(self.config.max_concurrency)
        self._inflight_operations: set[asyncio.Future[Any]] = set()
        self._operations_idle = asyncio.Event()
        self._operations_idle.set()
        self._available_trace_slots = self.config.max_concurrency
        self._inflight_trace_writes: set[asyncio.Future[Any]] = set()
        self._trace_writes_idle = asyncio.Event()
        self._trace_writes_idle.set()
        self._lifecycle_lock = asyncio.Lock()
        self._idle = asyncio.Event()
        self._idle.set()
        self._accepted_runs = 0
        self._rejected_runs = 0
        self._closing = False
        self._closed = False
        self._finalization_task: asyncio.Task[None] | None = None
        self._planner_config_fingerprint = _planner_config_fingerprint(self.chat_backend)
        self._retrieval_config_fingerprint = _retrieval_config_fingerprint(self.retriever)
        self._config_fingerprint = self._fingerprint_config()

    @property
    def config_fingerprint(self) -> str:
        return self._config_fingerprint

    @property
    def planner_config_fingerprint(self) -> str:
        return self._planner_config_fingerprint

    @property
    def retrieval_config_fingerprint(self) -> str:
        return self._retrieval_config_fingerprint

    async def __aenter__(self) -> AgenticRAGRuntime:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def run(
        self,
        question: str,
        *,
        request_id: str | None = None,
        filters: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgenticRunResult:
        question = _validate_question(question, self.config.max_query_chars)
        request_id = _validate_optional_identifier(request_id, self.config.max_identifier_chars, "request_id")
        if filters is not None and not isinstance(filters, Mapping):
            raise ValueError("filters must be a mapping or null")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping or null")
        run_metadata = dict(self._base_metadata)
        if metadata is not None:
            run_metadata.update(dict(metadata))
        run_metadata, canonical_metadata = _canonical_json_mapping(run_metadata, "metadata")
        if len(canonical_metadata) > self.config.max_metadata_chars:
            raise ValueError(f"metadata exceeds max_metadata_chars={self.config.max_metadata_chars}")
        run_filters: dict[str, Any] | None = None
        if filters is not None:
            run_filters, canonical_filters = _canonical_json_mapping(filters, "filters")
            if len(canonical_filters) > self.config.max_filter_chars:
                raise ValueError(f"filters exceeds max_filter_chars={self.config.max_filter_chars}")

        state = _RunState(
            run_id=self._id_factory(),
            request_id=request_id,
            question=question,
            start_ns=self._clock(),
            started_at=_utc_iso(self._utc_now()),
        )
        await self._begin_run()
        acquired = False
        cancellation: asyncio.CancelledError | None = None
        trace: RunTrace | None = None
        try:
            elapsed_seconds = self._elapsed_ms(state.start_ns) / 1_000.0
            queue_timeout = max(self.config.total_timeout_seconds - elapsed_seconds, 0.0)
            if queue_timeout <= 0:
                raise asyncio.TimeoutError
            await asyncio.wait_for(self._semaphore.acquire(), timeout=queue_timeout)
            acquired = True
            state.execution_start_ns = self._clock()
            state.queue_latency_ms = self._duration_ms(state.start_ns, state.execution_start_ns)
            remaining = self.config.total_timeout_seconds - self._elapsed_ms(state.start_ns) / 1_000.0
            if remaining <= 0:
                raise asyncio.TimeoutError
            await asyncio.wait_for(self._execute(state, filters=run_filters), timeout=remaining)
        except _ExecutionFailure as failure:
            state.status = RunStatus.TIMED_OUT if failure.timed_out else RunStatus.FAILED
            state.termination_reason = failure.reason
            state.error = failure.error
        except asyncio.TimeoutError as error:
            if state.execution_start_ns is None:
                state.queue_latency_ms = self._elapsed_ms(state.start_ns)
            state.status = RunStatus.TIMED_OUT
            state.termination_reason = TerminationReason.MAX_DURATION
            state.budget_dimension = "max_duration"
            state.error = _error_trace("runtime", ErrorCategory.TIMEOUT, "total_deadline_exceeded", error, retryable=True)
        except asyncio.CancelledError as error:
            if state.execution_start_ns is None:
                state.queue_latency_ms = self._elapsed_ms(state.start_ns)
            state.status = RunStatus.CANCELLED
            state.termination_reason = TerminationReason.CANCELLED
            state.error = _error_trace("runtime", ErrorCategory.CANCELLED, "run_cancelled", error)
            cancellation = error
        except Exception as error:  # pragma: no cover - defensive boundary
            state.status = RunStatus.FAILED
            state.termination_reason = TerminationReason.ERROR
            state.error = _error_trace("runtime", ErrorCategory.INTERNAL, "unexpected_runtime_error", error)
        finally:
            if acquired:
                self._semaphore.release()
            try:
                trace = self._finalize_trace(state, run_metadata, run_filters)
                try:
                    await _finish_shielded(self._write_trace(trace))
                except (_TraceDeliveryPending, TraceDeliveryPendingError) as trace_error:
                    if self.config.trace_required and cancellation is None:
                        raise TracePersistenceError("trace_confirmation_timeout") from trace_error
                    _LOGGER.warning(
                        "agentic trace confirmation deadline exceeded; delivery outcome pending",
                    )
                except _TraceCapacityExhausted as trace_error:
                    if self.config.trace_required and cancellation is None:
                        raise TracePersistenceError("trace_capacity_exhausted") from trace_error
                    _LOGGER.warning("agentic trace delivery not started; sink capacity remained saturated")
                except Exception as trace_error:
                    if self.config.trace_required and cancellation is None:
                        raise TracePersistenceError("trace_delivery_failed") from trace_error
                    _LOGGER.warning(
                        "agentic trace delivery failed (%s)",
                        type(trace_error).__name__,
                    )
            finally:
                await _finish_shielded(self._end_run())

        assert trace is not None
        if cancellation is not None:
            raise cancellation
        from justatom.agentic.telemetry import derive_run_metrics

        return AgenticRunResult(
            run_id=state.run_id,
            answer=state.answer,
            evidence=tuple(state.context.values()),
            trace=trace,
            metrics=derive_run_metrics(trace),
        )

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            if self._finalization_task is None:
                self._closing = True
                self._finalization_task = asyncio.create_task(self._finalize_close())
            task = self._finalization_task
        await asyncio.shield(task)

    async def admission_metrics(self) -> dict[str, int]:
        """Return a process-local, point-in-time admission snapshot."""

        async with self._lifecycle_lock:
            return {
                "current_admitted_run_count": self._accepted_runs,
                "rejected_run_count": self._rejected_runs,
                "max_concurrency": self.config.max_concurrency,
                "max_queued_runs": self.config.max_queued_runs,
                "admission_capacity": self.config.max_concurrency + self.config.max_queued_runs,
            }

    async def _execute(self, state: _RunState, *, filters: dict[str, Any] | None) -> None:
        state.seen_queries.add(normalize_query(state.question))
        await self._search_step(state, state.question, filters=filters)

        while True:
            if len(state.steps) >= self.config.max_steps:
                state.termination_reason = TerminationReason.MAX_STEPS
                state.budget_dimension = "max_steps"
                return
            if state.llm_calls >= self.config.max_llm_calls:
                state.termination_reason = TerminationReason.MAX_LLM_CALLS
                state.budget_dimension = "max_llm_calls"
                return
            if self.config.max_tokens is not None and state.observed_total_tokens >= self.config.max_tokens:
                state.termination_reason = TerminationReason.MAX_TOKENS
                state.budget_dimension = "max_tokens"
                return

            remaining_retrievals = self.config.max_retrieval_calls - state.retrieval_index
            builder = self._new_step(state)
            try:
                request = PlannerRequest(
                    question=state.question,
                    observations=tuple(state.observations),
                    context_documents=tuple(state.context.values()),
                    remaining_retrieval_calls=max(remaining_retrievals, 0),
                    remaining_steps=max(self.config.max_steps - len(state.steps) - 1, 0),
                )
                reply = await self._call_planner(state, builder, request)
                builder.action = reply.decision.action
                builder.decision = self._decision_trace(reply.decision)
            finally:
                self._finish_step(state, builder)

            decision = reply.decision
            if decision.action is AgentAction.ANSWER:
                state.answer = decision.answer.strip() if decision.answer is not None else None
                state.final_cited_document_ids = decision.cited_document_ids
                state.termination_reason = TerminationReason.ANSWERED
                return

            assert decision.query is not None
            if len(state.steps) >= self.config.max_steps:
                state.termination_reason = TerminationReason.MAX_STEPS
                state.budget_dimension = "max_steps"
                return
            if remaining_retrievals <= 0:
                state.termination_reason = TerminationReason.MAX_RETRIEVAL_CALLS
                state.budget_dimension = "max_retrieval_calls"
                return
            if self.config.max_tokens is not None and state.observed_total_tokens >= self.config.max_tokens:
                state.termination_reason = TerminationReason.MAX_TOKENS
                state.budget_dimension = "max_tokens"
                return
            query = _validate_question(decision.query, self.config.max_query_chars, name="planner query")
            normalized = normalize_query(query)
            if normalized in state.seen_queries:
                state.termination_reason = TerminationReason.REPEATED_QUERY
                return
            if state.no_progress_streak >= self.config.no_progress_limit:
                state.termination_reason = TerminationReason.NO_PROGRESS
                return
            state.seen_queries.add(normalized)
            await self._search_step(state, query, filters=filters)

    async def _search_step(self, state: _RunState, query: str, *, filters: dict[str, Any] | None) -> None:
        if state.retrieval_index >= self.config.max_retrieval_calls:
            state.termination_reason = TerminationReason.MAX_RETRIEVAL_CALLS
            state.budget_dimension = "max_retrieval_calls"
            return
        builder = self._new_step(state, AgentAction.SEARCH)
        try:
            documents = await self._call_retrieval(state, builder, query, filters=filters)
            observation, new_documents = self._observe_documents(state, query, documents)
            state.observations.append(observation)
            state.no_progress_streak = 0 if new_documents else state.no_progress_streak + 1
        finally:
            self._finish_step(state, builder)

    async def _call_retrieval(
        self,
        state: _RunState,
        builder: _StepBuilder,
        query: str,
        *,
        filters: dict[str, Any] | None,
    ) -> list[Any]:
        call_id = self._id_factory()
        call_index = state.call_index
        state.call_index += 1
        retrieval_index = state.retrieval_index
        state.retrieval_index += 1
        started_ns = self._clock()
        timing = _ComponentTiming(queue_start_ns=started_ns)
        started_offset_ms = self._duration_ms(state.start_ns, started_ns)
        documents: list[Any] = []
        backend_document_count: int | None = None
        truncated_document_count = 0
        status = CallStatus.OK
        error_trace: ErrorTrace | None = None
        caught: BaseException | None = None
        try:
            documents = await self._call_component(
                lambda: self.retriever.retrieve(query, top_k=self.config.top_k, filters=filters),
                timeout=self.config.retrieval_timeout_seconds,
                timing=timing,
            )
            if not isinstance(documents, list):
                raise TypeError("retriever returned a non-list result")
            backend_document_count = len(documents)
            truncated_document_count = max(backend_document_count - self.config.top_k, 0)
            documents = documents[: self.config.top_k]
            for rank, document in enumerate(documents, start=1):
                _document_identifier(
                    document,
                    retrieval_index,
                    rank,
                    self.config.max_identifier_chars,
                )
        except asyncio.TimeoutError as error:
            status = CallStatus.TIMEOUT
            caught = error
            if timing.execution_start_ns is None:
                error_trace = _error_trace(
                    "runtime",
                    ErrorCategory.TIMEOUT,
                    "retrieval_capacity_timeout",
                    error,
                    retryable=True,
                )
            else:
                error_trace = _error_trace(
                    "retrieval",
                    ErrorCategory.TIMEOUT,
                    "retrieval_timeout",
                    error,
                    retryable=True,
                )
        except asyncio.CancelledError as error:
            status = CallStatus.CANCELLED
            caught = error
            error_trace = _error_trace("retrieval", ErrorCategory.CANCELLED, "retrieval_cancelled", error)
        except Exception as error:
            documents = []
            backend_document_count = None
            truncated_document_count = 0
            status = CallStatus.ERROR
            caught = error
            error_trace = _error_trace("retrieval", ErrorCategory.BACKEND, "retrieval_backend_error", error, retryable=True)
        ended_ns = self._clock()
        latency_ms = self._duration_ms(started_ns, ended_ns)
        queue_latency_ms = self._duration_ms(started_ns, timing.execution_start_ns or ended_ns)
        snapshots = tuple(self._document_trace(document, rank, retrieval_index) for rank, document in enumerate(documents, start=1))
        attempts = (
            (
                AttemptTrace(
                    attempt_index=0,
                    status=status,
                    latency_ms=self._duration_ms(timing.execution_start_ns, ended_ns),
                    error=error_trace,
                ),
            )
            if timing.execution_start_ns is not None
            else ()
        )
        builder.calls.append(
            CallTrace(
                call_id=call_id,
                call_index=call_index,
                kind=CallKind.RETRIEVAL,
                backend=_retrieval_mode(self.retriever),
                model=None,
                started_offset_ms=started_offset_ms,
                latency_ms=latency_ms,
                status=status,
                attempts=attempts,
                queue_latency_ms=queue_latency_ms,
                retrieval=RetrievalPayload(
                    retrieval_index=retrieval_index,
                    query_sha256=self._captured_hash(query),
                    normalized_query_sha256=self._captured_normalized_query_hash(query),
                    query_text=query if self.config.capture_text is TextCapturePolicy.FULL else None,
                    mode=_retrieval_mode(self.retriever),
                    collection=_retrieval_collection(self.retriever),
                    index_revision=_retrieval_index_revision(self.retriever),
                    top_k_requested=self.config.top_k,
                    documents=snapshots,
                    backend_document_count=backend_document_count,
                    truncated_document_count=truncated_document_count,
                ),
                error=error_trace,
            )
        )
        if caught is not None:
            builder.error = error_trace
            if isinstance(caught, asyncio.CancelledError):
                raise caught
            raise _ExecutionFailure(
                TerminationReason.TIMEOUT if status is CallStatus.TIMEOUT else TerminationReason.RETRIEVAL_ERROR,
                error_trace or _error_trace("retrieval", ErrorCategory.INTERNAL, "retrieval_error", caught),
                timed_out=status is CallStatus.TIMEOUT,
            )
        return documents

    async def _call_planner(self, state: _RunState, builder: _StepBuilder, request: PlannerRequest):
        call_id = self._id_factory()
        call_index = state.call_index
        state.call_index += 1
        state.llm_calls += 1
        started_ns = self._clock()
        timing = _ComponentTiming(queue_start_ns=started_ns)
        started_offset_ms = self._duration_ms(state.start_ns, started_ns)
        reply = None
        status = CallStatus.OK
        error_trace: ErrorTrace | None = None
        caught: BaseException | None = None
        try:
            reply = await self._call_component(
                lambda: self.chat_backend.plan(request),
                timeout=self.config.planner_timeout_seconds,
                timing=timing,
            )
            if not isinstance(reply, PlannerReply) or reply.decision.action not in {
                AgentAction.SEARCH,
                AgentAction.ANSWER,
            }:
                raise _InvalidPlannerReply("planner returned an unsupported decision")
            try:
                if reply.decision.action is AgentAction.SEARCH:
                    _validate_question(reply.decision.query, self.config.max_query_chars, name="planner query")
                else:
                    _validate_answer(reply.decision.answer, self.config.max_answer_chars)
                _validate_optional_text(reply.decision.reason, self.config.max_reason_chars, "planner reason")
                if len(reply.decision.cited_document_ids) > self.config.max_context_documents:
                    raise ValueError("planner returned too many citations")
                if any(len(document_id) > self.config.max_identifier_chars for document_id in reply.decision.cited_document_ids):
                    raise ValueError("planner returned an oversized citation identifier")
            except ValueError as error:
                raise _InvalidPlannerReply("planner decision exceeded a runtime bound") from error
        except asyncio.TimeoutError as error:
            status = CallStatus.TIMEOUT
            caught = error
            if timing.execution_start_ns is None:
                error_trace = _error_trace(
                    "runtime",
                    ErrorCategory.TIMEOUT,
                    "planner_capacity_timeout",
                    error,
                    retryable=True,
                )
            else:
                error_trace = _error_trace(
                    "planner",
                    ErrorCategory.TIMEOUT,
                    "planner_timeout",
                    error,
                    retryable=True,
                )
        except asyncio.CancelledError as error:
            status = CallStatus.CANCELLED
            caught = error
            error_trace = _error_trace("planner", ErrorCategory.CANCELLED, "planner_cancelled", error)
        except _InvalidPlannerReply as error:
            status = CallStatus.ERROR
            caught = error
            error_trace = _error_trace(
                "planner",
                ErrorCategory.VALIDATION,
                "invalid_planner_reply",
                error,
            )
        except Exception as error:
            caught = error
            backend_attempts = tuple(getattr(error, "attempts", ()))
            terminal_attempt = backend_attempts[-1] if backend_attempts else None
            backend_error = (
                terminal_attempt.error
                if terminal_attempt is not None and terminal_attempt.error is not None
                else next(
                    (attempt.error for attempt in reversed(backend_attempts) if attempt.error is not None),
                    None,
                )
            )
            timed_out = (
                terminal_attempt.status is CallStatus.TIMEOUT
                if terminal_attempt is not None
                else backend_error is not None and backend_error.category is ErrorCategory.TIMEOUT
            )
            status = CallStatus.TIMEOUT if timed_out else CallStatus.ERROR
            error_trace = backend_error or _error_trace(
                "planner",
                ErrorCategory.BACKEND,
                "planner_backend_error",
                error,
                retryable=True,
            )
        ended_ns = self._clock()
        latency_ms = self._duration_ms(started_ns, ended_ns)
        queue_latency_ms = self._duration_ms(started_ns, timing.execution_start_ns or ended_ns)
        if isinstance(reply, PlannerReply):
            attempts = tuple(reply.attempts)
        else:
            attempts = tuple(getattr(caught, "attempts", ())) if caught is not None else ()
        if not attempts and timing.execution_start_ns is not None:
            backend_succeeded = reply is not None
            attempts = (
                AttemptTrace(
                    attempt_index=0,
                    status=CallStatus.OK if backend_succeeded else status,
                    latency_ms=self._duration_ms(timing.execution_start_ns, ended_ns),
                    provider_request_id=getattr(reply, "provider_request_id", None),
                    error=None if backend_succeeded else error_trace,
                ),
            )
        usage = reply.usage if reply is not None else None
        observed_tokens = _observed_total_tokens(usage)
        if observed_tokens is not None:
            state.observed_total_tokens += observed_tokens
        builder.calls.append(
            CallTrace(
                call_id=call_id,
                call_index=call_index,
                kind=CallKind.PLANNER,
                backend=self.chat_backend.backend_name,
                model=reply.model if reply is not None else self.chat_backend.model_name,
                started_offset_ms=started_offset_ms,
                latency_ms=latency_ms,
                status=status,
                attempts=attempts,
                queue_latency_ms=queue_latency_ms,
                time_to_first_token_ms=reply.time_to_first_token_ms if reply is not None else None,
                finish_reason=reply.finish_reason if reply is not None else None,
                cache_hit=reply.cache_hit if reply is not None else None,
                tokens=usage,
                cost=reply.cost if reply is not None else None,
                error=error_trace,
            )
        )
        if caught is not None:
            builder.error = error_trace
            if isinstance(caught, asyncio.CancelledError):
                raise caught
            raise _ExecutionFailure(
                (
                    TerminationReason.TIMEOUT
                    if status is CallStatus.TIMEOUT
                    else (
                        TerminationReason.INVALID_ACTION
                        if isinstance(caught, _InvalidPlannerReply)
                        else TerminationReason.PLANNER_ERROR
                    )
                ),
                error_trace or _error_trace("planner", ErrorCategory.INTERNAL, "planner_error", caught),
                timed_out=status is CallStatus.TIMEOUT,
            )
        return reply

    def _observe_documents(self, state: _RunState, query: str, documents: list[Any]) -> tuple[SearchObservation, int]:
        observed: list[EvidenceDocument] = []
        new_documents = 0
        for rank, document in enumerate(documents, start=1):
            document_id = _document_identifier(
                document,
                state.retrieval_index - 1,
                rank,
                self.config.max_identifier_chars,
            )
            content = _document_content(document)
            content = content[: self.config.max_document_chars]
            evidence = EvidenceDocument(
                document_id=document_id,
                content=content,
                score=_safe_score(getattr(document, "score", None)),
                rank=rank,
                retrieval_index=state.retrieval_index - 1,
            )
            observed.append(
                EvidenceDocument(
                    document_id=document_id,
                    content="",
                    score=evidence.score,
                    rank=rank,
                    retrieval_index=evidence.retrieval_index,
                )
            )
            if document_id in state.context:
                continue
            if len(state.context) >= self.config.max_context_documents:
                continue
            remaining_chars = self.config.max_context_chars - state.context_chars
            if remaining_chars <= 0:
                continue
            retained = evidence
            if len(content) > remaining_chars:
                retained = EvidenceDocument(
                    document_id=document_id,
                    content=content[:remaining_chars],
                    score=evidence.score,
                    rank=rank,
                    retrieval_index=evidence.retrieval_index,
                )
            state.context[document_id] = retained
            state.context_chars += len(retained.content)
            new_documents += 1
        return SearchObservation(query=query, documents=tuple(observed)), new_documents

    def _document_trace(self, document: Any, rank: int, retrieval_index: int) -> DocumentTrace:
        content = _document_content(document)
        document_id = _document_identifier(document, retrieval_index, rank, self.config.max_identifier_chars)
        return DocumentTrace(
            document_id=document_id,
            rank=rank,
            score=_safe_score(getattr(document, "score", None)),
            content_chars=len(content),
            content_sha256=self._captured_hash(content),
            content=content[: self.config.max_document_chars] if self.config.capture_text is TextCapturePolicy.FULL else None,
        )

    def _new_step(self, state: _RunState, action: AgentAction = AgentAction.STOP) -> _StepBuilder:
        return _StepBuilder(step_index=len(state.steps), start_ns=self._clock(), action=action)

    def _finish_step(self, state: _RunState, builder: _StepBuilder) -> None:
        if builder.finished:
            return
        builder.finished = True
        state.steps.append(
            StepTrace(
                step_index=builder.step_index,
                started_offset_ms=self._duration_ms(state.start_ns, builder.start_ns),
                latency_ms=self._elapsed_ms(builder.start_ns),
                action=builder.action,
                calls=tuple(builder.calls),
                context_document_ids=tuple(state.context),
                decision=builder.decision,
                error=builder.error,
            )
        )

    def _finalize_trace(
        self,
        state: _RunState,
        metadata: dict[str, Any],
        filters: dict[str, Any] | None,
    ) -> RunTrace:
        ended_ns = self._clock()
        execution_ms = 0.0
        if state.execution_start_ns is not None:
            execution_ms = self._duration_ms(state.execution_start_ns, ended_ns)
        answer = state.answer
        return RunTrace(
            schema_version=TRACE_SCHEMA_VERSION,
            run_id=state.run_id,
            request_id=state.request_id,
            experiment_id=self.config.experiment_id,
            variant=self.config.variant,
            seed=self.config.seed,
            config_fingerprint=self.config_fingerprint,
            planner_config_fingerprint=self.planner_config_fingerprint,
            retrieval_config_fingerprint=self.retrieval_config_fingerprint,
            capture_text=self.config.capture_text,
            query_sha256=self._captured_hash(state.question),
            normalized_query_sha256=self._captured_normalized_query_hash(state.question),
            query_text=state.question if self.config.capture_text is TextCapturePolicy.FULL else None,
            started_at=state.started_at,
            duration_ms=self._duration_ms(state.start_ns, ended_ns),
            queue_latency_ms=state.queue_latency_ms,
            execution_ms=execution_ms,
            status=state.status,
            termination_reason=state.termination_reason,
            budget_dimension=state.budget_dimension,
            limits=self.config.limits,
            steps=tuple(state.steps),
            final_context_document_ids=tuple(state.context),
            answer_sha256=self._captured_hash(answer) if answer is not None else None,
            answer_text=answer if answer is not None and self.config.capture_text is TextCapturePolicy.FULL else None,
            final_context_chars=state.context_chars,
            final_cited_document_ids=state.final_cited_document_ids,
            filters_sha256=self._captured_json_hash(filters) if filters is not None else None,
            filters=filters if filters is not None and self.config.capture_text is TextCapturePolicy.FULL else None,
            error=state.error,
            metadata=metadata,
        )

    def _decision_trace(self, decision: PlannerDecision) -> DecisionTrace:
        query = decision.query.strip() if decision.query is not None else None
        answer = decision.answer.strip() if decision.answer is not None else None
        reason = decision.reason.strip() if decision.reason is not None else None
        return DecisionTrace(
            action=decision.action,
            query_sha256=self._captured_hash(query) if query is not None else None,
            normalized_query_sha256=(self._captured_normalized_query_hash(query) if query is not None else None),
            query_text=query if query is not None and self.config.capture_text is TextCapturePolicy.FULL else None,
            answer_sha256=self._captured_hash(answer) if answer is not None else None,
            answer_text=answer if answer is not None and self.config.capture_text is TextCapturePolicy.FULL else None,
            reason_sha256=self._captured_hash(reason) if reason is not None else None,
            reason_text=reason if reason is not None and self.config.capture_text is TextCapturePolicy.FULL else None,
            cited_document_ids=decision.cited_document_ids,
        )

    async def _begin_run(self) -> None:
        async with self._lifecycle_lock:
            if self._closing or self._closed:
                raise RuntimeError("AgenticRAGRuntime is closed")
            capacity = self.config.max_concurrency + self.config.max_queued_runs
            if self._accepted_runs >= capacity:
                self._rejected_runs += 1
                raise AgenticCapacityError("agentic run capacity exhausted")
            self._accepted_runs += 1
            self._idle.clear()

    async def _end_run(self) -> None:
        async with self._lifecycle_lock:
            self._accepted_runs -= 1
            if self._accepted_runs == 0:
                self._idle.set()

    async def _finalize_close(self) -> None:
        await self._idle.wait()
        await self._operations_idle.wait()
        await self._trace_writes_idle.wait()
        first_error: BaseException | None = None
        later_error: BaseException | None = None
        for resource in (self.trace_sink, self.chat_backend):
            try:
                await resource.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
                else:
                    later_error = error
        async with self._lifecycle_lock:
            self._closed = True
        if first_error is not None:
            if later_error is not None:
                raise first_error from later_error
            raise first_error

    async def _call_component(
        self,
        operation_factory: Callable[[], Awaitable[Any]],
        *,
        timeout: float,
        timing: _ComponentTiming,
    ) -> Any:
        """Run one backend operation without waiting for cancellation drain.

        A backend task is deliberately left running after the caller times out
        or is cancelled. Cancelling a coroutine that awaits ``to_thread`` or
        ``run_in_executor`` marks the asyncio task as cancelled immediately,
        even though its native worker is still running. Keeping the task alive
        lets the detached operation retain its component-concurrency permit
        until the backend really exits, so timed-out native/GPU work cannot
        accumulate behind apparently free runtime slots.
        """

        started_ns = timing.queue_start_ns
        acquired = False
        detached = False
        task: asyncio.Future[Any] | None = None
        try:
            await asyncio.wait_for(self._component_semaphore.acquire(), timeout=timeout)
            acquired = True
            remaining = timeout - self._elapsed_ms(started_ns) / 1_000.0
            if remaining <= 0:
                raise asyncio.TimeoutError
            timing.execution_start_ns = self._clock()
            task = asyncio.ensure_future(operation_factory())
            done, _ = await asyncio.wait((task,), timeout=remaining)
            if task not in done:
                raise asyncio.TimeoutError
            return task.result()
        except (asyncio.TimeoutError, asyncio.CancelledError):
            if task is not None:
                detached = True
                self._track_component_operation(task)
            raise
        finally:
            if acquired and not detached:
                self._component_semaphore.release()

    def _track_component_operation(self, task: asyncio.Future[Any]) -> None:
        self._inflight_operations.add(task)
        self._operations_idle.clear()
        task.add_done_callback(self._component_operation_finished)

    def _component_operation_finished(self, task: asyncio.Future[Any]) -> None:
        self._component_semaphore.release()
        self._inflight_operations.discard(task)
        if not self._inflight_operations:
            self._operations_idle.set()
        if not task.cancelled():
            task.exception()

    async def _write_trace(self, trace: RunTrace) -> None:
        acquired = False
        detached = False
        task: asyncio.Future[Any] | None = None
        try:
            if self._available_trace_slots <= 0:
                raise _TraceCapacityExhausted
            self._available_trace_slots -= 1
            acquired = True
            task = asyncio.ensure_future(self.trace_sink.write(trace))
            done, _ = await asyncio.wait((task,), timeout=self.config.trace_timeout_seconds)
            if task not in done:
                task.cancel()
                detached = True
                self._track_trace_write(task)
                raise _TraceDeliveryPending
            task.result()
        except asyncio.CancelledError:
            if task is not None and not detached:
                task.cancel()
                detached = True
                self._track_trace_write(task)
            raise
        finally:
            if acquired and not detached:
                self._available_trace_slots += 1

    def _track_trace_write(self, task: asyncio.Future[Any]) -> None:
        self._inflight_trace_writes.add(task)
        self._trace_writes_idle.clear()
        task.add_done_callback(self._trace_write_finished)

    def _trace_write_finished(self, task: asyncio.Future[Any]) -> None:
        self._available_trace_slots += 1
        self._inflight_trace_writes.discard(task)
        if not self._inflight_trace_writes:
            self._trace_writes_idle.set()
        if not task.cancelled():
            task.exception()

    def _fingerprint_config(self) -> str:
        payload = {
            "max_steps": self.config.max_steps,
            "max_retrieval_calls": self.config.max_retrieval_calls,
            "max_llm_calls": self.config.max_llm_calls,
            "max_tokens": self.config.max_tokens,
            "total_timeout_seconds": self.config.total_timeout_seconds,
            "retrieval_timeout_seconds": self.config.retrieval_timeout_seconds,
            "planner_timeout_seconds": self.config.planner_timeout_seconds,
            "trace_timeout_seconds": self.config.trace_timeout_seconds,
            "max_concurrency": self.config.max_concurrency,
            "max_queued_runs": self.config.max_queued_runs,
            "top_k": self.config.top_k,
            "max_request_bytes": self.config.max_request_bytes,
            "max_query_chars": self.config.max_query_chars,
            "max_answer_chars": self.config.max_answer_chars,
            "max_reason_chars": self.config.max_reason_chars,
            "max_identifier_chars": self.config.max_identifier_chars,
            "max_filter_chars": self.config.max_filter_chars,
            "max_metadata_chars": self.config.max_metadata_chars,
            "max_document_chars": self.config.max_document_chars,
            "max_context_chars": self.config.max_context_chars,
            "max_context_documents": self.config.max_context_documents,
            "no_progress_limit": self.config.no_progress_limit,
            "capture_text": self.config.capture_text.value,
            "trace_required": self.config.trace_required,
            "trace_sink_type": _type_name(self.trace_sink),
            "trace_sink_required": _fingerprint_scalar(getattr(self.trace_sink, "required", None)),
            "trace_max_pending_writes": _fingerprint_scalar(getattr(self.trace_sink, "max_pending_writes", None)),
            "experiment_id": self.config.experiment_id,
            "variant": self.config.variant,
            "seed": self.config.seed,
            "metadata": self._base_metadata,
            "planner_config_fingerprint": self.planner_config_fingerprint,
            "retrieval_config_fingerprint": self.retrieval_config_fingerprint,
        }
        return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))

    def _captured_hash(self, value: str) -> str | None:
        if self.config.capture_text is TextCapturePolicy.NONE:
            return None
        return sha256_text(value)

    def _captured_normalized_query_hash(self, value: str) -> str | None:
        if self.config.capture_text is TextCapturePolicy.NONE:
            return None
        return normalized_query_sha256(value)

    def _captured_json_hash(self, value: Mapping[str, Any]) -> str | None:
        if self.config.capture_text is TextCapturePolicy.NONE:
            return None
        _, canonical = _canonical_json_mapping(value, "trace value")
        return sha256_text(canonical)

    def _elapsed_ms(self, start_ns: int) -> float:
        return self._duration_ms(start_ns, self._clock())

    @staticmethod
    def _duration_ms(start_ns: int, end_ns: int) -> float:
        return max(end_ns - start_ns, 0) / 1_000_000.0


async def build_agentic_runtime(
    config: Mapping[str, Any],
    retriever: AgentRetriever,
    *,
    transport: Any | None = None,
) -> AgenticRAGRuntime | None:
    """Build an optional agentic runtime from a strict configuration mapping."""

    if not isinstance(config, Mapping):
        raise AgenticConfigurationError("agentic config must be a mapping")
    _reject_unknown(config, _ROOT_CONFIG_KEYS, "agentic")
    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise AgenticConfigurationError("agentic.enabled must be a boolean")
    if not enabled:
        return None

    planner = _mapping(config.get("planner"), "agentic.planner")
    trace = _mapping(config.get("trace", {}), "agentic.trace")
    _reject_unknown(planner, _PLANNER_CONFIG_KEYS, "agentic.planner")
    _reject_unknown(trace, _TRACE_CONFIG_KEYS, "agentic.trace")
    backend_name = _required_string(planner, "backend", prefix="agentic.planner")
    if backend_name != "openai-compatible":
        raise AgenticConfigurationError("agentic.planner.backend must be 'openai-compatible'")

    from justatom.agentic.openai_compatible import OpenAICompatibleChatBackend
    from justatom.agentic.telemetry import JsonlTraceSink, NullTraceSink

    capture_raw = trace.get("capture_text", "hash")
    try:
        capture_text = TextCapturePolicy(capture_raw)
    except (TypeError, ValueError) as error:
        raise AgenticConfigurationError("agentic.trace.capture_text must be none, hash, or full") from error
    trace_required = trace.get("required", True)
    if not isinstance(trace_required, bool):
        raise AgenticConfigurationError("agentic.trace.required must be a boolean")
    trace_path = trace.get("path")
    if trace_path is None:
        if trace_required:
            raise AgenticConfigurationError("agentic.trace.path is required when agentic.trace.required is true")
    elif not isinstance(trace_path, (str, Path)) or not str(trace_path).strip():
        raise AgenticConfigurationError("agentic.trace.path must be a non-empty path or null")
    trace_max_pending_writes = _integer(trace, "max_pending_writes", 64)

    runtime_config = AgenticRuntimeConfig(
        max_steps=_integer(config, "max_steps", 4),
        max_retrieval_calls=_integer(config, "max_retrieval_calls", 3),
        max_llm_calls=_integer(config, "max_llm_calls", 3),
        max_tokens=_optional_integer(config, "max_tokens"),
        total_timeout_seconds=_number(config, "total_timeout_seconds", 60.0),
        retrieval_timeout_seconds=_number(config, "retrieval_timeout_seconds", 30.0),
        planner_timeout_seconds=_number(config, "planner_timeout_seconds", 30.0),
        trace_timeout_seconds=_number(trace, "timeout_seconds", 5.0),
        max_concurrency=_integer(config, "max_concurrency", 2),
        max_queued_runs=_nonnegative_integer(config, "max_queued_runs", 8),
        top_k=_integer(config, "top_k", 10),
        max_request_bytes=_integer(config, "max_request_bytes", 65_536),
        max_query_chars=_integer(config, "max_query_chars", 512),
        max_answer_chars=_integer(config, "max_answer_chars", 16_000),
        max_reason_chars=_integer(config, "max_reason_chars", 4_000),
        max_identifier_chars=_integer(config, "max_identifier_chars", 512),
        max_filter_chars=_integer(config, "max_filter_chars", 16_000),
        max_metadata_chars=_integer(config, "max_metadata_chars", 8_000),
        max_document_chars=_integer(config, "max_document_chars", 2_000),
        max_context_chars=_integer(config, "max_context_chars", 24_000),
        max_context_documents=_integer(config, "max_context_documents", 50),
        no_progress_limit=_integer(config, "no_progress_limit", 2),
        capture_text=capture_text,
        trace_required=trace_required,
        experiment_id=_optional_string(config.get("experiment_id"), "agentic.experiment_id"),
        variant=_optional_string(config.get("variant"), "agentic.variant"),
        seed=config.get("seed"),
        metadata=dict(_mapping(config.get("metadata", {}), "agentic.metadata")),
    )
    backend = OpenAICompatibleChatBackend(
        base_url=_required_string(planner, "base_url", prefix="agentic.planner"),
        model=_required_string(planner, "model", prefix="agentic.planner"),
        api_key=_optional_string(planner.get("api_key"), "agentic.planner.api_key"),
        timeout_seconds=_number(planner, "timeout_seconds", 30.0),
        temperature=_number(planner, "temperature", 0.0, nonnegative=True),
        max_tokens=_integer(planner, "max_tokens", 512),
        max_response_bytes=_integer(planner, "max_response_bytes", 1_048_576),
        seed=_optional_seed(
            planner.get("seed") if planner.get("seed") is not None else config.get("seed"),
            "agentic.planner.seed",
        ),
        system_prompt=_optional_string(planner.get("system_prompt"), "agentic.planner.system_prompt"),
        transport=transport,
    )
    sink: TraceSink
    if trace_path is None:
        sink = NullTraceSink(required=False)
    else:
        sink = JsonlTraceSink(
            trace_path,
            required=trace_required,
            max_pending_writes=trace_max_pending_writes,
        )
    try:
        return AgenticRAGRuntime(retriever, backend, config=runtime_config, trace_sink=sink)
    except BaseException:
        await sink.close()
        await backend.close()
        raise


def _reject_unknown(values: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(str(key) for key in values if key not in allowed)
    if unknown:
        raise AgenticConfigurationError(f"unknown {section} keys: {', '.join(unknown)}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AgenticConfigurationError(f"{name} must be a mapping")
    return value


def _required_string(values: Mapping[str, Any], key: str, *, prefix: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AgenticConfigurationError(f"{prefix}.{key} must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AgenticConfigurationError(f"{name} must be a non-empty string or null")
    return value.strip()


def _integer(values: Mapping[str, Any], key: str, default: int) -> int:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AgenticConfigurationError(f"{key} must be a positive integer")
    return value


def _nonnegative_integer(values: Mapping[str, Any], key: str, default: int) -> int:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AgenticConfigurationError(f"{key} must be a non-negative integer")
    return value


def _optional_integer(values: Mapping[str, Any], key: str) -> int | None:
    value = values.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AgenticConfigurationError(f"{key} must be a positive integer or null")
    return value


def _optional_seed(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise AgenticConfigurationError(f"{name} must be an integer or null")
    return value


def _number(values: Mapping[str, Any], key: str, default: float, *, nonnegative: bool = False) -> float:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AgenticConfigurationError(f"{key} must be a finite number")
    if (nonnegative and value < 0) or (not nonnegative and value <= 0):
        qualifier = "non-negative" if nonnegative else "positive"
        raise AgenticConfigurationError(f"{key} must be a {qualifier} finite number")
    return float(value)


def _validate_question(value: Any, max_chars: int, *, name: str = "question") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    value = value.strip()
    if len(value) > max_chars:
        raise ValueError(f"{name} exceeds max_query_chars={max_chars}")
    return value


def _validate_answer(value: Any, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("planner answer must be a non-empty string")
    value = value.strip()
    if len(value) > max_chars:
        raise ValueError(f"planner answer exceeds max_answer_chars={max_chars}")
    return value


def _validate_optional_text(value: Any, max_chars: int, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string or null")
    value = value.strip()
    if len(value) > max_chars:
        raise ValueError(f"{name} exceeds its configured character limit={max_chars}")
    return value


def _validate_optional_identifier(value: Any, max_chars: int, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string or null")
    value = value.strip()
    if len(value) > max_chars:
        raise ValueError(f"{name} exceeds max_identifier_chars={max_chars}")
    return value


def _observed_total_tokens(usage: TokenUsage | None) -> int | None:
    if usage is None:
        return None
    if usage.total_tokens is not None:
        return usage.total_tokens
    if usage.input_tokens is not None and usage.output_tokens is not None:
        return usage.input_tokens + usage.output_tokens
    return None


def _canonical_json_mapping(value: Mapping[str, Any], name: str) -> tuple[dict[str, Any], str]:
    try:
        serialized = json.dumps(dict(value), allow_nan=False, ensure_ascii=False)
        normalized = json.loads(serialized)
        canonical = json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise AgenticConfigurationError(f"{name} must contain only finite JSON values") from error
    if not isinstance(normalized, dict):  # pragma: no cover - Mapping always serializes to an object
        raise AgenticConfigurationError(f"{name} must be a mapping")
    return normalized, canonical


def _safe_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return score if math.isfinite(score) else None


def _document_content(document: Any) -> str:
    content = getattr(document, "content", "")
    return content if isinstance(content, str) else str(content)


def _document_identifier(document: Any, retrieval_index: int, rank: int, max_chars: int) -> str:
    value = str(getattr(document, "id", "") or f"anonymous-{retrieval_index}-{rank}")
    if not value.strip():
        raise ValueError("retriever document id must not be blank")
    if len(value) > max_chars:
        raise ValueError(f"retriever document id exceeds max_identifier_chars={max_chars}")
    return value


def _retrieval_mode(retriever: Any) -> str | None:
    mode = getattr(retriever, "mode", None)
    value = getattr(mode, "value", mode) if mode is not None else type(retriever).__name__
    return str(value) if value is not None else None


def _retrieval_collection(retriever: Any) -> str | None:
    store = getattr(retriever, "store", None)
    value = getattr(store, "collection_name", None)
    if value is None:
        getter = getattr(store, "get_collection_name", None)
        if callable(getter):
            try:
                value = getter()
            except Exception:
                return None
    return str(value) if value is not None else None


def _retrieval_index_revision(retriever: Any) -> str | None:
    for owner in (retriever, getattr(retriever, "store", None)):
        value = getattr(owner, "index_revision", None)
        if value is not None:
            return str(value)
    return None


def _type_name(value: Any) -> str | None:
    if value is None:
        return None
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _fingerprint_scalar(value: Any) -> str | int | float | bool | None:
    enum_value = getattr(value, "value", value)
    if enum_value is None or isinstance(enum_value, (str, bool, int)):
        return enum_value
    if isinstance(enum_value, float) and math.isfinite(enum_value):
        return enum_value
    return _type_name(enum_value)


def _planner_config_fingerprint(backend: Any) -> str:
    explicit = getattr(backend, "config_fingerprint", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    payload = {
        "backend_type": _type_name(backend),
        "backend_name": getattr(backend, "backend_name", None),
        "model_name": getattr(backend, "model_name", None),
        "prompt_fingerprint": getattr(backend, "prompt_fingerprint", None),
    }
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))


def _retrieval_config_fingerprint(retriever: Any) -> str:
    explicit = getattr(retriever, "config_fingerprint", None)
    if isinstance(explicit, str) and explicit:
        return explicit

    delegate = getattr(retriever, "retriever", retriever)
    store = getattr(retriever, "store", None)
    if store is None:
        store = getattr(delegate, "store", None)
    embedder = getattr(retriever, "embedder", None)
    if embedder is None:
        embedder = getattr(delegate, "embedder", None)
    profile = getattr(embedder, "profile", None)
    profile_payload = {
        name: _fingerprint_scalar(getattr(profile, name))
        for name in (
            "query_prefix",
            "document_prefix",
            "max_length",
            "batch_size",
            "skip_prefix_if_present",
        )
        if profile is not None and hasattr(profile, name)
    }
    base_url = getattr(embedder, "base_url", None)
    store_url = getattr(store, "_url", None)
    index_revision = _retrieval_index_revision(retriever)
    extra_body = getattr(embedder, "extra_body", None)
    extra_body_fingerprint = None
    if isinstance(extra_body, Mapping):
        try:
            _, canonical = _canonical_json_mapping(extra_body, "embedder.extra_body")
            extra_body_fingerprint = sha256_text(canonical)
        except AgenticConfigurationError:
            extra_body_fingerprint = "unavailable"
    payload = {
        "runtime_type": _type_name(retriever),
        "retriever_type": _type_name(delegate),
        "mode": _retrieval_mode(retriever),
        "collection": _retrieval_collection(retriever),
        "index_revision": index_revision,
        "index_revision_available": index_revision is not None,
        "alpha": _safe_score(getattr(delegate, "alpha", None)),
        "store_type": _type_name(store),
        "store_endpoint_sha256": sha256_text(store_url) if isinstance(store_url, str) else None,
        "store_grpc_port": _fingerprint_scalar(getattr(store, "_grpc_port", None)),
        "store_grpc_secure": _fingerprint_scalar(getattr(store, "_grpc_secure", None)),
        "embedder_type": _type_name(embedder),
        "embedder_model": _fingerprint_scalar(getattr(embedder, "model", None)),
        "embedder_device": _fingerprint_scalar(getattr(embedder, "device", None)),
        "embedder_timeout_seconds": _fingerprint_scalar(getattr(embedder, "timeout", None)),
        "embedder_base_url_sha256": sha256_text(base_url) if isinstance(base_url, str) else None,
        "embedder_profile": profile_payload,
        "embedder_encoding_format": _fingerprint_scalar(getattr(embedder, "encoding_format", None)),
        "embedder_extra_body_fingerprint": extra_body_fingerprint,
    }
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))


def _error_trace(
    component: str,
    category: ErrorCategory,
    code: str,
    error: BaseException,
    *,
    retryable: bool = False,
) -> ErrorTrace:
    return ErrorTrace(
        component=component,
        category=category,
        code=code,
        exception_type=type(error).__name__,
        retryable=retryable,
    )


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


async def _finish_shielded(awaitable: Any) -> None:
    task = asyncio.ensure_future(awaitable)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await asyncio.shield(task)
        except BaseException:
            pass
        raise


__all__ = [
    "AgenticCapacityError",
    "AgenticConfigurationError",
    "AgenticRAGRuntime",
    "AgenticRunResult",
    "AgenticRuntimeConfig",
    "build_agentic_runtime",
]
