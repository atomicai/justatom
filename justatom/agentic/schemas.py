from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any

TRACE_SCHEMA_VERSION = 2
_WHITESPACE = re.compile(r"\s+")


class AgentAction(str, Enum):
    SEARCH = "search"
    ANSWER = "answer"
    STOP = "stop"


class AgentObjective(str, Enum):
    ANSWER = "answer"
    CONTEXT = "context"


class RunStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class CallStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class CallKind(str, Enum):
    PLANNER = "planner"
    RETRIEVAL = "retrieval"
    RERANKER = "reranker"
    ANSWER = "answer"


class TerminationReason(str, Enum):
    ANSWERED = "answered"
    AGENT_STOP = "agent_stop"
    MAX_STEPS = "max_steps"
    MAX_RETRIEVAL_CALLS = "max_retrieval_calls"
    MAX_LLM_CALLS = "max_llm_calls"
    MAX_TOKENS = "max_tokens"
    MAX_DURATION = "max_duration"
    NO_PROGRESS = "no_progress"
    REPEATED_QUERY = "repeated_query"
    INVALID_ACTION = "invalid_action"
    PLANNER_ERROR = "planner_error"
    RETRIEVAL_ERROR = "retrieval_error"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class TextCapturePolicy(str, Enum):
    NONE = "none"
    HASH = "hash"
    FULL = "full"


class ErrorCategory(str, Enum):
    VALIDATION = "validation"
    PARSE = "parse"
    TIMEOUT = "timeout"
    BACKEND = "backend"
    INTERNAL = "internal"
    CANCELLED = "cancelled"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_query(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return _WHITESPACE.sub(" ", normalized.strip()).casefold()


def normalized_query_sha256(value: str) -> str:
    return sha256_text(normalize_query(value))


def _nonnegative_optional_int(value: int | None, name: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError(f"{name} must be a non-negative integer or null")


def _nonnegative_optional_float(value: float | None, name: str) -> None:
    if value is None:
        return
    try:
        finite = math.isfinite(value) if not isinstance(value, bool) and isinstance(value, (int, float)) else False
    except OverflowError:
        finite = False
    if not finite or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number or null")


def _finite_optional_float(value: float | None, name: str) -> None:
    if value is None:
        return
    try:
        finite = math.isfinite(value) if not isinstance(value, bool) and isinstance(value, (int, float)) else False
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number or null")


def _nonempty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _optional_string(value: Any, name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}  # type: ignore[arg-type]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    source: str = "unknown"

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        ):
            _nonnegative_optional_int(getattr(self, name), name)
        if self.source not in {"provider", "tokenizer", "unknown"}:
            raise ValueError("token source must be provider, tokenizer, or unknown")


@dataclass(frozen=True, slots=True)
class CostUsage:
    usd: float | None = None
    source: str = "unavailable"
    pricing_id: str | None = None

    def __post_init__(self) -> None:
        _nonnegative_optional_float(self.usd, "usd")
        if self.source not in {"provider", "price_table", "unavailable"}:
            raise ValueError("cost source must be provider, price_table, or unavailable")
        _optional_string(self.pricing_id, "pricing_id")


@dataclass(frozen=True, slots=True)
class ErrorTrace:
    component: str
    category: ErrorCategory
    code: str
    exception_type: str | None = None
    retryable: bool = False

    def __post_init__(self) -> None:
        _nonempty_string(self.component, "component")
        _nonempty_string(self.code, "code")
        if not isinstance(self.category, ErrorCategory):
            raise ValueError("category must be an ErrorCategory")
        _optional_string(self.exception_type, "exception_type")
        if not isinstance(self.retryable, bool):
            raise ValueError("retryable must be a boolean")


@dataclass(frozen=True, slots=True)
class AttemptTrace:
    attempt_index: int
    status: CallStatus
    latency_ms: float
    provider_request_id: str | None = None
    error: ErrorTrace | None = None

    def __post_init__(self) -> None:
        if isinstance(self.attempt_index, bool) or not isinstance(self.attempt_index, int) or self.attempt_index < 0:
            raise ValueError("attempt_index must be a non-negative integer")
        if not isinstance(self.status, CallStatus):
            raise ValueError("status must be a CallStatus")
        _nonnegative_optional_float(self.latency_ms, "latency_ms")
        _optional_string(self.provider_request_id, "provider_request_id")


@dataclass(frozen=True, slots=True)
class DocumentTrace:
    document_id: str
    rank: int
    score: float | None
    content_chars: int
    content_sha256: str | None
    content: str | None = None

    def __post_init__(self) -> None:
        _nonempty_string(self.document_id, "document_id")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        _finite_optional_float(self.score, "score")
        if isinstance(self.content_chars, bool) or not isinstance(self.content_chars, int) or self.content_chars < 0:
            raise ValueError("content_chars must be a non-negative integer")
        _optional_string(self.content_sha256, "content_sha256")
        _optional_string(self.content, "content")


@dataclass(frozen=True, slots=True)
class RetrievalPayload:
    retrieval_index: int
    query_sha256: str | None
    normalized_query_sha256: str | None
    query_text: str | None
    mode: str | None
    collection: str | None
    index_revision: str | None
    top_k_requested: int
    documents: tuple[DocumentTrace, ...]
    backend_document_count: int | None = None
    truncated_document_count: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.retrieval_index, bool) or not isinstance(self.retrieval_index, int) or self.retrieval_index < 0:
            raise ValueError("retrieval_index must be a non-negative integer")
        for name in ("query_sha256", "normalized_query_sha256", "query_text", "mode", "collection", "index_revision"):
            _optional_string(getattr(self, name), name)
        if isinstance(self.top_k_requested, bool) or not isinstance(self.top_k_requested, int) or self.top_k_requested <= 0:
            raise ValueError("top_k_requested must be a positive integer")
        _nonnegative_optional_int(self.backend_document_count, "backend_document_count")
        if (
            isinstance(self.truncated_document_count, bool)
            or not isinstance(self.truncated_document_count, int)
            or self.truncated_document_count < 0
        ):
            raise ValueError("truncated_document_count must be a non-negative integer")
        if len(self.documents) > self.top_k_requested:
            raise ValueError("retrieval documents must not exceed top_k_requested")
        if any(not isinstance(document, DocumentTrace) for document in self.documents):
            raise ValueError("retrieval documents must be DocumentTrace values")
        if tuple(document.rank for document in self.documents) != tuple(range(1, len(self.documents) + 1)):
            raise ValueError("retrieval document ranks must be consecutive and start at 1")
        if self.backend_document_count is not None and self.backend_document_count != (
            len(self.documents) + self.truncated_document_count
        ):
            raise ValueError("backend_document_count must equal retained plus truncated documents")


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    """Privacy-filtered planner decision recorded before runtime routing."""

    action: AgentAction
    query_sha256: str | None = None
    normalized_query_sha256: str | None = None
    query_text: str | None = None
    answer_sha256: str | None = None
    answer_text: str | None = None
    reason_sha256: str | None = None
    reason_text: str | None = None
    cited_document_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.action, AgentAction):
            raise ValueError("action must be an AgentAction")
        if self.action not in {AgentAction.SEARCH, AgentAction.ANSWER, AgentAction.STOP}:
            raise ValueError("decision traces only support search, answer, or stop")
        if any(not isinstance(document_id, str) or not document_id for document_id in self.cited_document_ids):
            raise ValueError("cited_document_ids must contain non-empty strings")
        for name in (
            "query_sha256",
            "normalized_query_sha256",
            "query_text",
            "answer_sha256",
            "answer_text",
            "reason_sha256",
            "reason_text",
        ):
            _optional_string(getattr(self, name), name)
        query_fields = (self.query_sha256, self.normalized_query_sha256, self.query_text)
        answer_fields = (self.answer_sha256, self.answer_text)
        if self.action is AgentAction.SEARCH:
            if any(value is not None for value in answer_fields):
                raise ValueError("search decision traces must not contain an answer")
            if self.cited_document_ids:
                raise ValueError("search decision traces must not contain citations")
        elif self.action is AgentAction.ANSWER:
            if any(value is not None for value in query_fields):
                raise ValueError("answer decision traces must not contain a query")
        else:
            if any(value is not None for value in (*query_fields, *answer_fields)):
                raise ValueError("stop decision traces must not contain a query or answer")
            if self.cited_document_ids:
                raise ValueError("stop decision traces must not contain citations")


@dataclass(frozen=True, slots=True)
class RoutingTrace:
    requested_route: str | None = None
    selected_route: str | None = None
    fallback_from: str | None = None
    fallback_reason: str | None = None
    signals: dict[str, float] | None = None

    def __post_init__(self) -> None:
        for name in ("requested_route", "selected_route", "fallback_from", "fallback_reason"):
            _optional_string(getattr(self, name), name)
        for key, value in (self.signals or {}).items():
            if not isinstance(key, str) or not key:
                raise ValueError("routing signal names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("routing signals must be finite numbers")


@dataclass(frozen=True, slots=True)
class CallTrace:
    call_id: str
    call_index: int
    kind: CallKind
    backend: str | None
    model: str | None
    started_offset_ms: float
    latency_ms: float
    status: CallStatus
    attempts: tuple[AttemptTrace, ...] = ()
    queue_latency_ms: float | None = None
    time_to_first_token_ms: float | None = None
    finish_reason: str | None = None
    cache_hit: bool | None = None
    tokens: TokenUsage | None = None
    cost: CostUsage | None = None
    retrieval: RetrievalPayload | None = None
    error: ErrorTrace | None = None

    def __post_init__(self) -> None:
        _nonempty_string(self.call_id, "call_id")
        if isinstance(self.call_index, bool) or not isinstance(self.call_index, int) or self.call_index < 0:
            raise ValueError("call_index must be a non-negative integer")
        if not isinstance(self.kind, CallKind):
            raise ValueError("kind must be a CallKind")
        if not isinstance(self.status, CallStatus):
            raise ValueError("status must be a CallStatus")
        for name in ("backend", "model", "finish_reason"):
            _optional_string(getattr(self, name), name)
        if self.cache_hit is not None and not isinstance(self.cache_hit, bool):
            raise ValueError("cache_hit must be a boolean or null")
        for name in ("started_offset_ms", "latency_ms", "queue_latency_ms", "time_to_first_token_ms"):
            _nonnegative_optional_float(getattr(self, name), name)
        if self.queue_latency_ms is not None and self.queue_latency_ms > self.latency_ms:
            raise ValueError("queue_latency_ms must not exceed latency_ms")
        if self.kind is CallKind.RETRIEVAL and self.retrieval is None:
            raise ValueError("retrieval calls require a retrieval payload")
        if self.kind is not CallKind.RETRIEVAL and self.retrieval is not None:
            raise ValueError("only retrieval calls may carry a retrieval payload")


@dataclass(frozen=True, slots=True)
class StepTrace:
    step_index: int
    started_offset_ms: float
    latency_ms: float
    action: AgentAction
    calls: tuple[CallTrace, ...]
    context_document_ids: tuple[str, ...]
    decision: DecisionTrace | None = None
    routing: RoutingTrace | None = None
    error: ErrorTrace | None = None

    def __post_init__(self) -> None:
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int) or self.step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        _nonnegative_optional_float(self.started_offset_ms, "started_offset_ms")
        _nonnegative_optional_float(self.latency_ms, "latency_ms")
        if not isinstance(self.action, AgentAction):
            raise ValueError("action must be an AgentAction")


@dataclass(frozen=True, slots=True)
class RunLimits:
    max_steps: int
    max_retrieval_calls: int
    max_llm_calls: int
    max_tokens: int | None
    top_k: int
    max_context_documents: int
    max_context_chars: int
    max_duration_ms: float

    def __post_init__(self) -> None:
        for name in (
            "max_steps",
            "max_retrieval_calls",
            "max_llm_calls",
            "top_k",
            "max_context_documents",
            "max_context_chars",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        _nonnegative_optional_int(self.max_tokens, "max_tokens")
        if self.max_tokens == 0:
            raise ValueError("max_tokens must be positive or null")
        _nonnegative_optional_float(self.max_duration_ms, "max_duration_ms")
        if self.max_duration_ms == 0:
            raise ValueError("max_duration_ms must be positive")


@dataclass(frozen=True, slots=True)
class RunTrace:
    schema_version: int
    run_id: str
    request_id: str | None
    experiment_id: str | None
    variant: str | None
    seed: int | None
    objective: AgentObjective
    config_fingerprint: str
    planner_config_fingerprint: str
    retrieval_config_fingerprint: str
    capture_text: TextCapturePolicy
    query_sha256: str | None
    normalized_query_sha256: str | None
    query_text: str | None
    started_at: str
    duration_ms: float
    queue_latency_ms: float
    execution_ms: float
    status: RunStatus
    termination_reason: TerminationReason
    budget_dimension: str | None
    limits: RunLimits
    steps: tuple[StepTrace, ...]
    final_context_document_ids: tuple[str, ...]
    answer_sha256: str | None
    answer_text: str | None
    final_context_chars: int = 0
    final_cited_document_ids: tuple[str, ...] = ()
    filters_sha256: str | None = None
    filters: dict[str, Any] | None = None
    error: ErrorTrace | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise ValueError("schema_version must be an integer")
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError(f"unsupported trace schema version: {self.schema_version}")
        for name in ("config_fingerprint", "planner_config_fingerprint", "retrieval_config_fingerprint"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.capture_text, TextCapturePolicy):
            raise ValueError("capture_text must be a TextCapturePolicy")
        _nonempty_string(self.run_id, "run_id")
        _nonempty_string(self.started_at, "started_at")
        for name in ("request_id", "experiment_id", "variant", "query_sha256", "normalized_query_sha256", "query_text"):
            _optional_string(getattr(self, name), name)
        for name in ("answer_sha256", "answer_text", "filters_sha256", "budget_dimension"):
            _optional_string(getattr(self, name), name)
        if not isinstance(self.status, RunStatus):
            raise ValueError("status must be a RunStatus")
        if not isinstance(self.termination_reason, TerminationReason):
            raise ValueError("termination_reason must be a TerminationReason")
        if not isinstance(self.limits, RunLimits):
            raise ValueError("limits must be RunLimits")
        for name in ("duration_ms", "queue_latency_ms", "execution_ms"):
            _nonnegative_optional_float(getattr(self, name), name)
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ValueError("seed must be an integer or null")
        if not isinstance(self.objective, AgentObjective):
            raise ValueError("objective must be an AgentObjective")
        if (
            isinstance(self.final_context_chars, bool)
            or not isinstance(self.final_context_chars, int)
            or self.final_context_chars < 0
        ):
            raise ValueError("final_context_chars must be a non-negative integer")
        if self.final_context_chars > self.limits.max_context_chars:
            raise ValueError("final_context_chars must not exceed limits.max_context_chars")
        if len(self.final_context_document_ids) > self.limits.max_context_documents:
            raise ValueError("final context documents must not exceed limits.max_context_documents")
        for field_name, document_ids in (
            ("final_context_document_ids", self.final_context_document_ids),
            ("final_cited_document_ids", self.final_cited_document_ids),
        ):
            if any(not isinstance(document_id, str) or not document_id for document_id in document_ids):
                raise ValueError(f"{field_name} must contain non-empty strings")
        if self.termination_reason is TerminationReason.ANSWERED and self.status is not RunStatus.COMPLETED:
            raise ValueError("answered traces must have completed status")
        if self.objective is AgentObjective.ANSWER:
            if self.termination_reason is TerminationReason.AGENT_STOP:
                raise ValueError("answer-objective traces must not terminate with agent_stop")
            if any(step.decision is not None and step.decision.action is AgentAction.STOP for step in self.steps):
                raise ValueError("answer-objective traces must not contain stop decisions")
        else:
            if self.termination_reason is TerminationReason.ANSWERED:
                raise ValueError("context-objective traces must not terminate with answered")
            if self.answer_sha256 is not None or self.answer_text is not None:
                raise ValueError("context-objective traces must not contain an answer")
            if self.final_cited_document_ids:
                raise ValueError("context-objective traces must not contain final citations")
            if any(step.decision is not None and step.decision.action is AgentAction.ANSWER for step in self.steps):
                raise ValueError("context-objective traces must not contain answer decisions")
        if (self.status is RunStatus.CANCELLED) != (self.termination_reason is TerminationReason.CANCELLED):
            raise ValueError("cancelled status and termination reason must agree")
        allowed_reasons = {
            RunStatus.COMPLETED: {
                TerminationReason.ANSWERED,
                TerminationReason.AGENT_STOP,
                TerminationReason.MAX_STEPS,
                TerminationReason.MAX_RETRIEVAL_CALLS,
                TerminationReason.MAX_LLM_CALLS,
                TerminationReason.MAX_TOKENS,
                TerminationReason.NO_PROGRESS,
                TerminationReason.REPEATED_QUERY,
            },
            RunStatus.FAILED: {
                TerminationReason.INVALID_ACTION,
                TerminationReason.PLANNER_ERROR,
                TerminationReason.RETRIEVAL_ERROR,
                TerminationReason.ERROR,
            },
            RunStatus.TIMED_OUT: {
                TerminationReason.MAX_DURATION,
                TerminationReason.TIMEOUT,
            },
            RunStatus.CANCELLED: {TerminationReason.CANCELLED},
        }
        if self.termination_reason not in allowed_reasons[self.status]:
            raise ValueError("status and termination_reason are inconsistent")

        raw_text_values: list[Any] = [self.query_text, self.answer_text, self.filters]
        hash_values: list[Any] = [
            self.query_sha256,
            self.normalized_query_sha256,
            self.answer_sha256,
            self.filters_sha256,
        ]
        for step in self.steps:
            if step.decision is not None:
                raw_text_values.extend(
                    (
                        step.decision.query_text,
                        step.decision.answer_text,
                        step.decision.reason_text,
                    )
                )
                hash_values.extend(
                    (
                        step.decision.query_sha256,
                        step.decision.normalized_query_sha256,
                        step.decision.answer_sha256,
                        step.decision.reason_sha256,
                    )
                )
            for call in step.calls:
                if call.retrieval is None:
                    continue
                raw_text_values.append(call.retrieval.query_text)
                hash_values.extend(
                    (
                        call.retrieval.query_sha256,
                        call.retrieval.normalized_query_sha256,
                    )
                )
                for document in call.retrieval.documents:
                    raw_text_values.append(document.content)
                    hash_values.append(document.content_sha256)
        if self.capture_text is not TextCapturePolicy.FULL and any(value is not None for value in raw_text_values):
            raise ValueError("raw trace text requires capture_text=full")
        if self.capture_text is TextCapturePolicy.NONE and any(value is not None for value in hash_values):
            raise ValueError("trace hashes require capture_text=hash or full")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)

    @classmethod
    def from_dict(cls, value: Any) -> RunTrace:
        """Decode one strict current-schema trace produced by :meth:`to_dict`."""

        from justatom.agentic.serialization import run_trace_from_dict

        return run_trace_from_dict(value)


@dataclass(frozen=True, slots=True)
class EvidenceDocument:
    document_id: str
    content: str
    score: float | None
    rank: int
    retrieval_index: int


@dataclass(frozen=True, slots=True)
class SearchObservation:
    query: str
    documents: tuple[EvidenceDocument, ...]


@dataclass(frozen=True, slots=True)
class PlannerRequest:
    question: str
    observations: tuple[SearchObservation, ...]
    context_documents: tuple[EvidenceDocument, ...]
    remaining_retrieval_calls: int
    remaining_steps: int
    objective: AgentObjective = AgentObjective.ANSWER

    def __post_init__(self) -> None:
        if not isinstance(self.objective, AgentObjective):
            raise ValueError("objective must be an AgentObjective")


@dataclass(frozen=True, slots=True)
class PlannerDecision:
    action: AgentAction
    query: str | None = None
    answer: str | None = None
    reason: str | None = None
    cited_document_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.action is AgentAction.SEARCH:
            if not isinstance(self.query, str) or not self.query.strip():
                raise ValueError("search decisions require a non-empty query")
            if self.answer is not None:
                raise ValueError("search decisions must not include an answer")
            if self.cited_document_ids:
                raise ValueError("search decisions must not include citations")
        elif self.action is AgentAction.ANSWER:
            if not isinstance(self.answer, str) or not self.answer.strip():
                raise ValueError("answer decisions require a non-empty answer")
            if self.query is not None:
                raise ValueError("answer decisions must not include a query")
        elif self.action is AgentAction.STOP:
            if self.query is not None:
                raise ValueError("stop decisions must not include a query")
            if self.answer is not None:
                raise ValueError("stop decisions must not include an answer")
            if self.cited_document_ids:
                raise ValueError("stop decisions must not include citations")
        else:
            raise ValueError("planner decisions only support search, answer, or stop")
        if any(not isinstance(document_id, str) or not document_id.strip() for document_id in self.cited_document_ids):
            raise ValueError("cited_document_ids must contain non-empty strings")


@dataclass(frozen=True, slots=True)
class PlannerReply:
    decision: PlannerDecision
    model: str | None = None
    provider_request_id: str | None = None
    usage: TokenUsage | None = None
    cost: CostUsage | None = None
    attempts: tuple[AttemptTrace, ...] = ()
    cache_hit: bool | None = None
    time_to_first_token_ms: float | None = None
    finish_reason: str | None = None


__all__ = [
    "TRACE_SCHEMA_VERSION",
    "AgentAction",
    "AgentObjective",
    "AttemptTrace",
    "CallKind",
    "CallStatus",
    "CallTrace",
    "CostUsage",
    "DecisionTrace",
    "DocumentTrace",
    "ErrorCategory",
    "ErrorTrace",
    "EvidenceDocument",
    "PlannerDecision",
    "PlannerReply",
    "PlannerRequest",
    "RetrievalPayload",
    "RoutingTrace",
    "RunLimits",
    "RunStatus",
    "RunTrace",
    "SearchObservation",
    "StepTrace",
    "TerminationReason",
    "TextCapturePolicy",
    "TokenUsage",
    "normalize_query",
    "normalized_query_sha256",
    "sha256_text",
]
