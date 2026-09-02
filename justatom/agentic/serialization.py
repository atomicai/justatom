from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
from enum import Enum
from typing import Any, TypeVar

from justatom.agentic.schemas import (
    AgentAction,
    AttemptTrace,
    CallKind,
    CallStatus,
    CallTrace,
    CostUsage,
    DecisionTrace,
    DocumentTrace,
    ErrorCategory,
    ErrorTrace,
    RetrievalPayload,
    RoutingTrace,
    RunLimits,
    RunStatus,
    RunTrace,
    StepTrace,
    TerminationReason,
    TextCapturePolicy,
    TokenUsage,
)

_EnumT = TypeVar("_EnumT", bound=Enum)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _optional_mapping(value: Any, name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _mapping(value, name)


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    return value


def _required(value: Mapping[str, Any], key: str, name: str) -> Any:
    if key not in value:
        raise ValueError(f"{name}.{key} is required")
    return value[key]


def _reject_unknown(value: Mapping[str, Any], schema: type[Any], name: str) -> None:
    allowed = {field.name for field in fields(schema)}
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {', '.join(unknown)}")


def _enum(enum_type: type[_EnumT], value: Any, name: str) -> _EnumT:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is invalid") from error


def _error(value: Any, name: str) -> ErrorTrace | None:
    data = _optional_mapping(value, name)
    if data is None:
        return None
    _reject_unknown(data, ErrorTrace, name)
    return ErrorTrace(
        component=_required(data, "component", name),
        category=_enum(ErrorCategory, _required(data, "category", name), f"{name}.category"),
        code=_required(data, "code", name),
        exception_type=data.get("exception_type"),
        retryable=data.get("retryable", False),
    )


def _attempt(value: Any, name: str) -> AttemptTrace:
    data = _mapping(value, name)
    _reject_unknown(data, AttemptTrace, name)
    return AttemptTrace(
        attempt_index=_required(data, "attempt_index", name),
        status=_enum(CallStatus, _required(data, "status", name), f"{name}.status"),
        latency_ms=_required(data, "latency_ms", name),
        provider_request_id=data.get("provider_request_id"),
        error=_error(data.get("error"), f"{name}.error"),
    )


def _document(value: Any, name: str) -> DocumentTrace:
    data = _mapping(value, name)
    _reject_unknown(data, DocumentTrace, name)
    return DocumentTrace(
        document_id=_required(data, "document_id", name),
        rank=_required(data, "rank", name),
        score=_required(data, "score", name),
        content_chars=_required(data, "content_chars", name),
        content_sha256=_required(data, "content_sha256", name),
        content=data.get("content"),
    )


def _retrieval(value: Any, name: str) -> RetrievalPayload | None:
    data = _optional_mapping(value, name)
    if data is None:
        return None
    _reject_unknown(data, RetrievalPayload, name)
    documents = _sequence(_required(data, "documents", name), f"{name}.documents")
    return RetrievalPayload(
        retrieval_index=_required(data, "retrieval_index", name),
        query_sha256=_required(data, "query_sha256", name),
        normalized_query_sha256=_required(data, "normalized_query_sha256", name),
        query_text=_required(data, "query_text", name),
        mode=_required(data, "mode", name),
        collection=_required(data, "collection", name),
        index_revision=_required(data, "index_revision", name),
        top_k_requested=_required(data, "top_k_requested", name),
        documents=tuple(_document(document, f"{name}.documents[{index}]") for index, document in enumerate(documents)),
        backend_document_count=data.get("backend_document_count"),
        truncated_document_count=data.get("truncated_document_count", 0),
    )


def _tokens(value: Any, name: str) -> TokenUsage | None:
    data = _optional_mapping(value, name)
    if data is None:
        return None
    _reject_unknown(data, TokenUsage, name)
    return TokenUsage(
        input_tokens=data.get("input_tokens"),
        output_tokens=data.get("output_tokens"),
        total_tokens=data.get("total_tokens"),
        cached_input_tokens=data.get("cached_input_tokens"),
        reasoning_tokens=data.get("reasoning_tokens"),
        source=data.get("source", "unknown"),
    )


def _cost(value: Any, name: str) -> CostUsage | None:
    data = _optional_mapping(value, name)
    if data is None:
        return None
    _reject_unknown(data, CostUsage, name)
    return CostUsage(
        usd=data.get("usd"),
        source=data.get("source", "unavailable"),
        pricing_id=data.get("pricing_id"),
    )


def _call(value: Any, name: str) -> CallTrace:
    data = _mapping(value, name)
    _reject_unknown(data, CallTrace, name)
    attempts = _sequence(data.get("attempts", ()), f"{name}.attempts")
    return CallTrace(
        call_id=_required(data, "call_id", name),
        call_index=_required(data, "call_index", name),
        kind=_enum(CallKind, _required(data, "kind", name), f"{name}.kind"),
        backend=_required(data, "backend", name),
        model=_required(data, "model", name),
        started_offset_ms=_required(data, "started_offset_ms", name),
        latency_ms=_required(data, "latency_ms", name),
        status=_enum(CallStatus, _required(data, "status", name), f"{name}.status"),
        attempts=tuple(_attempt(attempt, f"{name}.attempts[{index}]") for index, attempt in enumerate(attempts)),
        queue_latency_ms=data.get("queue_latency_ms"),
        time_to_first_token_ms=data.get("time_to_first_token_ms"),
        finish_reason=data.get("finish_reason"),
        cache_hit=data.get("cache_hit"),
        tokens=_tokens(data.get("tokens"), f"{name}.tokens"),
        cost=_cost(data.get("cost"), f"{name}.cost"),
        retrieval=_retrieval(data.get("retrieval"), f"{name}.retrieval"),
        error=_error(data.get("error"), f"{name}.error"),
    )


def _routing(value: Any, name: str) -> RoutingTrace | None:
    data = _optional_mapping(value, name)
    if data is None:
        return None
    _reject_unknown(data, RoutingTrace, name)
    raw_signals = data.get("signals")
    signals = None if raw_signals is None else dict(_mapping(raw_signals, f"{name}.signals"))
    return RoutingTrace(
        requested_route=data.get("requested_route"),
        selected_route=data.get("selected_route"),
        fallback_from=data.get("fallback_from"),
        fallback_reason=data.get("fallback_reason"),
        signals=signals,
    )


def _decision(value: Any, name: str) -> DecisionTrace | None:
    data = _optional_mapping(value, name)
    if data is None:
        return None
    _reject_unknown(data, DecisionTrace, name)
    cited_ids = _sequence(data.get("cited_document_ids", ()), f"{name}.cited_document_ids")
    return DecisionTrace(
        action=_enum(AgentAction, _required(data, "action", name), f"{name}.action"),
        query_sha256=data.get("query_sha256"),
        normalized_query_sha256=data.get("normalized_query_sha256"),
        query_text=data.get("query_text"),
        answer_sha256=data.get("answer_sha256"),
        answer_text=data.get("answer_text"),
        reason_sha256=data.get("reason_sha256"),
        reason_text=data.get("reason_text"),
        cited_document_ids=tuple(cited_ids),
    )


def _step(value: Any, name: str) -> StepTrace:
    data = _mapping(value, name)
    _reject_unknown(data, StepTrace, name)
    calls = _sequence(_required(data, "calls", name), f"{name}.calls")
    context_ids = _sequence(_required(data, "context_document_ids", name), f"{name}.context_document_ids")
    return StepTrace(
        step_index=_required(data, "step_index", name),
        started_offset_ms=_required(data, "started_offset_ms", name),
        latency_ms=_required(data, "latency_ms", name),
        action=_enum(AgentAction, _required(data, "action", name), f"{name}.action"),
        calls=tuple(_call(call, f"{name}.calls[{index}]") for index, call in enumerate(calls)),
        context_document_ids=tuple(context_ids),
        decision=_decision(data.get("decision"), f"{name}.decision"),
        routing=_routing(data.get("routing"), f"{name}.routing"),
        error=_error(data.get("error"), f"{name}.error"),
    )


def _limits(value: Any, name: str) -> RunLimits:
    data = _mapping(value, name)
    _reject_unknown(data, RunLimits, name)
    return RunLimits(
        max_steps=_required(data, "max_steps", name),
        max_retrieval_calls=_required(data, "max_retrieval_calls", name),
        max_llm_calls=_required(data, "max_llm_calls", name),
        max_tokens=_required(data, "max_tokens", name),
        max_duration_ms=_required(data, "max_duration_ms", name),
    )


def run_trace_from_dict(value: Any) -> RunTrace:
    """Decode a JSON-compatible schema-v1 mapping into immutable trace objects."""

    data = _mapping(value, "trace")
    _reject_unknown(data, RunTrace, "trace")
    steps = _sequence(_required(data, "steps", "trace"), "trace.steps")
    context_ids = _sequence(_required(data, "final_context_document_ids", "trace"), "trace.final_context_document_ids")
    cited_ids = _sequence(data.get("final_cited_document_ids", ()), "trace.final_cited_document_ids")
    metadata = data.get("metadata")
    filters = data.get("filters")
    return RunTrace(
        schema_version=_required(data, "schema_version", "trace"),
        run_id=_required(data, "run_id", "trace"),
        request_id=_required(data, "request_id", "trace"),
        experiment_id=_required(data, "experiment_id", "trace"),
        variant=_required(data, "variant", "trace"),
        seed=_required(data, "seed", "trace"),
        config_fingerprint=_required(data, "config_fingerprint", "trace"),
        planner_config_fingerprint=_required(data, "planner_config_fingerprint", "trace"),
        retrieval_config_fingerprint=_required(data, "retrieval_config_fingerprint", "trace"),
        capture_text=_enum(TextCapturePolicy, _required(data, "capture_text", "trace"), "trace.capture_text"),
        query_sha256=_required(data, "query_sha256", "trace"),
        normalized_query_sha256=_required(data, "normalized_query_sha256", "trace"),
        query_text=_required(data, "query_text", "trace"),
        started_at=_required(data, "started_at", "trace"),
        duration_ms=_required(data, "duration_ms", "trace"),
        queue_latency_ms=_required(data, "queue_latency_ms", "trace"),
        execution_ms=_required(data, "execution_ms", "trace"),
        status=_enum(RunStatus, _required(data, "status", "trace"), "trace.status"),
        termination_reason=_enum(
            TerminationReason,
            _required(data, "termination_reason", "trace"),
            "trace.termination_reason",
        ),
        budget_dimension=_required(data, "budget_dimension", "trace"),
        limits=_limits(_required(data, "limits", "trace"), "trace.limits"),
        steps=tuple(_step(step, f"trace.steps[{index}]") for index, step in enumerate(steps)),
        final_context_document_ids=tuple(context_ids),
        answer_sha256=_required(data, "answer_sha256", "trace"),
        answer_text=_required(data, "answer_text", "trace"),
        final_context_chars=data.get("final_context_chars", 0),
        final_cited_document_ids=tuple(cited_ids),
        filters_sha256=data.get("filters_sha256"),
        filters=None if filters is None else dict(_mapping(filters, "trace.filters")),
        error=_error(data.get("error"), "trace.error"),
        metadata=None if metadata is None else dict(_mapping(metadata, "trace.metadata")),
    )


__all__ = ["run_trace_from_dict"]
