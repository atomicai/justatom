"""Export compact common records from optional Inspect benchmark logs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from justatom.agentic.benchmark import score_searches

if TYPE_CHECKING:
    from justatom.agentic.benchmark_data import BenchmarkCase


@dataclass(frozen=True, slots=True)
class _ExpectedCase:
    query_id: str
    query: str | None
    gold_ids: tuple[str, ...] | None
    metadata: dict[str, Any]


def export_results(
    logs: Sequence[Any],
    output: str | Path,
    *,
    cases: Sequence[BenchmarkCase] | None = None,
    methods: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write denominator-complete common records without importing Inspect.

    Supplying ``cases`` and ``methods`` is recommended because it permits rows
    for methods or samples absent from fatal or cancelled logs. When omitted,
    the exporter falls back to method metadata and dataset sample IDs retained
    in the logs.
    """

    materialized_logs = _sequence(logs, "logs")
    output_path = Path(output)
    if not output_path.is_dir():
        raise ValueError("output must be an existing directory")

    resolved_methods = _resolve_methods(materialized_logs, methods)
    expected_cases = _resolve_cases(materialized_logs, cases)
    if not resolved_methods:
        raise ValueError("methods must not be empty")
    if not expected_cases:
        raise ValueError("cases must not be empty")

    log_configs: dict[str, dict[str, Any]] = {}
    log_errors: dict[str, Any] = {}
    observed: dict[tuple[str, str], Any] = {}
    expected_ids = {case.query_id for case in expected_cases}
    expected_methods = set(resolved_methods)
    for log in materialized_logs:
        method = _log_method(log)
        if method not in expected_methods:
            raise ValueError(f"unexpected log method {method!r}")
        log_configs.setdefault(method, _log_config(log))
        if getattr(log, "error", None) is not None:
            log_errors.setdefault(method, getattr(log, "error"))
        for sample in getattr(log, "samples", None) or ():
            query_id = str(getattr(sample, "id", ""))
            if query_id not in expected_ids:
                raise ValueError(f"unexpected sample query_id {query_id!r}")
            key = (method, query_id)
            if key in observed:
                raise ValueError(f"duplicate sample for method={method!r}, query_id={query_id!r}")
            observed[key] = sample

    rows: list[dict[str, Any]] = []
    for method in resolved_methods:
        for case in expected_cases:
            rows.append(
                _row(
                    method,
                    case,
                    observed.get((method, case.query_id)),
                    log_configs.get(method, {}),
                    log_errors.get(method),
                )
            )

    summary = _summary(rows)
    results_text = "".join(_json(row) + "\n" for row in rows)
    summary_text = json.dumps(summary, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    _exclusive_pair_write(output_path / "results.jsonl", results_text, output_path / "summary.json", summary_text)
    return summary


def _resolve_methods(logs: Sequence[Any], methods: Sequence[str] | None) -> list[str]:
    if methods is None:
        resolved: list[str] = []
        for log in logs:
            method = _log_method(log)
            if method not in resolved:
                resolved.append(method)
        return resolved
    values = _sequence(methods, "methods")
    if any(not isinstance(method, str) or not method.strip() for method in values):
        raise ValueError("methods must contain only non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError("methods must be unique")
    return list(values)


def _resolve_cases(logs: Sequence[Any], cases: Sequence[BenchmarkCase] | None) -> list[_ExpectedCase]:
    if cases is not None:
        values = _sequence(cases, "cases")
        resolved = [
            _ExpectedCase(
                query_id=case.query_id,
                query=case.query,
                gold_ids=tuple(case.chunk_ids),
                metadata=dict(case.metadata),
            )
            for case in values
        ]
        if len({case.query_id for case in resolved}) != len(resolved):
            raise ValueError("cases must have unique query IDs")
        return resolved

    ordered_ids: list[str] = []
    details: dict[str, tuple[str | None, dict[str, Any]]] = {}
    for log in logs:
        dataset = getattr(getattr(log, "eval", None), "dataset", None)
        for raw_id in getattr(dataset, "sample_ids", None) or ():
            query_id = str(raw_id)
            if query_id not in ordered_ids:
                ordered_ids.append(query_id)
        for sample in getattr(log, "samples", None) or ():
            query_id = str(getattr(sample, "id", ""))
            if query_id not in ordered_ids:
                ordered_ids.append(query_id)
            metadata = getattr(sample, "metadata", None)
            metadata = metadata if isinstance(metadata, Mapping) else {}
            record = metadata.get("justatom")
            record = record if isinstance(record, Mapping) else {}
            query = record.get("query")
            if not isinstance(query, str):
                sample_input = getattr(sample, "input", None)
                query = sample_input if isinstance(sample_input, str) else None
            public_metadata = {str(key): value for key, value in metadata.items() if key != "justatom"}
            details.setdefault(query_id, (query, public_metadata))
    return [
        _ExpectedCase(query_id, details.get(query_id, (None, {}))[0], None, details.get(query_id, (None, {}))[1])
        for query_id in ordered_ids
    ]


def _row(
    method: str,
    case: _ExpectedCase,
    sample: Any | None,
    log_config: Mapping[str, Any],
    log_error: Any | None,
) -> dict[str, Any]:
    if sample is None:
        evaluation = _empty_evaluation(len(case.gold_ids or ()))
        return {
            "schema_version": 1,
            "query_id": case.query_id,
            "query": case.query,
            "method": method,
            "metadata": dict(case.metadata),
            "config": dict(log_config),
            "status": "missing",
            "error": _error_payload(log_error)
            or {
                "type": "MissingSample",
                "message": "sample not present in Inspect log",
            },
            "termination_reason": None,
            "duration_ms": None,
            "model_calls": None,
            "limit": None,
            "calls": [],
            "context_ids": [],
            "context_documents": [],
            "evaluation": evaluation,
            "usage": _unknown_usage(),
            "native_trace": None,
        }

    metadata = getattr(sample, "metadata", None)
    metadata = metadata if isinstance(metadata, Mapping) else {}
    record_value = metadata.get("justatom")
    record = record_value if isinstance(record_value, Mapping) else {}
    calls = _record_list(record, "calls")
    context_ids = _record_list(record, "context_ids")
    context_documents = _record_list(record, "context_documents")
    if case.gold_ids is not None:
        evaluation = score_searches(calls, context_ids, case.gold_ids)
    else:
        evaluation = _stored_evaluation(record)

    sample_error = getattr(sample, "error", None)
    limit = _limit_payload(getattr(sample, "limit", None))
    incomplete = not record
    terminal_error = _terminal_error(method, record, limit)
    status = "error" if sample_error is not None or incomplete or terminal_error is not None else "success"
    error = _error_payload(sample_error) or terminal_error
    if incomplete and error is None:
        error = {"type": "IncompleteSample", "message": "missing justatom sample record"}
    query = case.query
    if query is None:
        record_query = record.get("query")
        query = record_query if isinstance(record_query, str) else None
    config = record.get("config")
    config = dict(config) if isinstance(config, Mapping) else dict(log_config)
    native_trace = record.get("native_trace") if method == "native" else None
    return {
        "schema_version": 1,
        "query_id": case.query_id,
        "query": query,
        "method": method,
        "metadata": dict(case.metadata),
        "config": config,
        "status": status,
        "error": error,
        "termination_reason": record.get("termination_reason"),
        "duration_ms": record.get("duration_ms"),
        "model_calls": record.get("model_calls"),
        "limit": limit,
        "calls": calls,
        "context_ids": context_ids,
        "context_documents": context_documents,
        "evaluation": evaluation,
        "usage": _usage(method, record, sample),
        "native_trace": native_trace,
    }


def _usage(method: str, record: Mapping[str, Any], sample: Any) -> dict[str, Any]:
    if method == "native":
        metrics = record.get("native_metrics")
        if isinstance(metrics, Mapping):
            return _native_metrics_usage(metrics)
        trace = record.get("native_trace")
        if isinstance(trace, Mapping):
            return _native_trace_usage(trace)
        return _unknown_usage()

    return _react_usage(sample)


def _react_usage(sample: Any) -> dict[str, Any]:
    events = getattr(sample, "events", None)
    model_events = [event for event in events or () if getattr(event, "event", None) == "model"]
    if model_events:
        return _react_event_usage(model_events)

    raw_usage = getattr(sample, "model_usage", None)
    if not isinstance(raw_usage, Mapping) or not raw_usage:
        usage = _unknown_usage()
        usage["source"] = "inspect_model_usage"
        return usage
    values = list(raw_usage.values())
    observed_totals = {
        "input_tokens": _sum_usage_field(values, "input_tokens", integer=True),
        "output_tokens": _sum_usage_field(values, "output_tokens", integer=True),
        "total_tokens": _sum_usage_field(values, "total_tokens", integer=True),
        "cached_input_tokens": _sum_usage_field(values, "input_tokens_cache_read", integer=True),
        "reasoning_tokens": _sum_usage_field(values, "reasoning_tokens", integer=True),
    }
    observed_cost = None
    return {
        "source": "inspect_model_usage",
        **observed_totals,
        "cost_usd": observed_cost,
        "observed_token_totals": observed_totals,
        "token_coverage": None,
        "token_usage_coverage": None,
        "observed_cost_usd": observed_cost,
        "cost_coverage": None,
        "observed_only": True,
    }


def _react_event_usage(events: Sequence[Any]) -> dict[str, Any]:
    values = [getattr(getattr(event, "output", None), "usage", None) for event in events]
    field_names = {
        "input_tokens": "input_tokens",
        "output_tokens": "output_tokens",
        "total_tokens": "total_tokens",
        "cached_input_tokens": "input_tokens_cache_read",
        "reasoning_tokens": "reasoning_tokens",
    }
    observed_totals: dict[str, int | None] = {}
    token_coverage: dict[str, dict[str, int | float]] = {}
    complete_totals: dict[str, int | None] = {}
    for common_name, source_name in field_names.items():
        observed, coverage = _event_field(values, source_name, integer=True)
        observed_totals[common_name] = observed if isinstance(observed, int) else None
        token_coverage[common_name] = coverage
        complete_totals[common_name] = observed_totals[common_name] if _coverage_complete(coverage) else None
    costs = [_react_event_cost(event) for event in events]
    observed_cost, cost_coverage = _values_total(costs, integer=False)
    complete_cost = observed_cost if _coverage_complete(cost_coverage) else None
    usage_coverage = _coverage(sum(value is not None for value in values), len(values))
    observed_only = any(observed_totals[field] is not None and complete_totals[field] is None for field in observed_totals) or (
        observed_cost is not None and complete_cost is None
    )
    return {
        "source": "inspect_model_events",
        **complete_totals,
        "cost_usd": complete_cost,
        "observed_token_totals": observed_totals,
        "token_coverage": token_coverage,
        "token_usage_coverage": usage_coverage,
        "observed_cost_usd": observed_cost,
        "cost_coverage": cost_coverage,
        "observed_only": observed_only,
    }


def _react_event_cost(event: Any) -> float | int | None:
    call = getattr(event, "call", None)
    response = call.get("response") if isinstance(call, Mapping) else getattr(call, "response", None)
    if not isinstance(response, Mapping):
        return None
    provider_usage = response.get("usage")
    return provider_usage.get("cost") if isinstance(provider_usage, Mapping) else None


def _event_field(values: Sequence[Any], field: str, *, integer: bool) -> tuple[int | float | None, dict[str, int | float]]:
    extracted = [getattr(value, field, None) if value is not None else None for value in values]
    return _values_total(extracted, integer=integer)


def _values_total(extracted: Sequence[Any], *, integer: bool) -> tuple[int | float | None, dict[str, int | float]]:
    if integer:
        known = [value for value in extracted if isinstance(value, int) and not isinstance(value, bool) and value >= 0]
        observed: int | float | None = sum(known) if known else None
    else:
        known = [
            value
            for value in extracted
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
        ]
        total = math.fsum(float(value) for value in known)
        observed = total if known and math.isfinite(total) else None
    return observed, _coverage(len(known), len(extracted))


def _coverage(numerator: int, denominator: int) -> dict[str, int | float]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else 0.0,
    }


def _native_metrics_usage(metrics: Mapping[str, Any]) -> dict[str, Any]:
    raw_totals = metrics.get("token_totals")
    totals = raw_totals if isinstance(raw_totals, Mapping) else {}
    raw_token_coverage = metrics.get("token_coverage")
    token_coverage = dict(raw_token_coverage) if isinstance(raw_token_coverage, Mapping) else None
    raw_usage_coverage = metrics.get("token_usage_coverage")
    usage_coverage = dict(raw_usage_coverage) if isinstance(raw_usage_coverage, Mapping) else None
    raw_cost_coverage = metrics.get("cost_coverage")
    cost_coverage = dict(raw_cost_coverage) if isinstance(raw_cost_coverage, Mapping) else None
    observed_totals = {
        field: _optional_nonnegative_int(totals.get(field))
        for field in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens")
    }
    complete_totals = {field: _covered_total(observed_totals[field], token_coverage, field) for field in observed_totals}
    observed_cost = _optional_nonnegative_float(metrics.get("cost_total_usd"))
    complete_cost = observed_cost if _coverage_complete(cost_coverage) else None
    observed_only = any(observed_totals[field] is not None and complete_totals[field] is None for field in observed_totals) or (
        observed_cost is not None and complete_cost is None
    )
    return {
        "source": "native_metrics",
        **complete_totals,
        "cost_usd": complete_cost,
        "observed_token_totals": observed_totals,
        "token_coverage": token_coverage,
        "token_usage_coverage": usage_coverage,
        "observed_cost_usd": observed_cost,
        "cost_coverage": cost_coverage,
        "observed_only": observed_only,
    }


def _native_trace_usage(trace: Mapping[str, Any]) -> dict[str, Any]:
    planner_calls: list[Mapping[str, Any]] = []
    raw_steps = trace.get("steps", ())
    steps = raw_steps if isinstance(raw_steps, Sequence) and not isinstance(raw_steps, (str, bytes)) else ()
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        raw_calls = step.get("calls", ())
        calls = raw_calls if isinstance(raw_calls, Sequence) and not isinstance(raw_calls, (str, bytes)) else ()
        planner_calls.extend(call for call in calls if isinstance(call, Mapping) and call.get("kind") == "planner")
    token_maps = [call.get("tokens") for call in planner_calls]
    if not planner_calls or any(not isinstance(tokens, Mapping) for tokens in token_maps):
        usage = _unknown_usage()
        usage["source"] = "native_trace"
        return usage
    validated_tokens = [tokens for tokens in token_maps if isinstance(tokens, Mapping)]
    return {
        "source": "native_trace",
        "input_tokens": _sum_mapping_field(validated_tokens, "input_tokens"),
        "output_tokens": _sum_mapping_field(validated_tokens, "output_tokens"),
        "total_tokens": _sum_mapping_field(validated_tokens, "total_tokens"),
        "cached_input_tokens": _sum_mapping_field(validated_tokens, "cached_input_tokens"),
        "reasoning_tokens": _sum_mapping_field(validated_tokens, "reasoning_tokens"),
        "cost_usd": None,
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[Mapping[str, Any]]] = {}
    by_domain: dict[str, list[Mapping[str, Any]]] = {}
    by_depth: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
        metadata = row.get("metadata")
        if isinstance(metadata, Mapping):
            if metadata.get("domain") is not None:
                by_domain.setdefault(str(metadata["domain"]), []).append(row)
            if metadata.get("depth") is not None:
                by_depth.setdefault(str(metadata["depth"]), []).append(row)
    return {
        "schema_version": 1,
        "overall": _aggregate(rows),
        "by_method": {key: _aggregate(value) for key, value in by_method.items()},
        "by_domain": {key: _aggregate(value) for key, value in by_domain.items()},
        "by_depth": {key: _aggregate(value) for key, value in by_depth.items()},
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    requested = len(rows)
    all_gold = 0
    recall = 0.0
    for row in rows:
        final_context = row["evaluation"]["final_context"]
        all_gold += int(bool(final_context["all_gold"]))
        recall += float(final_context["recall"])
    return {
        "requested": requested,
        "successes": sum(row.get("status") == "success" for row in rows),
        "errors": sum(row.get("status") == "error" for row in rows),
        "missing": sum(row.get("status") == "missing" for row in rows),
        "all_gold_rate": all_gold / requested if requested else 0.0,
        "mean_recall": recall / requested if requested else 0.0,
    }


def _stored_evaluation(record: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = record.get("evaluation")
    return dict(evaluation) if isinstance(evaluation, Mapping) else _empty_evaluation(0)


def _empty_evaluation(gold_count: int) -> dict[str, Any]:
    metric = {
        "all_gold": False,
        "recall": 0.0,
        "hits": 0,
        "gold_count": gold_count,
        "retrieved_count": 0,
    }
    return {
        "per_hop": [],
        "cumulative": [],
        "final_context": metric,
        "first_complete_hop": None,
        "search_calls": 0,
    }


def _unknown_usage() -> dict[str, Any]:
    return {
        "source": "unavailable",
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "cached_input_tokens": None,
        "reasoning_tokens": None,
        "cost_usd": None,
    }


def _log_method(log: Any) -> str:
    for metadata in (getattr(log, "metadata", None), getattr(getattr(log, "eval", None), "metadata", None)):
        if isinstance(metadata, Mapping):
            method = metadata.get("method")
            if isinstance(method, str) and method.strip():
                return method
    raise ValueError("Inspect log is missing method metadata")


def _log_config(log: Any) -> dict[str, Any]:
    for metadata in (getattr(log, "metadata", None), getattr(getattr(log, "eval", None), "metadata", None)):
        if isinstance(metadata, Mapping):
            return dict(metadata)
    return {}


def _record_list(record: Mapping[str, Any], field: str) -> list[Any]:
    value = record.get(field, [])
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"sample record {field} must be a sequence")
    return list(value)


def _error_payload(error: Any | None) -> dict[str, str] | None:
    if error is None:
        return None
    message = getattr(error, "message", None)
    return {
        "type": type(error).__name__,
        "message": message if isinstance(message, str) else str(error),
    }


def _limit_payload(limit: Any | None) -> dict[str, Any] | None:
    if limit is None:
        return None
    if isinstance(limit, Mapping):
        return dict(limit)
    model_dump = getattr(limit, "model_dump", None)
    if callable(model_dump):
        value = model_dump()
        return dict(value) if isinstance(value, Mapping) else None
    limit_type = getattr(limit, "type", None)
    limit_value = getattr(limit, "limit", None)
    if limit_type is None and limit_value is None:
        return None
    return {"type": limit_type, "limit": limit_value, "reason": getattr(limit, "reason", None)}


def _terminal_error(method: str, record: Mapping[str, Any], limit: Mapping[str, Any] | None) -> dict[str, str] | None:
    if limit is not None and limit.get("type") in {"time", "working"}:
        return {"type": "InspectLimit", "message": f"{limit['type']} limit reached"}
    if method == "native":
        trace = record.get("native_trace")
        native_status = trace.get("status") if isinstance(trace, Mapping) else None
        if native_status in {"failed", "timed_out", "cancelled"}:
            return {"type": "NativeRunStatus", "message": str(native_status)}
    termination = record.get("termination_reason")
    if termination in {"provider_max_tokens", "provider_model_length"}:
        return {"type": "ProviderTruncation", "message": str(termination).removeprefix("provider_")}
    if termination in {"provider_content_filter", "provider_refusal"}:
        return {"type": "ProviderRefusal", "message": str(termination).removeprefix("provider_")}
    if isinstance(termination, str) and termination.startswith("provider_"):
        return {"type": "ProviderInvalidTermination", "message": termination.removeprefix("provider_")}
    if isinstance(termination, str) and (
        termination in {"error", "timeout", "timed_out", "cancelled"} or termination.endswith("_error")
    ):
        return {"type": "TerminationReason", "message": termination}
    return None


def _sum_usage_field(values: Sequence[Any], field: str, *, integer: bool) -> int | float | None:
    extracted = [getattr(value, field, None) if not isinstance(value, Mapping) else value.get(field) for value in values]
    if any(value is None or isinstance(value, bool) for value in extracted):
        return None
    if integer:
        if any(not isinstance(value, int) or value < 0 for value in extracted):
            return None
        return sum(extracted)
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in extracted):
        return None
    total = math.fsum(float(value) for value in extracted)
    return total if math.isfinite(total) else None


def _sum_mapping_field(values: Sequence[Mapping[str, Any]], field: str) -> int | None:
    extracted = [_optional_nonnegative_int(value.get(field)) for value in values]
    return sum(extracted) if all(value is not None for value in extracted) else None


def _covered_total(value: int | None, coverage_by_field: Mapping[str, Any] | None, field: str) -> int | None:
    if value is None:
        return None
    if coverage_by_field is None or field not in coverage_by_field:
        return value
    coverage = coverage_by_field.get(field)
    return value if isinstance(coverage, Mapping) and _coverage_complete(coverage) else None


def _coverage_complete(coverage: Mapping[str, Any] | None) -> bool:
    if coverage is None:
        return True
    numerator = coverage.get("numerator")
    denominator = coverage.get("denominator")
    return (
        isinstance(numerator, int)
        and not isinstance(numerator, bool)
        and isinstance(denominator, int)
        and not isinstance(denominator, bool)
        and denominator > 0
        and numerator == denominator
    )


def _optional_nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _optional_nonnegative_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


def _sequence(value: Any, name: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return list(value)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _exclusive_pair_write(results_path: Path, results_text: str, summary_path: Path, summary_text: str) -> None:
    if results_path.exists():
        raise FileExistsError(results_path)
    if summary_path.exists():
        raise FileExistsError(summary_path)
    created: list[Path] = []
    try:
        with results_path.open("x", encoding="utf-8") as stream:
            created.append(results_path)
            stream.write(results_text)
        with summary_path.open("x", encoding="utf-8") as stream:
            created.append(summary_path)
            stream.write(summary_text)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise


__all__ = ["export_results"]
