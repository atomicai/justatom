from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai.log import EvalError, EvalLog, EvalSample, EvalSampleLimit  # noqa: E402
from inspect_ai.model import ModelUsage  # noqa: E402

from justatom.agentic.benchmark_data import BenchmarkCase
from justatom.agentic.inspect_results import export_results


def _record(
    query_id: str,
    query: str,
    method: str,
    *,
    context_ids: list[str],
    native_metrics: dict[str, Any] | None = None,
    native_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    calls = [
        {
            "index": 1,
            "query": query,
            "status": "ok",
            "latency_ms": 1.5,
            "documents": [
                {"chunk_id": chunk_id, "rank": rank, "score": 1.0 / rank} for rank, chunk_id in enumerate(context_ids, start=1)
            ],
            "error_type": None,
        }
    ]
    record: dict[str, Any] = {
        "schema_version": 1,
        "query_id": query_id,
        "query": query,
        "method": method,
        "config": {"method": method, "budget": {"top_k": 5}},
        "calls": calls,
        "context_ids": list(context_ids),
        "context_documents": [{"chunk_id": chunk_id, "content": f"text {chunk_id}"} for chunk_id in context_ids],
        "termination_reason": "agent_stop",
    }
    if native_metrics is not None:
        record["native_metrics"] = native_metrics
    if native_trace is not None:
        record["native_trace"] = native_trace
    return record


def _sample(
    query_id: str,
    query: str,
    record: dict[str, Any],
    *,
    error: EvalError | None = None,
    usage: dict[str, ModelUsage] | None = None,
    limit: EvalSampleLimit | None = None,
    events: list[Any] | None = None,
) -> EvalSample:
    return EvalSample.model_construct(
        id=query_id,
        epoch=1,
        input=query,
        target="",
        metadata={"justatom": record},
        model_usage=usage or {},
        error=error,
        limit=limit,
        events=events or [],
    )


def _log(
    method: str,
    sample_ids: list[str],
    samples: list[EvalSample] | None,
    *,
    status: str = "success",
    error: EvalError | None = None,
) -> EvalLog:
    config = {"schema_version": 1, "method": method, "budget": {"top_k": 5}}
    return EvalLog.model_construct(
        status=status,
        metadata=config,
        eval=SimpleNamespace(metadata=config, dataset=SimpleNamespace(sample_ids=sample_ids)),
        samples=samples,
        error=error,
    )


def test_export_results_keeps_error_partial_coverage_and_missing_samples_in_denominators(tmp_path) -> None:
    cases = [
        BenchmarkCase("q1", "first question", ("a",), {"domain": "alpha", "depth": 1}),
        BenchmarkCase("q2", "second question", ("b", "c"), {"domain": "alpha", "depth": 2}),
    ]
    react = _log(
        "react",
        ["q1", "q2"],
        [
            _sample(
                "q1",
                "first question",
                _record("q1", "first question", "react", context_ids=["a"]),
                usage={"provider/model": ModelUsage(input_tokens=10, output_tokens=2, total_tokens=12)},
            ),
            _sample(
                "q2",
                "second question",
                _record("q2", "second question", "react", context_ids=["b"]),
                error=EvalError(message="provider failed", traceback="private", traceback_ansi="private"),
                usage={"provider/model": ModelUsage(input_tokens=4, output_tokens=1, total_tokens=5)},
            ),
        ],
    )
    native_metrics = {
        "token_totals": {
            "input_tokens": 7,
            "output_tokens": 3,
            "total_tokens": 10,
            "cached_input_tokens": None,
            "reasoning_tokens": None,
        },
        "token_coverage": {
            field: {"numerator": 1, "denominator": 1, "rate": 1.0} for field in ("input_tokens", "output_tokens", "total_tokens")
        },
        "token_usage_coverage": {"numerator": 1, "denominator": 1, "rate": 1.0},
        "cost_total_usd": None,
        "cost_coverage": {"numerator": 0, "denominator": 1, "rate": 0.0},
    }
    native = _log(
        "native",
        ["q1", "q2"],
        [
            _sample(
                "q1",
                "first question",
                _record(
                    "q1",
                    "first question",
                    "native",
                    context_ids=["a"],
                    native_metrics=native_metrics,
                    native_trace={"schema_version": 2, "run_id": "native-run"},
                ),
                usage={"mockllm/unused": ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)},
            )
        ],
        status="cancelled",
    )

    summary = export_results(
        [react, native],
        tmp_path,
        cases=cases,
        methods=["react", "native"],
    )

    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert [(row["method"], row["query_id"], row["status"]) for row in rows] == [
        ("react", "q1", "success"),
        ("react", "q2", "error"),
        ("native", "q1", "success"),
        ("native", "q2", "missing"),
    ]
    assert rows[1]["evaluation"]["final_context"] == {
        "all_gold": False,
        "recall": 0.5,
        "hits": 1,
        "gold_count": 2,
        "retrieved_count": 1,
    }
    assert rows[1]["error"] == {"type": "EvalError", "message": "provider failed"}
    assert rows[2]["native_trace"] == {"schema_version": 2, "run_id": "native-run"}
    assert rows[2]["usage"] == {
        "source": "native_metrics",
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
        "cached_input_tokens": None,
        "reasoning_tokens": None,
        "cost_usd": None,
        "observed_token_totals": native_metrics["token_totals"],
        "token_coverage": native_metrics["token_coverage"],
        "token_usage_coverage": native_metrics["token_usage_coverage"],
        "observed_cost_usd": None,
        "cost_coverage": native_metrics["cost_coverage"],
        "observed_only": False,
    }
    assert rows[0]["usage"]["source"] == "inspect_model_usage"
    assert rows[0]["usage"]["total_tokens"] == 12
    assert rows[0]["usage"]["cost_usd"] is None
    assert rows[3]["evaluation"]["final_context"]["recall"] == 0.0
    assert rows[3]["query"] == "second question"
    assert rows[3]["metadata"] == {"domain": "alpha", "depth": 2}

    assert summary["overall"] == {
        "requested": 4,
        "successes": 2,
        "errors": 1,
        "missing": 1,
        "all_gold_rate": 0.5,
        "mean_recall": 0.625,
    }
    assert summary["by_method"]["react"] == {
        "requested": 2,
        "successes": 1,
        "errors": 1,
        "missing": 0,
        "all_gold_rate": 0.5,
        "mean_recall": 0.75,
    }
    assert summary["by_method"]["native"]["missing"] == 1
    assert summary["by_domain"]["alpha"] == summary["overall"]
    assert summary["by_depth"]["2"]["requested"] == 2
    assert json.loads((tmp_path / "summary.json").read_text()) == summary


def test_two_argument_export_infers_method_and_dataset_ids_from_real_eval_log(tmp_path) -> None:
    record = _record("q1", "known query", "react", context_ids=["a"])
    record["evaluation"] = {
        "per_hop": [],
        "cumulative": [],
        "final_context": {"all_gold": True, "recall": 1.0, "hits": 1, "gold_count": 1, "retrieved_count": 1},
        "first_complete_hop": 1,
        "search_calls": 1,
    }
    log = _log("react", ["q1", "q2"], [_sample("q1", "known query", record)])

    summary = export_results([log], tmp_path)

    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert [(row["query_id"], row["status"]) for row in rows] == [("q1", "success"), ("q2", "missing")]
    assert rows[1]["query"] is None
    assert rows[1]["metadata"] == {}
    assert rows[1]["evaluation"]["final_context"]["all_gold"] is False
    assert summary["overall"]["requested"] == 2
    assert summary["overall"]["all_gold_rate"] == 0.5


def test_native_usage_falls_back_to_real_trace_instead_of_inspect_model_usage(tmp_path) -> None:
    trace = {
        "schema_version": 2,
        "steps": [
            {
                "calls": [
                    {"kind": "retrieval", "tokens": None},
                    {
                        "kind": "planner",
                        "tokens": {
                            "input_tokens": 6,
                            "output_tokens": 2,
                            "total_tokens": 8,
                            "cached_input_tokens": 1,
                            "reasoning_tokens": 0,
                        },
                    },
                ]
            }
        ],
    }
    record = _record("q", "question", "native", context_ids=["a"], native_trace=trace)
    log = _log(
        "native",
        ["q"],
        [
            _sample(
                "q",
                "question",
                record,
                usage={"mockllm/unused": ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)},
            )
        ],
    )

    export_results(
        [log],
        tmp_path,
        cases=[BenchmarkCase("q", "question", ("a",), {})],
        methods=["native"],
    )

    row = json.loads((tmp_path / "results.jsonl").read_text())
    assert row["usage"] == {
        "source": "native_trace",
        "input_tokens": 6,
        "output_tokens": 2,
        "total_tokens": 8,
        "cached_input_tokens": 1,
        "reasoning_tokens": 0,
        "cost_usd": None,
    }


def test_time_limited_sample_is_an_error_with_partial_coverage_and_fairness_fields(tmp_path) -> None:
    record = _record("q", "question", "react", context_ids=["a"])
    record.update({"duration_ms": 123.5, "model_calls": 2, "termination_reason": "max_time"})
    sample = _sample(
        "q",
        "question",
        record,
        limit=EvalSampleLimit(type="time", limit=1.0, reason="deadline reached"),
    )

    export_results(
        [_log("react", ["q"], [sample], status="cancelled")],
        tmp_path,
        cases=[BenchmarkCase("q", "question", ("a", "b"), {})],
        methods=["react"],
    )

    row = json.loads((tmp_path / "results.jsonl").read_text())
    assert row["status"] == "error"
    assert row["error"] == {"type": "InspectLimit", "message": "time limit reached"}
    assert row["evaluation"]["final_context"]["recall"] == 0.5
    assert row["duration_ms"] == 123.5
    assert row["model_calls"] == 2
    assert row["limit"] == {"type": "time", "limit": 1.0}


@pytest.mark.parametrize("native_status", ["failed", "timed_out", "cancelled"])
def test_native_terminal_failure_status_is_not_exported_as_success(tmp_path, native_status: str) -> None:
    record = _record(
        "q",
        "question",
        "native",
        context_ids=["a"],
        native_trace={"schema_version": 2, "status": native_status, "steps": []},
    )
    record["termination_reason"] = "error"

    export_results(
        [_log("native", ["q"], [_sample("q", "question", record)])],
        tmp_path,
        cases=[BenchmarkCase("q", "question", ("a", "b"), {})],
        methods=["native"],
    )

    row = json.loads((tmp_path / "results.jsonl").read_text())
    assert row["status"] == "error"
    assert row["error"] == {"type": "NativeRunStatus", "message": native_status}
    assert row["evaluation"]["final_context"]["recall"] == 0.5


def test_native_partial_usage_preserves_observations_without_claiming_complete_totals(tmp_path) -> None:
    native_metrics = {
        "token_totals": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "cached_input_tokens": None,
            "reasoning_tokens": None,
        },
        "token_coverage": {
            "input_tokens": {"numerator": 1, "denominator": 2, "rate": 0.5},
            "output_tokens": {"numerator": 1, "denominator": 2, "rate": 0.5},
            "total_tokens": {"numerator": 1, "denominator": 2, "rate": 0.5},
            "cached_input_tokens": {"numerator": 0, "denominator": 2, "rate": 0.0},
            "reasoning_tokens": {"numerator": 0, "denominator": 2, "rate": 0.0},
        },
        "token_usage_coverage": {"numerator": 1, "denominator": 2, "rate": 0.5},
        "cost_total_usd": 0.004,
        "cost_coverage": {"numerator": 1, "denominator": 2, "rate": 0.5},
    }
    record = _record("q", "question", "native", context_ids=["a"], native_metrics=native_metrics)

    export_results(
        [_log("native", ["q"], [_sample("q", "question", record)])],
        tmp_path,
        cases=[BenchmarkCase("q", "question", ("a",), {})],
        methods=["native"],
    )

    usage = json.loads((tmp_path / "results.jsonl").read_text())["usage"]
    assert usage["input_tokens"] is None
    assert usage["output_tokens"] is None
    assert usage["total_tokens"] is None
    assert usage["cost_usd"] is None
    assert usage["observed_token_totals"] == native_metrics["token_totals"]
    assert usage["observed_cost_usd"] == 0.004
    assert usage["token_usage_coverage"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert usage["cost_coverage"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert usage["observed_only"] is True


def test_react_partial_model_events_do_not_turn_observed_usage_into_complete_totals(tmp_path) -> None:
    known = ModelUsage(input_tokens=10, output_tokens=2, total_tokens=12, total_cost=0.001)
    record = _record("q", "question", "react", context_ids=["a"])
    record["model_calls"] = 2
    sample = _sample(
        "q",
        "question",
        record,
        usage={"provider/model": known},
        events=[
            SimpleNamespace(event="model", output=SimpleNamespace(usage=known)),
            SimpleNamespace(event="model", output=None),
        ],
    )

    export_results(
        [_log("react", ["q"], [sample])],
        tmp_path,
        cases=[BenchmarkCase("q", "question", ("a",), {})],
        methods=["react"],
    )

    usage = json.loads((tmp_path / "results.jsonl").read_text())["usage"]
    assert usage["input_tokens"] is None
    assert usage["output_tokens"] is None
    assert usage["total_tokens"] is None
    assert usage["cost_usd"] is None
    assert usage["observed_token_totals"]["total_tokens"] == 12
    assert usage["observed_cost_usd"] == 0.001
    assert usage["token_usage_coverage"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert usage["token_coverage"]["total_tokens"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert usage["cost_coverage"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert usage["observed_only"] is True


@pytest.mark.parametrize("existing_name", ["results.jsonl", "summary.json"])
def test_export_never_overwrites_either_result_artifact(tmp_path, existing_name: str) -> None:
    existing = tmp_path / existing_name
    existing.write_text("sentinel", encoding="utf-8")
    case = BenchmarkCase("q", "query", ("a",), {})

    with pytest.raises(FileExistsError):
        export_results([], tmp_path, cases=[case], methods=["react"])

    assert existing.read_text(encoding="utf-8") == "sentinel"
    other = tmp_path / ({"results.jsonl": "summary.json", "summary.json": "results.jsonl"}[existing_name])
    assert not other.exists()


def test_serialization_failure_leaves_no_partial_result_artifacts(tmp_path) -> None:
    case = BenchmarkCase("q", "query", ("a",), {"not_json": object()})

    with pytest.raises(TypeError):
        export_results([], tmp_path, cases=[case], methods=["react"])

    assert not (tmp_path / "results.jsonl").exists()
    assert not (tmp_path / "summary.json").exists()
