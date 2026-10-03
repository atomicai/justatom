from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai import eval_async
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput, get_model
from inspect_ai.tool import ToolCall

from justatom.etc.schema import Document


class Retriever:
    def __init__(self):
        self.queries = []

    async def retrieve(self, query, *, top_k=5, **kwargs):
        self.queries.append((query, top_k, kwargs))
        return [Document(id="a" if query == "original" else "b", content="Evidence " + query)]


def search_output(query):
    return ModelOutput(
        model="fixture",
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content="", tool_calls=[ToolCall(id="search-1", function="search", arguments={"query": query})]
                ),
                stop_reason="tool_calls",
            )
        ],
    )


def execute(tmp_path, task, outputs):
    model = get_model("mockllm/harness", custom_outputs=outputs, memoize=False)
    logs = asyncio.run(eval_async(task, model=model, log_dir=str(tmp_path), max_samples=1, log_model_api=False))
    assert len(logs) == 1
    return logs[0]


def test_react_runs_real_tool_loop_and_scores_only_after_search(tmp_path):
    from justatom.agentic.benchmark import SearchBudget
    from justatom.agentic.benchmark_data import BenchmarkCase
    from justatom.agentic.inspect_harness import build_task

    retriever = Retriever()
    case = BenchmarkCase("q1", "original", ("a", "b"), {})
    task = build_task([case], retriever, budget=SearchBudget(top_k=1, max_searches=3))
    log = execute(tmp_path, task, [search_output("followup"), ModelOutput.from_content("fixture", "DONE")])
    assert log.status == "success"
    sample = log.samples[0]
    assert sample.error is None
    record = sample.metadata["justatom"]
    assert [c["query"] for c in record["calls"]] == ["original", "followup"]
    assert record["context_ids"] == ["a", "b"]
    assert record["evaluation"]["final_context"]["all_gold"] is True
    assert record["evaluation"]["first_complete_hop"] == 2
    assert retriever.queries == [("original", 1, {}), ("followup", 1, {})]
    assert any(getattr(m, "tool_calls", None) for m in sample.messages)


def test_react_never_exposes_gold_to_model_and_does_not_retry_incorrect_answer(tmp_path):
    from justatom.agentic.benchmark_data import BenchmarkCase
    from justatom.agentic.inspect_harness import build_task

    seen = []

    def output(messages, tools, tool_choice, config):
        seen.append(json.dumps([m.model_dump() for m in messages]))
        return ModelOutput.from_content("fixture", "DONE")

    task = build_task([BenchmarkCase("q1", "original", ("gold-only-secret",), {})], Retriever())
    log = execute(tmp_path, task, output)
    assert len(seen) == 1
    assert "gold-only-secret" not in seen[0]
    assert log.samples[0].metadata["justatom"]["evaluation"]["final_context"]["all_gold"] is False


def test_search_cap_stops_react_before_extra_model_or_backend_work(tmp_path):
    from justatom.agentic.benchmark import SearchBudget
    from justatom.agentic.benchmark_data import BenchmarkCase
    from justatom.agentic.inspect_harness import build_task

    retriever = Retriever()
    task = build_task([BenchmarkCase("q", "original", ("a", "b"), {})], retriever, budget=SearchBudget(top_k=1, max_searches=2))
    log = execute(tmp_path, task, [search_output("followup")])
    record = log.samples[0].metadata["justatom"]
    assert log.samples[0].error is None
    assert len(retriever.queries) == 2
    assert record["termination_reason"] == "max_searches"
    assert record["evaluation"]["final_context"]["all_gold"] is True


def test_one_search_configuration_never_calls_model(tmp_path):
    from justatom.agentic.benchmark import SearchBudget
    from justatom.agentic.benchmark_data import BenchmarkCase
    from justatom.agentic.inspect_harness import build_task

    task = build_task([BenchmarkCase("q", "original", ("a",), {})], Retriever(), budget=SearchBudget(max_searches=1))
    log = execute(tmp_path, task, [])
    assert log.samples[0].error is None
    assert log.samples[0].metadata["justatom"]["evaluation"]["final_context"]["all_gold"] is True


def test_native_control_keeps_real_runtime_trace_and_matches_common_metrics(tmp_path):
    from justatom.agentic.benchmark import SearchBudget
    from justatom.agentic.benchmark_data import BenchmarkCase
    from justatom.agentic.inspect_harness import build_task
    from justatom.agentic.schemas import AgentAction, PlannerDecision, PlannerReply

    class Planner:
        backend_name = "fixture"
        model_name = "fixture"
        prompt_fingerprint = "fixture-prompt"
        config_fingerprint = "fixture-config"

        async def plan(self, request):
            assert request.question == "original"
            assert not hasattr(request, "chunk_ids")
            return PlannerReply(PlannerDecision(action=AgentAction.SEARCH, query="followup"))

        async def close(self):
            pass

    task = build_task(
        [BenchmarkCase("q", "original", ("a", "b"), {})],
        Retriever(),
        method="native",
        planner_factory=Planner,
        max_model_calls=1,
        budget=SearchBudget(top_k=1, max_searches=2),
    )
    log = execute(tmp_path, task, [])
    sample = log.samples[0]
    assert sample.error is None
    record = sample.metadata["justatom"]
    assert record["native_trace"]["schema_version"] == 2
    assert [call["index"] for call in record["calls"]] == [1, 2]
    assert record["context_ids"] == ["a", "b"]
    assert record["evaluation"]["first_complete_hop"] == 2
    assert record["evaluation"]["final_context"]["all_gold"] is True


def test_initial_retrieval_error_is_recorded_and_scored_without_dropping_sample(tmp_path):
    from justatom.agentic.benchmark_data import BenchmarkCase
    from justatom.agentic.inspect_harness import build_task

    class BrokenRetriever:
        async def retrieve(self, *args, **kwargs):
            raise RuntimeError("private backend diagnostic")

    task = build_task([BenchmarkCase("q", "original", ("a",), {})], BrokenRetriever())
    log = execute(tmp_path, task, [])
    sample = log.samples[0]
    assert sample.error is not None
    record = sample.metadata["justatom"]
    assert record["calls"][0]["status"] == "error"
    assert record["evaluation"]["final_context"]["all_gold"] is False


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
def test_followup_failure_keeps_partial_model_history_and_context(tmp_path, error_type):
    from justatom.agentic.benchmark_data import BenchmarkCase
    from justatom.agentic.inspect_harness import build_task

    class BrokenFollowup(Retriever):
        async def retrieve(self, query, **kwargs):
            if query == "followup":
                raise error_type("backend failed")
            return await super().retrieve(query, **kwargs)

    task = build_task([BenchmarkCase("q", "original", ("a", "b"), {})], BrokenFollowup())
    log = execute(tmp_path, task, [search_output("followup"), ModelOutput.from_content("fixture", "DONE")])
    sample = log.samples[0]
    assert sample.error is not None
    assert any(getattr(message, "tool_calls", None) for message in sample.messages)
    record = sample.metadata["justatom"]
    assert record["model_calls"] == 1
    assert record["context_ids"] == ["a"]
    assert [call["status"] for call in record["calls"]] == ["ok", "error"]
    assert record["evaluation"]["final_context"]["recall"] == 0.5


def test_model_call_limit_stops_repeated_queries(tmp_path):
    from justatom.agentic.benchmark_data import BenchmarkCase
    from justatom.agentic.inspect_harness import build_task

    task = build_task([BenchmarkCase("q", "original", ("a", "b"), {})], Retriever(), max_model_calls=1)
    log = execute(tmp_path, task, [search_output("followup")])
    sample = log.samples[0]
    assert sample.error is None
    assert sample.metadata["justatom"]["termination_reason"] == "max_model_calls"
    assert sample.metadata["justatom"]["model_calls"] == 1
    assert len(sample.metadata["justatom"]["calls"]) <= 2


def test_native_timeout_retains_trace_and_partial_coverage(tmp_path):
    from justatom.agentic.benchmark_data import BenchmarkCase
    from justatom.agentic.inspect_harness import build_task

    class SlowPlanner:
        backend_name = "fixture"
        model_name = "fixture"
        prompt_fingerprint = "fixture-prompt"
        config_fingerprint = "fixture-config"

        async def plan(self, request):
            await asyncio.sleep(1.1)
            raise RuntimeError("test backend finishes after the deadline")

        async def close(self):
            pass

    task = build_task(
        [BenchmarkCase("q", "original", ("a", "b"), {})], Retriever(), method="native", planner_factory=SlowPlanner, time_limit=1
    )
    log = execute(tmp_path, task, [])
    record = log.samples[0].metadata["justatom"]
    assert record["native_trace"]["schema_version"] == 2
    assert record["native_trace"]["status"] in {"timed_out", "cancelled"}
    assert record["context_ids"] == ["a"]
    assert record["context_documents"] == [{"chunk_id": "a", "content": "Evidence original"}]
    assert record["calls"][0]["status"] == "ok"
    assert record["evaluation"]["final_context"]["recall"] == 0.5


@pytest.mark.parametrize("token_cap", [99, 100])
def test_observed_token_boundary_stops_before_search_and_counts_completion(tmp_path, token_cap):
    from inspect_ai.model import ModelUsage

    from justatom.agentic.benchmark_data import BenchmarkCase
    from justatom.agentic.inspect_harness import build_task

    output = search_output("followup")
    output.usage = ModelUsage(input_tokens=80, output_tokens=20, total_tokens=100)
    retriever = Retriever()
    task = build_task([BenchmarkCase("q", "original", ("a", "b"), {})], retriever, token_limit=token_cap)
    log = execute(tmp_path, task, [output])
    sample = log.samples[0]
    assert sample.error is None
    record = sample.metadata["justatom"]
    assert record["termination_reason"] == "max_token"
    assert record["model_calls"] == 1
    assert [call[0] for call in retriever.queries] == ["original"]


def test_transport_timeout_is_not_silently_treated_as_success(tmp_path):
    from justatom.agentic.benchmark_data import BenchmarkCase
    from justatom.agentic.inspect_harness import build_task

    class TimeoutRetriever:
        async def retrieve(self, *args, **kwargs):
            raise TimeoutError("transport timeout")

    task = build_task([BenchmarkCase("q", "original", ("a",), {})], TimeoutRetriever())
    log = execute(tmp_path, task, [])
    assert log.samples[0].error is not None
    assert log.samples[0].metadata["justatom"]["calls"][0]["error_type"] == "TimeoutError"
