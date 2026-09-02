from __future__ import annotations

import math
from dataclasses import replace

import pytest

from justatom.agentic.evaluation import EvidenceLabels, evaluate_trace
from justatom.agentic.schemas import (
    TRACE_SCHEMA_VERSION,
    AgentAction,
    CallKind,
    CallStatus,
    CallTrace,
    DocumentTrace,
    RetrievalPayload,
    RunLimits,
    RunStatus,
    RunTrace,
    StepTrace,
    TerminationReason,
    TextCapturePolicy,
)


def _retrieval(
    index: int,
    document_ids: tuple[str, ...],
    *,
    status: CallStatus = CallStatus.OK,
) -> CallTrace:
    documents = tuple(
        DocumentTrace(
            document_id=document_id,
            rank=rank,
            score=1.0 / rank,
            content_chars=1,
            content_sha256=f"hash-{document_id}",
        )
        for rank, document_id in enumerate(document_ids, start=1)
    )
    return CallTrace(
        call_id=f"retrieval-{index}",
        call_index=index,
        kind=CallKind.RETRIEVAL,
        backend="test",
        model=None,
        started_offset_ms=float(index),
        latency_ms=1.0,
        status=status,
        retrieval=RetrievalPayload(
            retrieval_index=index,
            query_sha256=f"query-{index}",
            normalized_query_sha256=f"normalized-{index}",
            query_text=None,
            mode="dense",
            collection="docs",
            index_revision=None,
            top_k_requested=max(len(document_ids), 1),
            documents=documents,
        ),
    )


def _trace(
    hops: tuple[tuple[str, ...], ...],
    *,
    final_context: tuple[str, ...],
    final_citations: tuple[str, ...] = (),
    hop_statuses: tuple[CallStatus, ...] | None = None,
) -> RunTrace:
    statuses = (CallStatus.OK,) * len(hops) if hop_statuses is None else hop_statuses
    if len(statuses) != len(hops):
        raise ValueError("hop_statuses must align with hops")
    steps = tuple(
        StepTrace(
            step_index=index,
            started_offset_ms=float(index),
            latency_ms=1.0,
            action=AgentAction.SEARCH,
            calls=(_retrieval(index, documents, status=statuses[index]),),
            context_document_ids=(),
        )
        for index, documents in enumerate(hops)
    )
    return RunTrace(
        schema_version=TRACE_SCHEMA_VERSION,
        run_id="run",
        request_id=None,
        experiment_id=None,
        variant=None,
        seed=None,
        config_fingerprint="config",
        planner_config_fingerprint="planner-config",
        retrieval_config_fingerprint="retrieval-config",
        capture_text=TextCapturePolicy.HASH,
        query_sha256="query",
        normalized_query_sha256="normalized-query",
        query_text=None,
        started_at="2026-01-01T00:00:00Z",
        duration_ms=2.0,
        queue_latency_ms=0.0,
        execution_ms=2.0,
        status=RunStatus.COMPLETED,
        termination_reason=TerminationReason.ANSWERED,
        budget_dimension=None,
        limits=RunLimits(
            max_steps=5,
            max_retrieval_calls=5,
            max_llm_calls=5,
            max_tokens=None,
            max_duration_ms=1000.0,
        ),
        steps=steps,
        final_context_document_ids=final_context,
        answer_sha256="answer",
        answer_text=None,
        final_cited_document_ids=final_citations,
    )


def test_evaluate_trace_reports_per_hop_cumulative_and_final_context_metrics():
    trace = _trace(
        (("noise", "a"), ("b", "noise")),
        final_context=("b", "noise"),
        final_citations=("b",),
    )
    labels = EvidenceLabels(
        qrels={"a": 1, "b": 2, "explicit-negative": 0},
        required_evidence_groups=(("a", "a-alias"), ("b",)),
    )

    result = evaluate_trace(trace, labels, k=2)

    assert result["retrieval_hop_count"] == 2
    assert result["per_hop"][0]["hit_at_depth"] is True
    assert result["per_hop"][0]["recall_at_depth"] == 0.5
    assert result["per_hop"][0]["required_evidence_group_recall"] == 0.5
    assert result["per_hop"][1]["required_evidence_complete"] is False
    assert result["cumulative"][1]["recall_at_depth"] == 1.0
    assert result["cumulative"][1]["evaluation_depth"] == 4
    assert result["cumulative"][1]["required_evidence_complete"] is True
    assert result["final_context"]["recall_at_depth"] == 0.5
    assert result["final_citations"]["selection_precision"] == 1.0
    assert result["final_citations"]["recall_at_depth"] == 0.5
    assert result["first_hit_hop"] == 1
    assert result["first_hit_retrieval_index"] == 0
    assert result["first_hit_rank"] == 2
    assert result["completion_hop"] == 2
    assert result["completion_rank"] == 1
    assert result["evidence_completed"] is True


def test_k_is_applied_to_each_hop_before_cumulative_metrics():
    trace = _trace((("n1", "n2", "gold"), ("gold",)), final_context=("n1", "n2", "gold"))
    result = evaluate_trace(trace, EvidenceLabels(qrels={"gold": 1}), k=2)

    assert result["per_hop"][0]["hit_at_depth"] is False
    assert result["per_hop"][0]["reciprocal_rank_at_depth"] == 0.0
    assert result["per_hop"][1]["hit_at_depth"] is True
    assert result["first_hit_hop"] == 2
    assert result["cumulative"][0]["document_ids"] == ["n1", "n2"]
    assert result["final_context"]["hit_at_depth"] is False


def test_cumulative_ranking_preserves_duplicate_slots_before_a_later_hit():
    trace = _trace((("noise", "noise"), ("gold",)), final_context=("noise", "gold"))

    result = evaluate_trace(trace, EvidenceLabels(qrels={"gold": 1}), k=2)

    assert result["cumulative"][1]["document_ids"] == ["noise", "noise", "gold"]
    assert result["cumulative"][1]["reciprocal_rank_at_depth"] == pytest.approx(1 / 3)
    assert result["cumulative"][1]["unique_retrieved_document_count"] == 2


def test_cumulative_ranking_zero_fills_under_returned_hop_slots():
    trace = _trace((("noise",), ("gold",)), final_context=("noise", "gold"))

    result = evaluate_trace(trace, EvidenceLabels(qrels={"gold": 1}), k=2)

    cumulative = result["cumulative"][1]
    assert cumulative["document_ids"] == ["noise", "gold"]
    assert cumulative["evaluation_depth"] == 4
    assert cumulative["observed_depth"] == 2
    assert cumulative["reciprocal_rank_at_depth"] == pytest.approx(1 / 3)
    assert cumulative["ndcg_at_depth"] == pytest.approx(0.5)


def test_failed_retrieval_consumes_cumulative_depth_without_contributing_documents():
    trace = _trace(
        (("gold",), ()),
        final_context=("gold",),
        hop_statuses=(CallStatus.OK, CallStatus.ERROR),
    )

    result = evaluate_trace(trace, EvidenceLabels(qrels={"gold": 1}), k=2)

    assert result["retrieval_hop_count"] == 2
    assert result["successful_hop_count"] == 1
    assert result["failed_hop_count"] == 1
    assert result["per_hop"][1]["status"] == "error"
    assert result["per_hop"][1]["successful"] is False
    assert result["per_hop"][1]["document_ids"] == []
    cumulative = result["cumulative"][1]
    assert cumulative["status"] == "error"
    assert cumulative["successful"] is False
    assert cumulative["evaluation_depth"] == 4
    assert cumulative["observed_depth"] == 1
    assert cumulative["precision_at_depth"] == pytest.approx(1 / 4)


def test_precision_zero_fills_unreturned_slots_but_citation_precision_uses_observed_selection():
    trace = _trace(
        (("gold",),),
        final_context=("gold",),
        final_citations=("gold",),
    )

    result = evaluate_trace(trace, EvidenceLabels(qrels={"gold": 1}), k=3)

    assert result["per_hop"][0]["evaluation_depth"] == 3
    assert result["per_hop"][0]["observed_depth"] == 1
    assert result["per_hop"][0]["precision_at_depth"] == pytest.approx(1 / 3)
    assert result["final_context"]["precision_at_depth"] == pytest.approx(1 / 3)
    assert result["final_citations"]["evaluation_depth"] == 1
    assert result["final_citations"]["selection_precision"] == 1.0


def test_citation_duplicates_consume_occurrence_slots_and_set_view_is_explicit():
    trace = _trace(
        (("gold",),),
        final_context=("gold",),
        final_citations=("gold", "gold", "noise"),
    )

    result = evaluate_trace(trace, EvidenceLabels(qrels={"gold": 1}), k=2)

    occurrences = result["final_citations"]
    assert occurrences["document_ids"] == ["gold", "gold"]
    assert occurrences["selection_precision"] == pytest.approx(1 / 2)
    assert occurrences["emitted_occurrence_count"] == 3
    assert occurrences["evaluated_occurrence_count"] == 2
    assert occurrences["evaluated_unique_count"] == 1
    assert occurrences["evaluated_duplicate_count"] == 1
    citation_set = result["final_citation_set"]
    assert citation_set["document_ids"] == ["gold", "noise"]
    assert citation_set["selection_precision"] == pytest.approx(1 / 2)


def test_completion_falls_back_to_covering_all_positive_qrels_when_groups_are_absent():
    trace = _trace((("a",), ("b",)), final_context=("a", "b"))
    result = evaluate_trace(trace, EvidenceLabels(qrels={"a": 1, "b": 1}), k=1)

    assert result["completion_hop"] == 2
    assert result["evidence_completed"] is True
    assert result["per_hop"][0]["required_evidence_complete"] is None


def test_evaluation_is_pure_and_does_not_add_gold_to_runtime_trace():
    trace = _trace((("a",),), final_context=("a",))
    before = trace.to_dict()

    evaluate_trace(trace, EvidenceLabels(qrels={"a": 1}), k=1)

    assert trace.to_dict() == before
    assert "qrels" not in trace.to_dict()
    assert "required_evidence_groups" not in trace.to_dict()


def test_empty_labels_produce_explicit_undefined_denominators():
    result = evaluate_trace(_trace((("a",),), final_context=("a",)), EvidenceLabels(), k=1)

    assert result["first_hit_hop"] is None
    assert result["completion_hop"] is None
    assert result["per_hop"][0]["precision_at_depth"] == 0.0
    assert result["per_hop"][0]["recall_at_depth"] is None
    assert result["per_hop"][0]["ndcg_at_depth"] is None


def test_ndcg_is_stable_for_large_finite_graded_relevance():
    trace = _trace((("gold", "less-relevant"),), final_context=("gold", "less-relevant"))

    result = evaluate_trace(trace, EvidenceLabels(qrels={"gold": 1024.0, "less-relevant": 512.0}), k=2)

    assert result["per_hop"][0]["ndcg_at_depth"] == pytest.approx(1.0)


def test_ndcg_uses_linear_graded_relevance_gain():
    trace = _trace((("less-relevant", "gold"),), final_context=("less-relevant", "gold"))

    result = evaluate_trace(trace, EvidenceLabels(qrels={"gold": 2.0, "less-relevant": 1.0}), k=2)

    expected = (1.0 + 2.0 / math.log2(3)) / (2.0 + 1.0 / math.log2(3))
    assert result["per_hop"][0]["ndcg_at_depth"] == pytest.approx(expected)


@pytest.mark.parametrize("schema_version", [True, 1.0, "1", None])
def test_run_trace_rejects_non_integer_schema_version(schema_version):
    with pytest.raises(ValueError, match="schema_version must be an integer"):
        replace(_trace((), final_context=()), schema_version=schema_version)


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_evaluate_trace_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_trace(_trace((), final_context=()), EvidenceLabels(), k=k)


def test_evidence_labels_validate_qrels_and_required_groups():
    with pytest.raises(ValueError, match="finite"):
        EvidenceLabels(qrels={"a": float("nan")})
    with pytest.raises(ValueError, match="finite"):
        EvidenceLabels(qrels={"a": 10**1000})
    with pytest.raises(ValueError, match="must not be empty"):
        EvidenceLabels(required_evidence_groups=((),))
