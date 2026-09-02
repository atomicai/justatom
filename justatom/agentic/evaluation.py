from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, cast

from justatom.agentic.schemas import CallKind, CallStatus, RunTrace


@dataclass(frozen=True, slots=True)
class EvidenceLabels:
    """Offline-only relevance and evidence-completion labels.

    Positive qrels use the conventional ``relevance > 0`` rule.  Every required
    evidence group is a set of interchangeable documents for one required
    evidence slot: a group is satisfied when at least one member is present,
    and a run completes the evidence target when every group is satisfied.
    """

    qrels: Mapping[str, float] | Iterable[str] = field(default_factory=dict)
    required_evidence_groups: Iterable[Iterable[str]] = ()

    def __post_init__(self) -> None:
        raw_qrels = self.qrels
        qrels_items: Iterable[tuple[Any, Any]]
        if isinstance(raw_qrels, Mapping):
            qrels_items = raw_qrels.items()
        else:
            if isinstance(raw_qrels, (str, bytes)):
                raise TypeError("qrels must be a mapping or an iterable of document ids")
            try:
                qrels_items = ((document_id, 1.0) for document_id in raw_qrels)
            except TypeError as exc:
                raise TypeError("qrels must be a mapping or an iterable of document ids") from exc

        normalized_qrels: dict[str, float] = {}
        try:
            for document_id, relevance in qrels_items:
                _validate_document_id(document_id)
                if isinstance(relevance, bool) or not isinstance(relevance, (int, float)):
                    raise TypeError("qrel relevance must be a finite number")
                try:
                    numeric_relevance = float(relevance)
                except OverflowError as exc:
                    raise ValueError("qrel relevance must be finite") from exc
                if not math.isfinite(numeric_relevance):
                    raise ValueError("qrel relevance must be finite")
                normalized_qrels[document_id] = numeric_relevance
        except TypeError as exc:
            if str(exc).startswith("qrel") or str(exc).startswith("document_id"):
                raise
            raise TypeError("qrels must be a mapping or an iterable of document ids") from exc

        if isinstance(self.required_evidence_groups, (str, bytes)):
            raise TypeError("required_evidence_groups must be an iterable of document-id groups")
        normalized_groups: list[frozenset[str]] = []
        try:
            for raw_group in self.required_evidence_groups:
                if isinstance(raw_group, (str, bytes)):
                    raise TypeError("each required evidence group must be an iterable of document ids")
                group = frozenset(raw_group)
                if not group:
                    raise ValueError("required evidence groups must not be empty")
                for document_id in group:
                    _validate_document_id(document_id)
                normalized_groups.append(group)
        except TypeError as exc:
            if str(exc).startswith("each required") or str(exc).startswith("document_id"):
                raise
            raise TypeError("required_evidence_groups must be an iterable of document-id groups") from exc

        object.__setattr__(self, "qrels", MappingProxyType(normalized_qrels))
        object.__setattr__(self, "required_evidence_groups", tuple(normalized_groups))

    @property
    def positive_qrels(self) -> frozenset[str]:
        qrels = cast(Mapping[str, float], self.qrels)
        return frozenset(document_id for document_id, relevance in qrels.items() if relevance > 0)

    @property
    def evidence_document_ids(self) -> frozenset[str]:
        groups = cast(tuple[frozenset[str], ...], self.required_evidence_groups)
        group_documents = {document_id for group in groups for document_id in group}
        return self.positive_qrels | frozenset(group_documents)


def _validate_document_id(document_id: object) -> None:
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id must be a non-empty string")


def _validate_inputs(trace: RunTrace, labels: EvidenceLabels, k: int) -> None:
    if not isinstance(trace, RunTrace):
        raise TypeError("trace must be a RunTrace")
    if not isinstance(labels, EvidenceLabels):
        raise TypeError("labels must be EvidenceLabels")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")


def _deduplicate(document_ids: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for document_id in document_ids:
        if document_id not in seen:
            seen.add(document_id)
            result.append(document_id)
    return result


def _graded_relevance(labels: EvidenceLabels) -> dict[str, float]:
    # Ranking metrics retain standard qrels semantics.  Required groups are an
    # independent completion target and may contain aliases that must not
    # inflate the qrels recall denominator.
    qrels = cast(Mapping[str, float], labels.qrels)
    return {document_id: relevance for document_id, relevance in qrels.items() if relevance > 0}


def _groups_satisfied(document_ids: set[str], groups: tuple[frozenset[str], ...]) -> int:
    return sum(not document_ids.isdisjoint(group) for group in groups)


def _dcg(ranked_gains: Iterable[tuple[int, float]], *, scale: float) -> float:
    """Compute linear-gain DCG on a shared scale to avoid finite-value overflow."""

    terms: list[float] = []
    for rank, gain in ranked_gains:
        if gain <= 0:
            continue
        # Dividing every gain by the same positive maximum leaves nDCG
        # unchanged while keeping even very large finite qrels summable.
        terms.append((gain / scale) / math.log2(rank + 1))
    return math.fsum(terms)


def _ranking_metrics(
    document_ids: list[str],
    labels: EvidenceLabels,
    *,
    evaluation_depth: int,
    rank_positions: list[int] | None = None,
) -> dict[str, Any]:
    positions = rank_positions if rank_positions is not None else list(range(1, len(document_ids) + 1))
    if len(positions) != len(document_ids):
        raise ValueError("rank_positions must align with document_ids")
    grades = _graded_relevance(labels)
    relevant_ids = set(grades)
    unique_ids = set(document_ids)
    relevant_retrieved_ids = unique_ids & relevant_ids
    precision_numerator = len(relevant_retrieved_ids)
    precision_denominator = evaluation_depth
    recall_numerator = len(relevant_retrieved_ids)
    recall_denominator = len(relevant_ids)

    reciprocal_rank: float | None = None
    ranked_gains: list[tuple[int, float]] = []
    seen: set[str] = set()
    for rank, document_id in zip(positions, document_ids):
        gain = grades.get(document_id, 0.0) if document_id not in seen else 0.0
        seen.add(document_id)
        ranked_gains.append((rank, gain))
        if reciprocal_rank is None and gain > 0:
            reciprocal_rank = 1.0 / rank

    ideal_gains = sorted(grades.values(), reverse=True)[:evaluation_depth]
    gain_scale = max(grades.values(), default=0.0)
    dcg = _dcg(ranked_gains, scale=gain_scale)
    ideal_dcg = _dcg(enumerate(ideal_gains, start=1), scale=gain_scale)

    groups = cast(tuple[frozenset[str], ...], labels.required_evidence_groups)
    groups_hit = _groups_satisfied(unique_ids, groups)
    group_count = len(groups)
    required_complete: bool | None = groups_hit == group_count if group_count else None

    return {
        "document_ids": list(document_ids),
        "evaluation_depth": evaluation_depth,
        "observed_depth": len(document_ids),
        "retrieved_document_count": len(document_ids),
        "unique_retrieved_document_count": len(unique_ids),
        "relevant_retrieved_count": len(relevant_retrieved_ids),
        "relevant_document_count": len(relevant_ids),
        "precision_numerator": precision_numerator,
        "precision_denominator": precision_denominator,
        "precision_at_depth": precision_numerator / precision_denominator if precision_denominator else None,
        "recall_numerator": recall_numerator,
        "recall_denominator": recall_denominator,
        "recall_at_depth": recall_numerator / recall_denominator if recall_denominator else None,
        "hit_at_depth": bool(relevant_retrieved_ids),
        "reciprocal_rank_at_depth": reciprocal_rank if reciprocal_rank is not None else (0.0 if relevant_ids else None),
        "ndcg_at_depth": dcg / ideal_dcg if ideal_dcg else None,
        "required_evidence_groups_hit": groups_hit,
        "required_evidence_group_count": group_count,
        "required_evidence_group_recall": groups_hit / group_count if group_count else None,
        "required_evidence_complete": required_complete,
    }


def _retrieval_hops(trace: RunTrace, k: int) -> list[dict[str, Any]]:
    recorded: list[tuple[int, int, int, CallStatus, list[tuple[int, str]]]] = []
    for step_position, step in enumerate(trace.steps):
        for call_position, call in enumerate(step.calls):
            if call.kind is not CallKind.RETRIEVAL or call.retrieval is None:
                continue
            ranked = (
                sorted(
                    ((document.rank, document.document_id) for document in call.retrieval.documents),
                    key=lambda item: item[0],
                )[:k]
                if call.status is CallStatus.OK
                else []
            )
            recorded.append((call.retrieval.retrieval_index, step_position, call_position, call.status, ranked))

    recorded.sort(key=lambda item: (item[0], item[1], item[2]))
    return [
        {
            "hop": hop,
            "retrieval_index": retrieval_index,
            "status": status.value,
            "successful": status is CallStatus.OK,
            "ranked_documents": ranked,
        }
        for hop, (retrieval_index, _step_position, _call_position, status, ranked) in enumerate(recorded, start=1)
    ]


def _completion_target_satisfied(seen: set[str], labels: EvidenceLabels) -> bool:
    groups = cast(tuple[frozenset[str], ...], labels.required_evidence_groups)
    if groups:
        return _groups_satisfied(seen, groups) == len(groups)
    positive_qrels = set(labels.positive_qrels)
    return bool(positive_qrels) and positive_qrels.issubset(seen)


def evaluate_trace(trace: RunTrace, labels: EvidenceLabels, k: int = 10) -> dict[str, Any]:
    """Evaluate recorded retrievals without exposing gold labels to the runtime.

    ``k`` truncates each retrieval hop.  Every retrieval call with a payload
    consumes a hop; a non-OK call contributes no documents but still spends
    ``k`` cumulative slots.  The cumulative ranked series preserves every
    consumed slot (including repeats), while evidence completion uses a
    deduplicated seen set.  Final-context metrics independently evaluate the
    first ``k`` document ids actually handed to the answer stage.
    """

    _validate_inputs(trace, labels, k)
    hops = _retrieval_hops(trace, k)
    relevant_ids = set(labels.positive_qrels) or set(labels.evidence_document_ids)

    per_hop: list[dict[str, Any]] = []
    cumulative: list[dict[str, Any]] = []
    cumulative_ranked_ids: list[str] = []
    cumulative_rank_positions: list[int] = []
    cumulative_seen: set[str] = set()

    first_hit_hop: int | None = None
    first_hit_retrieval_index: int | None = None
    first_hit_rank: int | None = None
    completion_hop: int | None = None
    completion_retrieval_index: int | None = None
    completion_rank: int | None = None

    for hop_record in hops:
        ranked_documents: list[tuple[int, str]] = hop_record["ranked_documents"]
        hop_document_ids = [document_id for _rank, document_id in ranked_documents]
        hop_metrics = _ranking_metrics(hop_document_ids, labels, evaluation_depth=k)
        hop_metrics.update(
            {
                "hop": hop_record["hop"],
                "retrieval_index": hop_record["retrieval_index"],
                "status": hop_record["status"],
                "successful": hop_record["successful"],
            }
        )
        per_hop.append(hop_metrics)

        if first_hit_hop is None:
            for rank, document_id in ranked_documents:
                if document_id in relevant_ids:
                    first_hit_hop = hop_record["hop"]
                    first_hit_retrieval_index = hop_record["retrieval_index"]
                    first_hit_rank = rank
                    break

        hop_slot_offset = (hop_record["hop"] - 1) * k
        for observed_rank, (rank, document_id) in enumerate(ranked_documents, start=1):
            cumulative_ranked_ids.append(document_id)
            cumulative_rank_positions.append(hop_slot_offset + observed_rank)
            if document_id not in cumulative_seen:
                cumulative_seen.add(document_id)
            if completion_hop is None and _completion_target_satisfied(cumulative_seen, labels):
                completion_hop = hop_record["hop"]
                completion_retrieval_index = hop_record["retrieval_index"]
                completion_rank = rank

        cumulative_metrics = _ranking_metrics(
            cumulative_ranked_ids,
            labels,
            evaluation_depth=hop_record["hop"] * k,
            rank_positions=cumulative_rank_positions,
        )
        cumulative_metrics.update(
            {
                "hop": hop_record["hop"],
                "retrieval_index": hop_record["retrieval_index"],
                "status": hop_record["status"],
                "successful": hop_record["successful"],
            }
        )
        cumulative.append(cumulative_metrics)

    final_context_ids = _deduplicate(trace.final_context_document_ids[:k])
    final_context = _ranking_metrics(final_context_ids, labels, evaluation_depth=k)
    final_citation_ids = list(trace.final_cited_document_ids[:k])
    final_citations = _ranking_metrics(
        final_citation_ids,
        labels,
        evaluation_depth=len(final_citation_ids),
    )
    final_citations["selection_precision"] = final_citations["precision_at_depth"]
    final_citations["emitted_occurrence_count"] = len(trace.final_cited_document_ids)
    final_citations["evaluated_occurrence_count"] = len(final_citation_ids)
    final_citations["evaluated_unique_count"] = len(set(final_citation_ids))
    final_citations["evaluated_duplicate_count"] = len(final_citation_ids) - len(set(final_citation_ids))
    final_citation_set_ids = _deduplicate(trace.final_cited_document_ids)[:k]
    final_citation_set = _ranking_metrics(
        final_citation_set_ids,
        labels,
        evaluation_depth=len(final_citation_set_ids),
    )
    final_citation_set["selection_precision"] = final_citation_set["precision_at_depth"]

    return {
        "run_id": trace.run_id,
        "k": k,
        "positive_qrel_count": len(labels.positive_qrels),
        "required_evidence_group_count": len(tuple(labels.required_evidence_groups)),
        "retrieval_hop_count": len(hops),
        "successful_hop_count": sum(hop["successful"] for hop in hops),
        "failed_hop_count": sum(not hop["successful"] for hop in hops),
        "per_hop": per_hop,
        "cumulative": cumulative,
        "final_context": final_context,
        "final_citations": final_citations,
        "final_citation_set": final_citation_set,
        "first_hit_hop": first_hit_hop,
        "first_hit_retrieval_index": first_hit_retrieval_index,
        "first_hit_rank": first_hit_rank,
        "completion_hop": completion_hop,
        "completion_retrieval_index": completion_retrieval_index,
        "completion_rank": completion_rank,
        "evidence_completed": completion_hop is not None,
    }


__all__ = ["EvidenceLabels", "evaluate_trace"]
