from __future__ import annotations

import os
import re
import unicodedata
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from justatom.tooling.ir_dataset.dense import DenseIndex
from justatom.tooling.ir_dataset.neighbors import NeighborCandidate, include_structural_neighbors, merge_neighbors
from justatom.tooling.ir_dataset.sparse import BM25Index


@dataclass(frozen=True, slots=True)
class GenerationContextConfig:
    bm25_k: int = 20
    dense_k: int = 20
    union_k: int = 30
    rrf_k: int = 60
    dense_block_size: int = 65_536
    device: str = "auto"
    output_dir: str | Path | None = None

    def __post_init__(self) -> None:
        for field in ("bm25_k", "dense_k", "union_k", "rrf_k", "dense_block_size"):
            if getattr(self, field) <= 0:
                raise ValueError(f"generation context {field} must be > 0")


def _require_columns(frame: pl.DataFrame, name: str, required: set[str]) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _structural_members(query: dict[str, Any], article_members: dict[str, list[dict[str, Any]]]) -> list[tuple[str, bool]]:
    query_start = int(query["start_unit"])
    query_end = int(query["end_unit"])
    structural_rows: list[tuple[str, bool, int, int]] = []
    for candidate in article_members[str(query["article_id"])]:
        candidate_id = str(candidate["passage_id"])
        if candidate_id == str(query["passage_id"]):
            continue
        candidate_start = int(candidate["start_unit"])
        candidate_end = int(candidate["end_unit"])
        gap = max(query_start - candidate_end, candidate_start - query_end, 0)
        structural_rows.append((candidate_id, gap <= 1, gap, candidate_start))
    structural_rows.sort(key=lambda row: (not row[1], row[2], row[3], row[0]))
    return [(candidate_id, adjacent) for candidate_id, adjacent, _, _ in structural_rows]


def _source_labels(candidate: NeighborCandidate) -> list[str]:
    labels = []
    if candidate.bm25_rank is not None:
        labels.append("bm25")
    if candidate.dense_rank is not None:
        labels.append("dense")
    if candidate.structural_rank is not None:
        labels.append("structural")
    return labels


def _candidate_strength(candidate: NeighborCandidate) -> tuple[float, int, str]:
    return (
        -candidate.rrf_score,
        candidate.structural_rank if candidate.structural_rank is not None else 2**31 - 1,
        candidate.candidate_id,
    )


def _deduplicate_candidates(
    candidates: list[NeighborCandidate],
    metadata: dict[str, dict[str, Any]],
    excluded_ids: set[str],
    target_content: str,
) -> list[NeighborCandidate]:
    filtered = [
        candidate
        for candidate in candidates
        if candidate.candidate_id not in excluded_ids and str(metadata[candidate.candidate_id]["content"]) != target_content
    ]
    filtered.sort(key=lambda candidate: (not candidate.adjacent, *_candidate_strength(candidate)))
    unique: list[NeighborCandidate] = []
    seen_content: set[str] = set()
    for candidate in filtered:
        content = str(metadata[candidate.candidate_id]["content"])
        if content not in seen_content:
            seen_content.add(content)
            unique.append(candidate)
    return unique


def _select_contexts(
    candidates: list[NeighborCandidate],
    metadata: dict[str, dict[str, Any]],
    target_article_id: str,
    target_id: str,
) -> list[tuple[NeighborCandidate, str]]:
    adjacent_siblings = [
        candidate
        for candidate in candidates
        if candidate.adjacent and str(metadata[candidate.candidate_id]["article_id"]) == target_article_id
    ]
    if not adjacent_siblings:
        raise ValueError(f"target {target_id} has no eligible adjacent same-article context passage")
    selected: list[tuple[NeighborCandidate, str]] = [(min(adjacent_siblings, key=_candidate_strength), "adjacent_sibling")]

    non_siblings = [
        candidate
        for candidate in candidates
        if str(metadata[candidate.candidate_id]["article_id"]) != target_article_id and candidate != selected[0][0]
    ]
    if not non_siblings:
        raise ValueError(f"target {target_id} has no eligible non-sibling context passage")
    selected.append((min(non_siblings, key=_candidate_strength), "strongest_non_sibling"))

    selected_ids = {candidate.candidate_id for candidate, _ in selected}
    remaining = [candidate for candidate in candidates if candidate.candidate_id not in selected_ids]
    if not remaining:
        raise ValueError(f"target {target_id} has fewer than three eligible context passages")
    selected.append((min(remaining, key=_candidate_strength), "strongest_remaining"))
    return selected


def build_generation_context(
    targets: pl.DataFrame,
    passages: pl.DataFrame,
    bm25_index: BM25Index,
    dense_index: DenseIndex,
    config: GenerationContextConfig | None = None,
) -> pl.DataFrame:
    """Build three collision-context passages for every row in a target subset."""
    if not isinstance(targets, pl.DataFrame) or not isinstance(passages, pl.DataFrame):
        raise TypeError("targets and passages must be polars DataFrames")
    _require_columns(targets, "targets", {"passage_id", "article_id", "content", "serialized_passage"})
    _require_columns(passages, "passages", {"passage_id", "article_id", "content", "serialized_passage", "start_unit", "end_unit"})
    active_config = config or GenerationContextConfig()
    if targets.is_empty():
        raise ValueError("targets must contain at least one target row")

    metadata = {str(row["passage_id"]): row for row in passages.to_dicts()}
    if len(metadata) != passages.height:
        raise ValueError("passages contains duplicate passage_id values")
    target_rows = targets.to_dicts()
    submitted_target_ids = {str(row["passage_id"]) for row in target_rows}
    missing_targets = sorted(submitted_target_ids - set(metadata))
    if missing_targets:
        raise ValueError(f"targets contain passage IDs absent from passages: {', '.join(missing_targets[:5])}")
    content_counts: dict[str, int] = defaultdict(int)
    for row in metadata.values():
        content_counts[_normalized_content(row["content"])] += 1
    duplicate_targets = sorted(
        str(row["passage_id"]) for row in target_rows if content_counts[_normalized_content(row["content"])] > 1
    )
    if duplicate_targets:
        raise ValueError("targets contain duplicate normalized content in the source corpus: " + ", ".join(duplicate_targets[:5]))

    article_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata.values():
        article_members[str(row["article_id"])].append(row)

    target_passage_ids = [str(row["passage_id"]) for row in target_rows]
    target_texts = [str(row["serialized_passage"]) for row in target_rows]
    bm25_hits = bm25_index.search(target_texts, k=active_config.bm25_k)
    dense_indices = dense_index.indices_for_ids(target_passage_ids)
    dense_hits = dense_index.search_embeddings(
        dense_index.embedding_rows(dense_indices),
        k=active_config.dense_k,
        block_size=active_config.dense_block_size,
        exclude_ids=target_passage_ids,
        device=active_config.device,
    )

    records: list[dict[str, Any]] = []
    for target, lexical_hits, semantic_hits in zip(target_rows, bm25_hits, dense_hits, strict=True):
        target_id = str(target["passage_id"])
        target_meta = metadata[target_id]
        candidates = merge_neighbors(
            target_id,
            lexical_hits,
            semantic_hits,
            rrf_k=active_config.rrf_k,
            limit=active_config.union_k,
        )
        candidates = include_structural_neighbors(candidates, _structural_members(target_meta, article_members))
        candidates = _deduplicate_candidates(candidates, metadata, {target_id}, str(target_meta["content"]))
        selected = _select_contexts(candidates, metadata, str(target_meta["article_id"]), target_id)
        for context_index, (candidate, selection_source) in enumerate(selected):
            candidate_meta = metadata[candidate.candidate_id]
            record = {f"target_{key}": value for key, value in target.items()}
            record.update({f"candidate_{key}": value for key, value in candidate_meta.items()})
            record.update(
                {
                    "context_index": context_index,
                    "same_article": str(candidate_meta["article_id"]) == str(target_meta["article_id"]),
                    "adjacent": candidate.adjacent,
                    "rrf_score": candidate.rrf_score,
                    "bm25_rank": candidate.bm25_rank,
                    "bm25_score": candidate.bm25_score,
                    "dense_rank": candidate.dense_rank,
                    "dense_score": candidate.dense_score,
                    "structural_rank": candidate.structural_rank,
                    "source_labels": _source_labels(candidate),
                    "selection_source": selection_source,
                }
            )
            records.append(record)

    context = pl.DataFrame(records).sort(["target_passage_id", "context_index"])
    if active_config.output_dir is not None:
        output_dir = Path(active_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "generation_context.parquet"
        temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
        context.write_parquet(temporary, compression="zstd")
        os.replace(temporary, output_path)
    return context


__all__ = ["GenerationContextConfig", "build_generation_context"]


def _normalized_content(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value))).strip().casefold()
