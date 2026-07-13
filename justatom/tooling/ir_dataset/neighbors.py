from __future__ import annotations

import os
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import polars as pl

from justatom.tooling.ir_dataset.dense import DenseIndex, DenseSearchHit
from justatom.tooling.ir_dataset.sparse import BM25Index, SearchHit


@dataclass(frozen=True, slots=True)
class NeighborCandidate:
    candidate_id: str
    rrf_score: float
    bm25_rank: int | None
    bm25_score: float | None
    dense_rank: int | None
    dense_score: float | None
    structural_rank: int | None = None
    adjacent: bool = False


@dataclass(frozen=True, slots=True)
class NeighborBuildConfig:
    bm25_k: int = 20
    dense_k: int = 20
    union_k: int = 30
    rrf_k: int = 60
    query_passages: int = 200
    dense_block_size: int = 65_536
    device: str = "mps"


@dataclass(frozen=True, slots=True)
class NeighborSummary:
    path: Path
    query_count: int
    row_count: int
    bm25_contributions: int
    dense_contributions: int
    structural_contributions: int


def merge_neighbors(
    query_id: str,
    bm25_hits: Sequence[SearchHit],
    dense_hits: Sequence[DenseSearchHit],
    *,
    rrf_k: int = 60,
    limit: int = 30,
) -> list[NeighborCandidate]:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be > 0")
    if limit <= 0:
        raise ValueError("neighbor limit must be > 0")
    merged: dict[str, dict[str, float | int | None]] = {}
    for source, hits in (("bm25", bm25_hits), ("dense", dense_hits)):
        for hit in hits:
            if hit.passage_id == query_id:
                continue
            row = merged.setdefault(
                hit.passage_id,
                {
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "dense_rank": None,
                    "dense_score": None,
                },
            )
            row["rrf_score"] = float(row["rrf_score"]) + 1.0 / (rrf_k + hit.rank)
            row[f"{source}_rank"] = int(hit.rank)
            row[f"{source}_score"] = float(hit.score)

    candidates = [NeighborCandidate(candidate_id=passage_id, **values) for passage_id, values in merged.items()]
    candidates.sort(key=lambda row: (-row.rrf_score, row.candidate_id))
    return candidates[:limit]


def include_structural_neighbors(
    ranked: Sequence[NeighborCandidate],
    structural: Sequence[tuple[str, bool]],
) -> list[NeighborCandidate]:
    output = list(ranked)
    positions = {row.candidate_id: index for index, row in enumerate(output)}
    for structural_rank, (candidate_id, adjacent) in enumerate(structural, start=1):
        if candidate_id in positions:
            index = positions[candidate_id]
            output[index] = replace(
                output[index],
                structural_rank=structural_rank,
                adjacent=bool(adjacent),
            )
            continue
        positions[candidate_id] = len(output)
        output.append(
            NeighborCandidate(
                candidate_id=candidate_id,
                rrf_score=0.0,
                bm25_rank=None,
                bm25_score=None,
                dense_rank=None,
                dense_score=None,
                structural_rank=structural_rank,
                adjacent=bool(adjacent),
            )
        )
    return output


def select_query_passages(frame: pl.DataFrame, count: int) -> pl.DataFrame:
    if count <= 0:
        raise ValueError("query passage count must be > 0")
    query_rows = frame.select(
        "article_id",
        pl.col("passage_id").alias("query_id"),
        pl.col("start_unit").alias("query_start"),
        pl.col("end_unit").alias("query_end"),
    )
    candidate_rows = frame.select(
        "article_id",
        pl.col("passage_id").alias("candidate_id"),
        pl.col("start_unit").alias("candidate_start"),
        pl.col("end_unit").alias("candidate_end"),
    )
    eligible_query_ids = (
        query_rows.join(candidate_rows, on="article_id", how="inner")
        .filter(pl.col("query_id") != pl.col("candidate_id"))
        .filter(
            pl.max_horizontal(
                pl.col("query_start") - pl.col("candidate_end"),
                pl.col("candidate_start") - pl.col("query_end"),
                pl.lit(0),
            )
            <= 1
        )
        .select(pl.col("query_id").alias("passage_id"))
        .unique()
    )
    return (
        frame.join(eligible_query_ids, on="passage_id", how="semi")
        .sort("corpus_rank")
        .head(count)
    )


def build_neighbor_artifact(
    *,
    passages_path: Path,
    bm25_index: BM25Index,
    dense_index: DenseIndex,
    output_path: Path,
    config: NeighborBuildConfig,
) -> NeighborSummary:
    frame = pl.read_parquet(passages_path).sort("corpus_rank")
    queries = select_query_passages(frame, config.query_passages)
    query_count = queries.height
    if query_count == 0:
        raise RuntimeError("No passages with same-article corpus siblings are available for neighbor diagnostics")
    query_ids = queries["passage_id"].to_list()
    query_texts = queries["serialized_passage"].to_list()
    dense_indices = dense_index.indices_for_ids(query_ids)
    dense_hits = dense_index.search_embeddings(
        dense_index.embedding_rows(dense_indices),
        k=config.dense_k,
        block_size=config.dense_block_size,
        exclude_ids=query_ids,
        device=config.device,
    )
    bm25_hits = bm25_index.search(query_texts, k=min(config.bm25_k + 1, frame.height))

    metadata = {
        row["passage_id"]: row
        for row in frame.select(
            "passage_id",
            "article_id",
            "title",
            "section",
            "content",
            "flows",
            "hubs",
            "start_unit",
            "end_unit",
        ).iter_rows(named=True)
    }
    article_members: dict[str, list[dict]] = defaultdict(list)
    for row in metadata.values():
        article_members[row["article_id"]].append(row)
    records: list[dict] = []
    bm25_contributions = 0
    dense_contributions = 0
    structural_contributions = 0
    for query_id, lexical, semantic in zip(query_ids, bm25_hits, dense_hits, strict=True):
        query_meta = metadata[query_id]
        candidates = merge_neighbors(
            query_id,
            lexical,
            semantic,
            rrf_k=config.rrf_k,
            limit=config.union_k,
        )
        structural_rows = []
        query_start = int(query_meta["start_unit"])
        query_end = int(query_meta["end_unit"])
        for row in article_members[query_meta["article_id"]]:
            if row["passage_id"] == query_id:
                continue
            candidate_start = int(row["start_unit"])
            candidate_end = int(row["end_unit"])
            gap = max(query_start - candidate_end, candidate_start - query_end, 0)
            structural_rows.append((row["passage_id"], gap <= 1, gap, candidate_start))
        structural_rows.sort(key=lambda row: (not row[1], row[2], row[3], row[0]))
        candidates = include_structural_neighbors(
            candidates,
            [(candidate_id, adjacent) for candidate_id, adjacent, _, _ in structural_rows],
        )
        for candidate in candidates:
            candidate_meta = metadata[candidate.candidate_id]
            bm25_contributions += int(candidate.bm25_rank is not None)
            dense_contributions += int(candidate.dense_rank is not None)
            structural_contributions += int(candidate.structural_rank is not None)
            records.append(
                {
                    "query_id": query_id,
                    "query_article_id": query_meta["article_id"],
                    "candidate_id": candidate.candidate_id,
                    "candidate_article_id": candidate_meta["article_id"],
                    "same_article": query_meta["article_id"] == candidate_meta["article_id"],
                    "rrf_score": candidate.rrf_score,
                    "bm25_rank": candidate.bm25_rank,
                    "bm25_score": candidate.bm25_score,
                    "dense_rank": candidate.dense_rank,
                    "dense_score": candidate.dense_score,
                    "structural_rank": candidate.structural_rank,
                    "adjacent": candidate.adjacent,
                    "query_title": query_meta["title"],
                    "query_section": query_meta["section"],
                    "query_preview": query_meta["content"][:500],
                    "candidate_title": candidate_meta["title"],
                    "candidate_section": candidate_meta["section"],
                    "candidate_preview": candidate_meta["content"][:500],
                    "query_flows": query_meta["flows"],
                    "candidate_flows": candidate_meta["flows"],
                    "query_hubs": query_meta["hubs"],
                    "candidate_hubs": candidate_meta["hubs"],
                }
            )
    if not records:
        raise RuntimeError("No neighbor rows were produced")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    pl.DataFrame(records).write_parquet(temporary, compression="zstd")
    os.replace(temporary, output_path)
    return NeighborSummary(
        path=output_path,
        query_count=query_count,
        row_count=len(records),
        bm25_contributions=bm25_contributions,
        dense_contributions=dense_contributions,
        structural_contributions=structural_contributions,
    )


__all__ = [
    "NeighborBuildConfig",
    "NeighborCandidate",
    "NeighborSummary",
    "build_neighbor_artifact",
    "include_structural_neighbors",
    "merge_neighbors",
    "select_query_passages",
]
