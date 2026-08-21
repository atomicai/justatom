from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ExactRankResult:
    """Exact ranks for a benchmark with one relevant document per query."""

    ranks: np.ndarray
    positive_indices: np.ndarray
    tie_policy: str = "corpus_order"

    def metrics(self) -> dict[str, float | int]:
        return single_positive_metrics(self.ranks)


def _float_matrix(name: str, values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty rank-2 matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _positive_index_vector(values: Any, *, query_count: int, corpus_count: int) -> np.ndarray:
    indices = np.asarray(values, dtype=np.int64)
    if indices.shape != (query_count,):
        raise ValueError(f"positive_indices must have shape ({query_count},)")
    if np.any(indices < 0) or np.any(indices >= corpus_count):
        raise ValueError("positive_indices contains an index outside the corpus")
    return indices


@torch.inference_mode()
def exact_single_positive_ranks(
    query_embeddings: Any,
    corpus_embeddings: Any,
    positive_indices: Any,
    *,
    device: str | torch.device = "cpu",
    query_batch_size: int = 64,
    corpus_block_size: int = 8192,
) -> ExactRankResult:
    """Compute exact cosine ranks without materialising the full query-corpus matrix.

    Scores are compared in float32. Equal scores are ordered by their position in
    the corpus split, which makes ranks deterministic across block sizes.
    """

    queries = _float_matrix("query_embeddings", query_embeddings)
    corpus = _float_matrix("corpus_embeddings", corpus_embeddings)
    if queries.shape[1] != corpus.shape[1]:
        raise ValueError("query and corpus embedding dimensions must match")
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    if corpus_block_size <= 0:
        raise ValueError("corpus_block_size must be positive")

    positive = _positive_index_vector(
        positive_indices,
        query_count=queries.shape[0],
        corpus_count=corpus.shape[0],
    )
    resolved_device = torch.device(device)
    ranks = np.ones(queries.shape[0], dtype=np.int64)

    for query_start in range(0, queries.shape[0], query_batch_size):
        query_stop = min(query_start + query_batch_size, queries.shape[0])
        query_block = torch.tensor(queries[query_start:query_stop], dtype=torch.float32, device=resolved_device)
        query_block = F.normalize(query_block.float(), p=2, dim=-1, eps=1e-8)
        positive_block = positive[query_start:query_stop]
        positive_vectors = torch.tensor(corpus[positive_block], dtype=torch.float32, device=resolved_device)
        positive_vectors = F.normalize(positive_vectors.float(), p=2, dim=-1, eps=1e-8)
        positive_scores = (query_block * positive_vectors).sum(dim=-1)
        block_ranks = torch.ones(query_stop - query_start, dtype=torch.long, device=resolved_device)
        positive_positions = torch.from_numpy(positive_block).to(resolved_device)

        for corpus_start in range(0, corpus.shape[0], corpus_block_size):
            corpus_stop = min(corpus_start + corpus_block_size, corpus.shape[0])
            document_block = torch.tensor(
                corpus[corpus_start:corpus_stop],
                dtype=torch.float32,
                device=resolved_device,
            )
            document_block = F.normalize(document_block.float(), p=2, dim=-1, eps=1e-8)
            scores = query_block @ document_block.T
            corpus_positions = torch.arange(corpus_start, corpus_stop, device=resolved_device)
            is_positive = corpus_positions[None, :] == positive_positions[:, None]
            # The positive score above and its cell in the GEMM can differ by a
            # few float32 ULPs. Excluding the positive explicitly prevents it
            # from ever outranking itself because of that implementation detail.
            block_ranks += ((scores > positive_scores[:, None]) & ~is_positive).sum(dim=-1)

            tied_before = (scores == positive_scores[:, None]) & (corpus_positions[None, :] < positive_positions[:, None])
            block_ranks += tied_before.sum(dim=-1)

        ranks[query_start:query_stop] = block_ranks.cpu().numpy()

    return ExactRankResult(ranks=ranks, positive_indices=positive)


def single_positive_metrics(ranks: Any) -> dict[str, float | int]:
    values = np.asarray(ranks, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("ranks must be a non-empty vector")
    if np.any(values <= 0):
        raise ValueError("ranks must be positive")

    reciprocal_at_10 = np.where(values <= 10, 1.0 / values, 0.0)
    ndcg_at_10 = np.where(values <= 10, 1.0 / np.log2(values + 1.0), 0.0)
    return {
        "queries": int(values.size),
        "hit_at_1": float(np.mean(values <= 1)),
        "recall_at_5": float(np.mean(values <= 5)),
        "recall_at_10": float(np.mean(values <= 10)),
        "mrr_at_10": float(np.mean(reciprocal_at_10)),
        "ndcg_at_10": float(np.mean(ndcg_at_10)),
        "mrr": float(np.mean(1.0 / values)),
        "mean_rank": float(np.mean(values)),
        "median_rank": float(np.median(values)),
    }


__all__ = ["ExactRankResult", "exact_single_positive_ranks", "single_positive_metrics"]
