from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from justatom.training.config import MemoryBankConfig, RerankerCacheConfig, RerankerConfig
from justatom.training.memory_bank import ContrastiveMemoryBank
from justatom.training.reranker import CachedTextReranker


class FakePairScorer:
    fingerprint = "fake-v1"

    def __init__(self, scores: dict[str, float] | None = None):
        self.scores = scores or {}
        self.calls: list[list[tuple[str, str]]] = []
        self.closed = False

    def score_pairs(self, queries, documents):
        pairs = list(zip(queries, documents))
        self.calls.append(pairs)
        return [self.scores.get(document, 0.25) for _, document in pairs]

    def close(self):
        self.closed = True


def reranker_config(**updates):
    config = RerankerConfig(
        enabled=True,
        prefilter_hard_negatives=3,
        prefilter_random_negatives=0,
        negatives=1,
        min_score_gap=0.1,
        cache=RerankerCacheConfig(mode="off"),
    )
    if "cache" not in updates:
        return replace(config, **updates)
    return replace(config, **updates)


def test_cached_reranker_persists_scores_across_instances(tmp_path):
    path = tmp_path / "scores.sqlite"
    config = reranker_config(cache=RerankerCacheConfig(mode="read-write", path=str(path), on_miss="score"))
    first_backend = FakePairScorer({"doc": 0.75})
    first = CachedTextReranker(config, backend=first_backend)

    scored = first.score_pairs(["query"], ["doc"])
    first.close()

    assert scored.values == (0.75,)
    assert scored.cache_hits == 0 and scored.cache_misses == 1
    assert first_backend.calls == [[("query", "doc")]]

    second_backend = FakePairScorer({"doc": 0.1})
    second = CachedTextReranker(config, backend=second_backend)
    cached = second.score_pairs(["query"], ["doc"])
    second.close()

    assert cached.values == (0.75,)
    assert cached.cache_hits == 1 and cached.cache_misses == 0
    assert second_backend.calls == []


def test_read_only_cache_fails_on_unknown_pair(tmp_path):
    path = tmp_path / "scores.sqlite"
    writable = reranker_config(cache=RerankerCacheConfig(mode="read-write", path=str(path), on_miss="score"))
    scorer = CachedTextReranker(writable, backend=FakePairScorer())
    scorer.score_pairs(["known"], ["doc"])
    scorer.close()
    read_only = replace(
        writable,
        cache=RerankerCacheConfig(mode="read-only", path=str(path), on_miss="error"),
    )
    scorer = CachedTextReranker(read_only, backend=FakePairScorer())

    with pytest.raises(RuntimeError, match="cache miss"):
        scorer.score_pairs(["unknown"], ["doc"])
    scorer.close()


def test_teacher_filters_ambiguous_candidates_then_keeps_hardest_safe_negative():
    config = reranker_config()
    backend = FakePairScorer(
        {
            "positive": 0.9,
            "ambiguous": 0.95,
            "hard-safe": 0.7,
            "easy-safe": 0.1,
        }
    )
    reranker = CachedTextReranker(config, backend=backend)
    bank = ContrastiveMemoryBank(
        MemoryBankConfig(enabled=True, size=3, mining="random", random_negatives=1),
        reranker=reranker,
    )
    bank.enqueue(
        F.normalize(torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]), dim=-1),
        {"doc_key_id": torch.tensor([1, 2, 3])},
        document_texts=["ambiguous", "hard-safe", "easy-safe"],
    )

    selection = bank.select(
        batch={"doc_key_id": torch.tensor([99])},
        query_vectors=F.normalize(torch.tensor([[1.0, 0.0]]), dim=-1),
        positive_vectors=F.normalize(torch.tensor([[1.0, 0.0]]), dim=-1),
        step=0,
        query_texts=["query"],
        positive_texts=["positive"],
    )

    assert selection.active_mask is not None
    assert selection.active_mask.tolist() == [[False, True, False]]
    assert selection.metrics["reranker/safe_negatives_mean"] == 2.0
    assert selection.metrics["reranker/ambiguous_mean"] == 1.0
    assert selection.metrics["reranker/above_positive_mean"] == 1.0
    assert selection.metrics["memory/active_negatives_mean"] == 1.0


def test_disabled_reranker_keeps_original_bank_selection_without_text_metadata():
    bank = ContrastiveMemoryBank(MemoryBankConfig(enabled=True, size=2, mining="all"))
    bank.enqueue(F.normalize(torch.eye(2), dim=-1), {"doc_key_id": torch.tensor([1, 2])})

    selection = bank.select(
        batch={"doc_key_id": torch.tensor([99])},
        query_vectors=F.normalize(torch.tensor([[1.0, 0.0]]), dim=-1),
        positive_vectors=F.normalize(torch.tensor([[1.0, 0.0]]), dim=-1),
        step=0,
    )

    assert bank.documents is None
    assert selection.active_mask is not None
    assert selection.active_mask.tolist() == [[True, True]]
    assert not any(key.startswith("reranker/") for key in selection.metrics)
