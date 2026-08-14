from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

from justatom.training.config import MarginConfig, MarginMode, MemoryBankConfig
from justatom.training.reranker import CachedTextReranker
from justatom.training.telemetry import scalar_distribution


@dataclass(frozen=True)
class MemorySelection:
    embeddings: torch.Tensor | None
    active_mask: torch.Tensor | None
    log_weights: torch.Tensor | None
    collision_g: torch.Tensor | None
    hard_weights: torch.Tensor | None
    metrics: dict[str, float | torch.Tensor | str]


class QueryMarginHead(nn.Module):
    """Learn a bounded per-query admission margin m(q)."""

    def __init__(self, embedding_dim: int, config: MarginConfig):
        super().__init__()
        if config.mode is not MarginMode.QUERY:
            raise ValueError("QueryMarginHead requires memory_bank.margin.mode=query")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        hidden_dim = max(32, min(256, embedding_dim // 2))
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.config = config
        self.embedding_dim = embedding_dim

    def forward(self, queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if queries.ndim != 2 or queries.shape[-1] != self.embedding_dim:
            raise ValueError(f"queries must have shape [batch, {self.embedding_dim}], got {tuple(queries.shape)}")
        delta = self.config.scale * torch.tanh(self.network(queries)).squeeze(-1)
        raw = self.config.base + delta
        return raw, raw.clamp(self.config.minimum, self.config.maximum)


class ContrastiveMemoryBank:
    """Typed FIFO bank of detached document embeddings used as extra negatives."""

    def __init__(
        self,
        config: MemoryBankConfig,
        *,
        reranker: CachedTextReranker | None = None,
        contrastive_temperature: float = 0.05,
    ):
        if not isinstance(config, MemoryBankConfig):
            raise TypeError("ContrastiveMemoryBank requires MemoryBankConfig")
        if config.mining not in {"all", "random", "hard", "mixed"}:
            raise ValueError("memory_bank.mining must be one of: all, random, hard, mixed")
        if config.adaptive.collision_beta <= 0.0:
            raise ValueError("memory_bank.adaptive.collision_beta must be positive")
        if float(contrastive_temperature) <= 0.0:
            raise ValueError("contrastive_temperature must be positive")

        self.config = config
        self.reranker = reranker
        self.contrastive_temperature = float(contrastive_temperature)
        self.embeddings: torch.Tensor | None = None
        self.doc_key_ids: torch.Tensor | None = None
        self.content_key_ids: torch.Tensor | None = None
        self.query_key_ids: torch.Tensor | None = None
        self.documents: list[str] | None = [] if reranker is not None else None

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.config.size > 0

    @property
    def current_size(self) -> int:
        return 0 if self.embeddings is None else int(self.embeddings.shape[0])

    @property
    def reranker_enabled(self) -> bool:
        return self.reranker is not None

    def _base_metrics(self) -> dict[str, float | str]:
        return {
            "memory/capacity": float(self.config.size),
            "memory/size": float(self.current_size),
            "memory/warmup_steps": float(self.config.warmup_steps),
            "memory/mining": self.config.mining,
            "memory/hard_k": 0.0,
            "memory/random_k": 0.0,
            "memory/valid_negatives_mean": 0.0,
            "memory/active_negatives_mean": 0.0,
            "memory/active_hard_negatives_mean": 0.0,
            "memory/active_random_negatives_mean": 0.0,
        }

    def _noop_selection(self) -> MemorySelection:
        metrics = self._base_metrics()
        if self.reranker is not None:
            metrics.update(self._base_reranker_metrics())
        return MemorySelection(
            embeddings=None,
            active_mask=None,
            log_weights=None,
            collision_g=None,
            hard_weights=None,
            metrics=metrics,
        )

    @staticmethod
    def _base_reranker_metrics() -> dict[str, float]:
        metrics = {
            "reranker/candidates_mean": 0.0,
            "reranker/safe_negatives_mean": 0.0,
            "reranker/ambiguous_mean": 0.0,
            "reranker/above_positive_mean": 0.0,
            "reranker/cache_hits": 0.0,
            "reranker/cache_misses": 0.0,
            "reranker/cache_hit_rate": 0.0,
            "reranker/skipped": 0.0,
        }
        metrics.update(scalar_distribution("reranker/teacher_weight", torch.empty(0)))
        metrics.update(scalar_distribution("reranker/selected_teacher_weight", torch.empty(0)))
        return metrics

    @torch.no_grad()
    def select(
        self,
        batch: dict[str, torch.Tensor],
        *,
        query_vectors: torch.Tensor,
        positive_vectors: torch.Tensor,
        step: int,
        query_texts: Sequence[str] | None = None,
        positive_texts: Sequence[str] | None = None,
    ) -> MemorySelection:
        if not self.enabled or self.embeddings is None or self.current_size == 0 or int(step) < self.config.warmup_steps:
            return self._noop_selection()
        if query_vectors.ndim != 2 or positive_vectors.shape != query_vectors.shape:
            raise ValueError("query_vectors and positive_vectors must have matching [batch, dim] shapes")

        embeddings = self.embeddings.to(device=query_vectors.device, dtype=query_vectors.dtype)
        valid = self._valid_identity_mask(batch, query_vectors.shape[0], embeddings.shape[0], query_vectors.device)
        query_norm = F.normalize(query_vectors.detach(), p=2, dim=-1, eps=1e-8)
        positive_norm = F.normalize(positive_vectors.detach(), p=2, dim=-1, eps=1e-8)
        bank_norm = F.normalize(embeddings, p=2, dim=-1, eps=1e-8)
        bank_similarities = query_norm @ bank_norm.T
        positive_similarities = (query_norm * positive_norm).sum(dim=-1)

        collision_g, hard_weights = self._adaptive_values(
            valid,
            bank_similarities,
            positive_similarities,
        )
        hard_k = self._scheduled_hard_k(int(step))
        random_k = self.config.random_negatives
        hard_active = torch.zeros_like(valid)
        random_active = torch.zeros_like(valid)
        reranker_metrics: dict[str, float] = {}
        reranker_log_weights: torch.Tensor | None = None

        if self.reranker is not None:
            reranker_config = self.reranker.config
            hard_k = reranker_config.prefilter_hard_negatives
            random_k = reranker_config.prefilter_random_negatives
            hard_candidates = self._topk_mask(bank_similarities, valid, hard_k)
            random_candidates = self._random_mask(valid & ~hard_candidates, random_k)
            candidate_mask = hard_candidates | random_candidates
            active, reranker_log_weights, reranker_metrics = self._reranker_selection(
                candidate_mask=candidate_mask,
                bank_similarities=bank_similarities,
                query_texts=query_texts,
                positive_texts=positive_texts,
            )
            hard_active = active & hard_candidates
            random_active = active & random_candidates
        elif self.config.mining == "all":
            active = valid.clone()
            hard_k = 0
            random_k = 0
        else:
            active = torch.zeros_like(valid)
            if self.config.mining in {"hard", "mixed"} and hard_k > 0:
                hard_active = self._topk_mask(bank_similarities, valid, hard_k)
                active |= hard_active
            if self.config.mining in {"random", "mixed"} and random_k > 0:
                random_active = self._random_mask(valid, random_k)
                active |= random_active
            active &= valid

        log_weights = reranker_log_weights
        if hard_weights is not None:
            if log_weights is None:
                log_weights = torch.zeros_like(bank_similarities)
            row_weights = torch.log(hard_weights.clamp_min(1e-8)).view(-1, 1)
            adaptive_log_weights = torch.where(
                hard_active,
                row_weights.expand_as(log_weights),
                torch.zeros_like(log_weights),
            )
            log_weights = log_weights + adaptive_log_weights

        metrics = self._selection_metrics(
            valid=valid,
            active=active,
            hard_active=hard_active,
            random_active=random_active,
            bank_similarities=bank_similarities,
            positive_similarities=positive_similarities,
            collision_g=collision_g,
            hard_weights=hard_weights,
            hard_k=hard_k,
            random_k=random_k,
        )
        metrics.update(reranker_metrics)
        return MemorySelection(
            embeddings=embeddings,
            active_mask=active,
            log_weights=log_weights,
            collision_g=collision_g,
            hard_weights=hard_weights,
            metrics=metrics,
        )

    def _reranker_selection(
        self,
        *,
        candidate_mask: torch.Tensor,
        bank_similarities: torch.Tensor,
        query_texts: Sequence[str] | None,
        positive_texts: Sequence[str] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float]]:
        assert self.reranker is not None
        if query_texts is None or positive_texts is None:
            raise ValueError("reranker selection requires query_texts and positive_texts")
        batch_size, bank_size = candidate_mask.shape
        if len(query_texts) != batch_size or len(positive_texts) != batch_size:
            raise ValueError("reranker texts must match the query batch size")
        if self.documents is None or len(self.documents) != bank_size:
            raise RuntimeError("reranker document metadata is not aligned with memory-bank embeddings")

        pair_queries: list[str] = []
        pair_documents: list[str] = []
        row_candidates: list[list[int]] = []
        for row in range(batch_size):
            indices = candidate_mask[row].nonzero(as_tuple=False).view(-1).detach().cpu().tolist()
            row_candidates.append(indices)
            pair_queries.append(str(query_texts[row]))
            pair_documents.append(str(positive_texts[row]))
            pair_queries.extend(str(query_texts[row]) for _ in indices)
            pair_documents.extend(self.documents[index] for index in indices)

        score_result = self.reranker.score_pairs(pair_queries, pair_documents)
        active = torch.zeros_like(candidate_mask)
        teacher_log_weights = torch.zeros_like(bank_similarities) if self.reranker.config.strategy == "teacher_weighted" else None
        positive_scores: list[float] = []
        candidate_scores: list[float] = []
        score_gaps: list[float] = []
        teacher_weights: list[float] = []
        selected_teacher_weights: list[float] = []
        safe_counts: list[int] = []
        ambiguous_counts: list[int] = []
        above_positive_counts: list[int] = []
        offset = 0
        for row, indices in enumerate(row_candidates):
            positive_score = score_result.values[offset]
            offset += 1
            row_scores = score_result.values[offset : offset + len(indices)]
            offset += len(indices)
            if positive_score is None:
                safe_counts.append(0)
                ambiguous_counts.append(len(indices))
                above_positive_counts.append(0)
                continue
            positive_scores.append(positive_score)
            safe_indices: list[int] = []
            scored_indices: list[int] = []
            scored_log_weights: list[float] = []
            above_positive = 0
            for bank_index, candidate_score in zip(indices, row_scores):
                if candidate_score is None:
                    continue
                candidate_scores.append(candidate_score)
                score_gap = positive_score - candidate_score
                score_gaps.append(score_gap)
                above_positive += int(candidate_score >= positive_score)
                if candidate_score <= positive_score - self.reranker.config.min_score_gap:
                    safe_indices.append(bank_index)
                if self.reranker.config.strategy == "teacher_weighted":
                    scaled_gap = (score_gap - self.reranker.config.min_score_gap) / self.reranker.config.teacher_temperature
                    if scaled_gap >= 0.0:
                        confidence = 1.0 / (1.0 + math.exp(-scaled_gap))
                    else:
                        exp_gap = math.exp(scaled_gap)
                        confidence = exp_gap / (1.0 + exp_gap)
                    confidence = max(
                        self.reranker.config.teacher_weight_floor,
                        min(confidence, 1.0),
                    )
                    scored_indices.append(bank_index)
                    scored_log_weights.append(math.log(confidence))
                    teacher_weights.append(confidence)
            safe_counts.append(len(safe_indices))
            ambiguous_counts.append(len(indices) - len(safe_indices))
            above_positive_counts.append(above_positive)
            if self.reranker.config.strategy == "filter" and safe_indices:
                safe_tensor = torch.tensor(safe_indices, device=bank_similarities.device, dtype=torch.long)
                count = min(self.reranker.config.negatives, len(safe_indices))
                selected = bank_similarities[row, safe_tensor].topk(count).indices
                active[row, safe_tensor[selected]] = True
            elif self.reranker.config.strategy == "teacher_weighted" and scored_indices:
                assert teacher_log_weights is not None
                scored_tensor = torch.tensor(
                    scored_indices,
                    device=bank_similarities.device,
                    dtype=torch.long,
                )
                row_log_weights = torch.tensor(
                    scored_log_weights,
                    device=bank_similarities.device,
                    dtype=bank_similarities.dtype,
                )
                adjusted_logits = bank_similarities[row, scored_tensor] / self.contrastive_temperature + row_log_weights
                count = min(self.reranker.config.negatives, len(scored_indices))
                selected = adjusted_logits.topk(count).indices
                selected_indices = scored_tensor[selected]
                active[row, selected_indices] = True
                teacher_log_weights[row, selected_indices] = row_log_weights[selected]
                selected_teacher_weights.extend(row_log_weights[selected].exp().detach().cpu().tolist())

        total_cache = score_result.cache_hits + score_result.cache_misses
        metrics = self._base_reranker_metrics()
        metrics.update(
            {
                "reranker/candidates_mean": float(candidate_mask.float().sum(dim=1).mean().item()),
                "reranker/safe_negatives_mean": float(sum(safe_counts) / max(batch_size, 1)),
                "reranker/ambiguous_mean": float(sum(ambiguous_counts) / max(batch_size, 1)),
                "reranker/above_positive_mean": float(sum(above_positive_counts) / max(batch_size, 1)),
                "reranker/cache_hits": float(score_result.cache_hits),
                "reranker/cache_misses": float(score_result.cache_misses),
                "reranker/cache_hit_rate": float(score_result.cache_hits / total_cache) if total_cache else 0.0,
                "reranker/skipped": float(score_result.skipped),
            }
        )
        metrics.update(scalar_distribution("reranker/positive_score", torch.tensor(positive_scores)))
        metrics.update(scalar_distribution("reranker/candidate_score", torch.tensor(candidate_scores)))
        metrics.update(scalar_distribution("reranker/positive_candidate_gap", torch.tensor(score_gaps)))
        metrics.update(scalar_distribution("reranker/teacher_weight", torch.tensor(teacher_weights)))
        metrics.update(
            scalar_distribution(
                "reranker/selected_teacher_weight",
                torch.tensor(selected_teacher_weights),
            )
        )
        return active, teacher_log_weights, metrics

    def _valid_identity_mask(
        self,
        batch: dict[str, torch.Tensor],
        batch_size: int,
        bank_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        valid = torch.ones(batch_size, bank_size, dtype=torch.bool, device=device)
        for key, bank_values in (
            ("doc_key_id", self.doc_key_ids),
            ("content_key_id", self.content_key_ids),
            ("query_key_id", self.query_key_ids),
        ):
            if bank_values is None:
                continue
            current_values = batch.get(key)
            if current_values is None:
                logger.warning("memory bank disabled for this batch because {} is missing", key)
                return torch.zeros_like(valid)
            valid &= current_values.to(device).view(-1, 1) != bank_values.to(device).view(1, -1)
        return valid

    def _adaptive_values(
        self,
        valid: torch.Tensor,
        bank_similarities: torch.Tensor,
        positive_similarities: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.config.adaptive.enabled:
            return None, None
        has_candidate = valid.any(dim=1)
        bank_max = bank_similarities.masked_fill(~valid, float("-inf")).max(dim=1).values
        collision_g = torch.where(
            has_candidate,
            bank_max - positive_similarities,
            torch.zeros_like(positive_similarities),
        )
        raw_weights = torch.sigmoid((self.config.adaptive.collision_threshold - collision_g) / self.config.adaptive.collision_beta)
        hard_weights = torch.where(has_candidate, raw_weights, torch.ones_like(raw_weights))
        return collision_g, hard_weights

    def _selection_metrics(
        self,
        *,
        valid: torch.Tensor,
        active: torch.Tensor,
        hard_active: torch.Tensor,
        random_active: torch.Tensor,
        bank_similarities: torch.Tensor,
        positive_similarities: torch.Tensor,
        collision_g: torch.Tensor | None,
        hard_weights: torch.Tensor | None,
        hard_k: int,
        random_k: int,
    ) -> dict[str, float | str]:
        metrics = self._base_metrics()
        metrics.update(
            {
                "memory/hard_k": float(hard_k),
                "memory/random_k": float(random_k),
                "memory/valid_negatives_mean": float(valid.float().sum(dim=1).mean().item()),
                "memory/active_negatives_mean": float(active.float().sum(dim=1).mean().item()),
                "memory/active_hard_negatives_mean": float(hard_active.float().sum(dim=1).mean().item()),
                "memory/active_random_negatives_mean": float(random_active.float().sum(dim=1).mean().item()),
                "memory/positive_similarity_mean": float(positive_similarities.mean().item()),
            }
        )
        selected = bank_similarities[active]
        metrics.update(scalar_distribution("memory/active_similarity", selected))
        if collision_g is not None:
            metrics.update(scalar_distribution("memory/collision_g", collision_g))
        if hard_weights is not None:
            metrics.update(scalar_distribution("memory/hard_weight", hard_weights))
        return metrics

    def enqueue(
        self,
        vectors: torch.Tensor,
        batch: dict[str, torch.Tensor],
        *,
        document_texts: Sequence[str] | None = None,
    ) -> None:
        if not self.enabled or vectors.numel() == 0:
            return
        vectors = F.normalize(vectors.detach().clone(), p=2, dim=-1, eps=1e-8)
        if vectors.ndim != 2:
            raise ValueError(f"memory bank vectors must be 2D, got shape={tuple(vectors.shape)}")
        self.embeddings = self._append(self.embeddings, vectors)
        self.doc_key_ids = self._append_ids(self.doc_key_ids, batch.get("doc_key_id"), vectors.device)
        self.content_key_ids = self._append_ids(
            self.content_key_ids,
            batch.get("content_key_id"),
            vectors.device,
        )
        self.query_key_ids = self._append_ids(self.query_key_ids, batch.get("query_key_id"), vectors.device)
        if self.reranker is not None:
            if document_texts is None or len(document_texts) != vectors.shape[0]:
                raise ValueError("reranker-enabled memory bank requires one document text per vector")
            assert self.documents is not None
            self.documents = [*self.documents, *(str(text) for text in document_texts)][-self.config.size :]

    def close(self) -> None:
        if self.reranker is not None:
            self.reranker.close()

    def _append(self, previous: torch.Tensor | None, current: torch.Tensor) -> torch.Tensor:
        merged = current if previous is None else torch.cat([previous.to(current), current], dim=0)
        return merged[-self.config.size :].detach()

    def _append_ids(
        self,
        previous: torch.Tensor | None,
        current: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        if current is None:
            return None
        current = current.detach().clone().to(device).view(-1)
        merged = current if previous is None else torch.cat([previous.to(device), current], dim=0)
        return merged[-self.config.size :].detach()

    def _scheduled_hard_k(self, step: int) -> int:
        if self.config.hard_negatives <= 0 or step < self.config.hard_warmup_steps:
            return 0
        progress = min(
            max((step - self.config.hard_warmup_steps) / float(self.config.hard_ramp_steps), 0.0),
            1.0,
        )
        return int(round(float(self.config.hard_negatives) * progress))

    @staticmethod
    def _topk_mask(scores: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
        k = max(0, min(int(k), int(scores.shape[1])))
        if k == 0:
            return torch.zeros_like(valid)
        values, indices = scores.masked_fill(~valid, float("-inf")).topk(k, dim=1)
        active = torch.zeros_like(valid)
        active.scatter_(1, indices, torch.isfinite(values))
        return active & valid

    @staticmethod
    def _random_mask(valid: torch.Tensor, k: int) -> torch.Tensor:
        k = max(0, min(int(k), int(valid.shape[1])))
        if k == 0:
            return torch.zeros_like(valid)
        scores = torch.rand(valid.shape, device=valid.device).masked_fill(~valid, float("-inf"))
        values, indices = scores.topk(k, dim=1)
        active = torch.zeros_like(valid)
        active.scatter_(1, indices, torch.isfinite(values))
        return active & valid


__all__ = ["ContrastiveMemoryBank", "MemorySelection", "QueryMarginHead"]
