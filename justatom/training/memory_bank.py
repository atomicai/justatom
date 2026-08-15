from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

from justatom.training.config import MarginConfig, MarginMode, MemoryBankConfig
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

    def __init__(self, config: MemoryBankConfig):
        if not isinstance(config, MemoryBankConfig):
            raise TypeError("ContrastiveMemoryBank requires MemoryBankConfig")
        if config.mining not in {"all", "random", "hard", "mixed"}:
            raise ValueError("memory_bank.mining must be one of: all, random, hard, mixed")
        if config.adaptive.collision_beta <= 0.0:
            raise ValueError("memory_bank.adaptive.collision_beta must be positive")

        self.config = config
        self.embeddings: torch.Tensor | None = None
        self.doc_key_ids: torch.Tensor | None = None
        self.content_key_ids: torch.Tensor | None = None
        self.query_key_ids: torch.Tensor | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.config.size > 0

    @property
    def current_size(self) -> int:
        return 0 if self.embeddings is None else int(self.embeddings.shape[0])

    def _base_metrics(self) -> dict[str, float | str]:
        metrics: dict[str, float | str] = {
            "memory/capacity": float(self.config.size),
            "memory/size": float(self.current_size),
            "memory/warmup_steps": float(self.config.warmup_steps),
            "memory/mass_ratio": float(self.config.mass_ratio),
            "memory/mass_ramp": 0.0,
            "memory/effective_mass_ratio": 0.0,
            "memory/mining": self.config.mining,
            "memory/hard_k": 0.0,
            "memory/random_k": 0.0,
            "memory/valid_negatives_mean": 0.0,
            "memory/active_negatives_mean": 0.0,
            "memory/active_hard_negatives_mean": 0.0,
            "memory/active_random_negatives_mean": 0.0,
            "memory/positive_similarity_mean": float("nan"),
        }
        metrics.update(scalar_distribution("memory/active_count", torch.zeros(1)))
        metrics.update(scalar_distribution("memory/active_similarity", torch.empty(0)))
        metrics.update(scalar_distribution("memory/candidate_mass_weight", torch.empty(0)))
        return metrics

    def _noop_selection(self) -> MemorySelection:
        return MemorySelection(
            embeddings=None,
            active_mask=None,
            log_weights=None,
            collision_g=None,
            hard_weights=None,
            metrics=self._base_metrics(),
        )

    def _mass_progress(self, step: int) -> float:
        if step < self.config.warmup_steps:
            return 0.0
        offset = int(step) - self.config.warmup_steps + 1
        return min(max(offset / float(self.config.mass_ramp_steps), 0.0), 1.0)

    def _effective_mass_ratio(self, step: int) -> float:
        return float(self.config.mass_ratio) * self._mass_progress(step)

    @torch.no_grad()
    def select(
        self,
        batch: dict[str, torch.Tensor],
        *,
        query_vectors: torch.Tensor,
        positive_vectors: torch.Tensor,
        step: int,
    ) -> MemorySelection:
        if self._effective_mass_ratio(step) == 0.0:
            return self._noop_selection()
        if not self.enabled or self.embeddings is None or self.current_size == 0:
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

        if self.config.mining == "all":
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

        candidate_log_weights = None
        if hard_weights is not None:
            candidate_log_weights = torch.zeros_like(bank_similarities)
            row_weights = torch.log(hard_weights.clamp_min(1e-8)).view(-1, 1)
            candidate_log_weights = torch.where(
                hard_active,
                row_weights.expand_as(candidate_log_weights),
                candidate_log_weights,
            )

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
        log_weights, mass_metrics = self._normalized_log_weights(active, candidate_log_weights, step=step)
        metrics.update(mass_metrics)
        return MemorySelection(
            embeddings=embeddings,
            active_mask=active,
            log_weights=log_weights.to(dtype=query_vectors.dtype),
            collision_g=collision_g,
            hard_weights=hard_weights,
            metrics=metrics,
        )

    def _normalized_log_weights(
        self,
        active: torch.Tensor,
        candidate_log_weights: torch.Tensor | None,
        *,
        step: int,
    ) -> tuple[torch.Tensor, dict[str, float | str]]:
        batch_size = int(active.shape[0])
        if batch_size < 2:
            raise ValueError("memory bank requires contrastive batch size >= 2")
        counts = active.sum(dim=1)
        safe_counts = counts.clamp_min(1).float()
        ratio = self._effective_mass_ratio(step)
        if ratio == 0.0:
            normalized = torch.zeros(active.shape, device=active.device, dtype=safe_counts.dtype)
        else:
            row_log_weight = math.log(ratio) + math.log(batch_size - 1) - safe_counts.log()
            normalized = torch.where(
                active,
                row_log_weight.view(-1, 1).expand_as(active),
                torch.zeros(active.shape, device=active.device, dtype=row_log_weight.dtype),
            )
            if candidate_log_weights is not None:
                normalized = normalized + torch.where(
                    active,
                    candidate_log_weights.to(normalized),
                    torch.zeros_like(normalized),
                )

        metrics = self._base_metrics()
        metrics.update(
            {
                "memory/mass_ramp": self._mass_progress(step),
                "memory/effective_mass_ratio": ratio,
            }
        )
        metrics.update(scalar_distribution("memory/active_count", counts))
        metrics.update(scalar_distribution("memory/candidate_mass_weight", normalized[active].exp()))
        return normalized, metrics

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

    def enqueue(self, vectors: torch.Tensor, batch: dict[str, torch.Tensor]) -> None:
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
        return round(float(self.config.hard_negatives) * progress)

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
