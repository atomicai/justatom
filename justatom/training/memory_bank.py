from __future__ import annotations

import torch
import torch.nn.functional as F
from loguru import logger


def _empty_similarity_metrics(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}SimilarityMean": float("nan"),
        f"{prefix}SimilarityP95": float("nan"),
        f"{prefix}SimilarityMax": float("nan"),
        f"{prefix}PositiveGapMean": float("nan"),
        f"{prefix}PositiveGapP05": float("nan"),
        f"{prefix}PositiveGapMin": float("nan"),
    }


def _empty_scalar_distribution_metrics(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}Mean": float("nan"),
        f"{prefix}Std": float("nan"),
        f"{prefix}Min": float("nan"),
        f"{prefix}P05": float("nan"),
        f"{prefix}P50": float("nan"),
        f"{prefix}P95": float("nan"),
        f"{prefix}Max": float("nan"),
    }


@torch.no_grad()
def _scalar_distribution_metrics(prefix: str, values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().reshape(-1).cpu()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return _empty_scalar_distribution_metrics(prefix)

    quantiles = torch.quantile(values, torch.tensor([0.05, 0.50, 0.95], dtype=values.dtype))
    return {
        f"{prefix}Mean": float(values.mean().item()),
        f"{prefix}Std": float(values.std(unbiased=False).item()),
        f"{prefix}Min": float(values.min().item()),
        f"{prefix}P05": float(quantiles[0].item()),
        f"{prefix}P50": float(quantiles[1].item()),
        f"{prefix}P95": float(quantiles[2].item()),
        f"{prefix}Max": float(values.max().item()),
    }


@torch.no_grad()
def _masked_similarity_metrics(
    prefix: str,
    scores: torch.Tensor,
    mask: torch.Tensor,
    *,
    positive_scores: torch.Tensor | None = None,
) -> dict[str, float]:
    if not bool(mask.any().item()):
        return _empty_similarity_metrics(prefix)

    selected = scores.detach().float().cpu()[mask.detach().cpu()]
    metrics = {
        f"{prefix}SimilarityMean": float(selected.mean().item()),
        f"{prefix}SimilarityP95": float(torch.quantile(selected, 0.95).item()),
        f"{prefix}SimilarityMax": float(selected.max().item()),
    }
    if positive_scores is None:
        metrics.update(
            {
                f"{prefix}PositiveGapMean": float("nan"),
                f"{prefix}PositiveGapP05": float("nan"),
                f"{prefix}PositiveGapMin": float("nan"),
            }
        )
        return metrics

    gaps = (positive_scores.detach().float() - scores.detach().float()).cpu()[mask.detach().cpu()]
    metrics.update(
        {
            f"{prefix}PositiveGapMean": float(gaps.mean().item()),
            f"{prefix}PositiveGapP05": float(torch.quantile(gaps, 0.05).item()),
            f"{prefix}PositiveGapMin": float(gaps.min().item()),
        }
    )
    return metrics


class ContrastiveMemoryBank:
    """FIFO queue of detached document embeddings for InfoNCE negatives."""

    def __init__(
        self,
        size: int = 0,
        *,
        warmup_steps: int = 0,
        mining_mode: str = "all",
        hard_negatives: int = 0,
        random_negatives: int = 0,
        hard_warmup_steps: int = 0,
        hard_ramp_steps: int = 1,
        too_hard_margin: float | None = None,
        hard_similarity_cap: float | None = None,
        soft_mode: str = "hard",
        adaptive_hard: bool = False,
        adaptive_hard_mode: str = "hard",
        hard_collision_threshold: float = 0.0,
        hard_collision_beta: float = 0.05,
    ):
        self.size = max(int(size), 0)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.mining_mode = str(mining_mode).strip().lower()
        if self.mining_mode not in {"all", "random", "hard", "mixed"}:
            raise ValueError("memory_bank_mining_mode must be one of: all, random, hard, mixed")
        self.hard_negatives = max(int(hard_negatives), 0)
        self.random_negatives = max(int(random_negatives), 0)
        self.hard_warmup_steps = max(int(hard_warmup_steps), 0)
        self.hard_ramp_steps = max(int(hard_ramp_steps), 1)
        self.too_hard_margin = too_hard_margin
        self.hard_similarity_cap = hard_similarity_cap
        self.soft_mode = str(soft_mode).strip().lower()
        if self.soft_mode not in {"hard", "soft-const", "soft"}:
            raise ValueError("memory_bank soft_mode must be one of: hard, soft-const, soft")
        self.adaptive_hard = bool(adaptive_hard)
        self.adaptive_hard_mode = str(adaptive_hard_mode).strip().lower()
        if self.adaptive_hard_mode not in {"hard", "soft"}:
            raise ValueError("memory_bank adaptive_hard_mode must be one of: hard, soft")
        self.hard_collision_threshold = float(hard_collision_threshold)
        self.hard_collision_beta = float(hard_collision_beta)
        if self.hard_collision_beta <= 0.0:
            raise ValueError(f"memory_bank hard_collision_beta must be > 0, got {self.hard_collision_beta}")
        self.embeddings: torch.Tensor | None = None
        self.doc_key_ids: torch.Tensor | None = None
        self.content_key_ids: torch.Tensor | None = None
        self.query_key_ids: torch.Tensor | None = None

    @property
    def enabled(self) -> bool:
        return self.size > 0

    @property
    def current_size(self) -> int:
        return 0 if self.embeddings is None else int(self.embeddings.shape[0])

    def get(
        self,
        batch: dict[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
        query_vectors: torch.Tensor | None = None,
        positive_vectors: torch.Tensor | None = None,
        step: int = 0,
        return_log_weights: bool = False,
    ):
        metrics = {
            "MemoryBankCapacity": float(self.size),
            "MemoryBankSize": float(self.current_size),
            "MemoryBankWarmupSteps": float(self.warmup_steps),
            "MemoryBankMiningMode": self.mining_mode,
            "MemoryBankHardNegatives": float(self.hard_negatives),
            "MemoryBankRandomNegatives": float(self.random_negatives),
            "MemoryBankHardWarmupSteps": float(self.hard_warmup_steps),
            "MemoryBankHardRampSteps": float(self.hard_ramp_steps),
            "MemoryBankTooHardMargin": float(self.too_hard_margin) if self.too_hard_margin is not None else -1.0,
            "MemoryBankHardSimilarityCap": (
                float(self.hard_similarity_cap) if self.hard_similarity_cap is not None else -1.0
            ),
            "MemoryBankSoftMode": self.soft_mode,
            "MemoryBankAdaptiveHardEnabled": float(int(self.adaptive_hard)),
            "MemoryBankAdaptiveHardMode": self.adaptive_hard_mode if self.adaptive_hard else "off",
            "MemoryBankHardCollisionThreshold": float(self.hard_collision_threshold),
            "MemoryBankHardCollisionBeta": float(self.hard_collision_beta),
            "MemoryBankAdaptiveHardAllowedRows": 0.0,
            "MemoryBankAdaptiveHardSuppressedRows": 0.0,
            "MemoryBankAdaptiveHardAllowedMean": 0.0,
            "MemoryBankAdaptiveHardSuppressedMean": 0.0,
        }
        metrics.update(_empty_similarity_metrics("MemoryBankValid"))
        metrics.update(_empty_similarity_metrics("MemoryBankActive"))
        metrics.update(_empty_similarity_metrics("MemoryBankActiveHard"))
        metrics.update(_empty_scalar_distribution_metrics("MemoryBankCollisionG"))
        metrics.update(_empty_scalar_distribution_metrics("MemoryBankCollisionBankMaxSimilarity"))
        metrics.update(_empty_scalar_distribution_metrics("MemoryBankAdaptiveHardWeight"))
        metrics["MemoryBankPositiveSimilarityMean"] = float("nan")
        if not self.enabled or self.embeddings is None or self.current_size == 0 or int(step) < self.warmup_steps:
            metrics["MemoryBankValidNegativesMean"] = 0.0
            metrics["MemoryBankValidNegativesMin"] = 0.0
            metrics["MemoryBankHardCandidateNegativesMean"] = 0.0
            metrics["MemoryBankHardCandidateNegativesMin"] = 0.0
            metrics["MemoryBankActiveNegativesMean"] = 0.0
            metrics["MemoryBankActiveHardNegativesMean"] = 0.0
            metrics["MemoryBankActiveRandomNegativesMean"] = 0.0
            metrics["MemoryBankActiveHardK"] = 0.0
            metrics["MemoryBankActiveRandomK"] = 0.0
            if return_log_weights:
                return None, None, None, metrics
            return None, None, metrics

        embeddings = self.embeddings.to(device=device, dtype=dtype)
        batch_size = int(batch["doc_key_id"].shape[0]) if "doc_key_id" in batch else int(batch["input_ids"].shape[0])
        valid = torch.ones(batch_size, embeddings.shape[0], dtype=torch.bool, device=device)

        for key, bank_values in (
            ("doc_key_id", self.doc_key_ids),
            ("content_key_id", self.content_key_ids),
            ("query_key_id", self.query_key_ids),
        ):
            current_values = batch.get(key)
            if bank_values is None:
                continue
            if current_values is None:
                logger.warning(
                    "memory_bank: batch missing {} while bank tracks it; "
                    "refusing bank negatives for safety (false-negative avoidance)",
                    key,
                )
                valid = torch.zeros_like(valid)
                break
            current_values = current_values.to(device=device).view(-1, 1)
            bank_values = bank_values.to(device=device).view(1, -1)
            valid &= current_values != bank_values
        pre_margin_valid = valid.clone()

        bank_sim = None
        pos_sim = None
        if query_vectors is not None:
            q_norm = F.normalize(query_vectors.detach(), p=2, dim=-1, eps=1e-8)
            emb_norm = F.normalize(embeddings, p=2, dim=-1, eps=1e-8)
            bank_sim = q_norm @ emb_norm.T
            if positive_vectors is not None:
                pos_sim = (q_norm * F.normalize(positive_vectors.detach(), p=2, dim=-1, eps=1e-8)).sum(
                    dim=1,
                    keepdim=True,
                )
            if pos_sim is not None and self.too_hard_margin is not None and self.soft_mode == "hard":
                valid &= bank_sim <= (pos_sim - float(self.too_hard_margin))

        hard_k = self._scheduled_hard_k(int(step))
        random_k = self.random_negatives
        hard_candidate = valid
        collision_g = None
        adaptive_hard_weight = None
        adaptive_row_has_candidate = None
        adaptive_hard_allowed = None
        if bank_sim is not None and self.hard_similarity_cap is not None:
            hard_candidate = valid & (bank_sim <= float(self.hard_similarity_cap))
        if self.adaptive_hard and bank_sim is not None and pos_sim is not None:
            row_has_candidate = pre_margin_valid.any(dim=1)
            bank_max_sim = bank_sim.masked_fill(~pre_margin_valid, float("-inf")).max(dim=1).values
            collision_g = bank_max_sim - pos_sim.view(-1)
            hard_allowed = row_has_candidate & (collision_g <= self.hard_collision_threshold)
            adaptive_row_has_candidate = row_has_candidate
            adaptive_hard_allowed = hard_allowed
            if self.adaptive_hard_mode == "hard":
                hard_candidate = hard_candidate & hard_allowed.view(-1, 1)
            else:
                raw_weight = torch.sigmoid(
                    (float(self.hard_collision_threshold) - collision_g) / float(self.hard_collision_beta)
                )
                adaptive_hard_weight = torch.where(row_has_candidate, raw_weight, torch.ones_like(raw_weight))
        hard_active = torch.zeros_like(valid)
        random_active = torch.zeros_like(valid)
        if self.mining_mode == "all":
            active = valid
            hard_k = 0
            random_k = 0
        else:
            active = torch.zeros_like(valid)
            if self.mining_mode in {"hard", "mixed"} and hard_k > 0 and bank_sim is not None:
                hard_active = self._topk_mask(bank_sim, hard_candidate, hard_k)
                active |= hard_active
            if self.mining_mode in {"random", "mixed"} and random_k > 0:
                random_active = self._random_mask(valid, random_k)
                active |= random_active
            active &= valid
        memory_log_weights = None
        if self.adaptive_hard and self.adaptive_hard_mode == "soft" and adaptive_hard_weight is not None:
            memory_log_weights = torch.zeros_like(bank_sim)
            hard_log_weight = torch.log(adaptive_hard_weight.clamp_min(1e-8)).view(-1, 1)
            memory_log_weights = torch.where(hard_active, hard_log_weight.expand_as(memory_log_weights), memory_log_weights)

        with torch.no_grad():
            valid_counts = valid.float().sum(dim=1)
            metrics["MemoryBankValidNegativesMean"] = valid_counts.mean().detach()
            metrics["MemoryBankValidNegativesMin"] = valid_counts.min().detach()
            hard_candidate_counts = hard_candidate.float().sum(dim=1)
            metrics["MemoryBankHardCandidateNegativesMean"] = hard_candidate_counts.mean().detach()
            metrics["MemoryBankHardCandidateNegativesMin"] = hard_candidate_counts.min().detach()
            active_counts = active.float().sum(dim=1)
            metrics["MemoryBankActiveNegativesMean"] = active_counts.mean().detach()
            metrics["MemoryBankActiveNegativesMin"] = active_counts.min().detach()
            metrics["MemoryBankActiveHardNegativesMean"] = hard_active.float().sum(dim=1).mean().detach()
            metrics["MemoryBankActiveRandomNegativesMean"] = random_active.float().sum(dim=1).mean().detach()
            metrics["MemoryBankActiveHardK"] = float(hard_k)
            metrics["MemoryBankActiveRandomK"] = float(random_k)
            if (
                self.adaptive_hard
                and bank_sim is not None
                and pos_sim is not None
                and adaptive_row_has_candidate is not None
                and adaptive_hard_allowed is not None
            ):
                row_has_candidate = adaptive_row_has_candidate
                if bool(row_has_candidate.any().item()):
                    hard_allowed = adaptive_hard_allowed
                    hard_suppressed = row_has_candidate & ~hard_allowed
                    metrics["MemoryBankAdaptiveHardAllowedRows"] = float(hard_allowed.float().sum().item())
                    metrics["MemoryBankAdaptiveHardSuppressedRows"] = float(hard_suppressed.float().sum().item())
                    metrics["MemoryBankAdaptiveHardAllowedMean"] = float(
                        hard_allowed[row_has_candidate].float().mean().item()
                    )
                    metrics["MemoryBankAdaptiveHardSuppressedMean"] = float(
                        hard_suppressed[row_has_candidate].float().mean().item()
                    )
                    if adaptive_hard_weight is not None:
                        metrics.update(
                            _scalar_distribution_metrics(
                                "MemoryBankAdaptiveHardWeight",
                                adaptive_hard_weight[row_has_candidate],
                            )
                        )
            if pos_sim is not None:
                metrics["MemoryBankPositiveSimilarityMean"] = float(pos_sim.detach().float().mean().item())
            if bank_sim is not None:
                if pos_sim is not None and bool(pre_margin_valid.any().item()):
                    bank_max_sim = bank_sim.masked_fill(~pre_margin_valid, float("-inf")).max(dim=1).values
                    row_has_candidate = pre_margin_valid.any(dim=1)
                    metrics.update(
                        _scalar_distribution_metrics(
                            "MemoryBankCollisionBankMaxSimilarity",
                            bank_max_sim[row_has_candidate],
                        )
                    )
                    metrics.update(
                        _scalar_distribution_metrics(
                            "MemoryBankCollisionG",
                            bank_max_sim[row_has_candidate] - pos_sim.view(-1)[row_has_candidate],
                        )
                    )
                expanded_pos = pos_sim.expand_as(bank_sim) if pos_sim is not None else None
                metrics.update(
                    _masked_similarity_metrics(
                        "MemoryBankValid",
                        bank_sim,
                        valid,
                        positive_scores=expanded_pos,
                    )
                )
                metrics.update(
                    _masked_similarity_metrics(
                        "MemoryBankActive",
                        bank_sim,
                        active,
                        positive_scores=expanded_pos,
                    )
                )
                metrics.update(
                    _masked_similarity_metrics(
                        "MemoryBankActiveHard",
                        bank_sim,
                        hard_active,
                        positive_scores=expanded_pos,
                    )
                )
        if return_log_weights:
            return embeddings, active, memory_log_weights, metrics
        return embeddings, active, metrics

    def enqueue(self, vectors: torch.Tensor, batch: dict[str, torch.Tensor]) -> None:
        if not self.enabled or vectors.numel() == 0:
            return
        vectors = F.normalize(vectors.detach().clone(), p=2, dim=-1, eps=1e-8)
        if vectors.dim() != 2:
            raise ValueError(f"memory bank vectors must be 2D, got shape={tuple(vectors.shape)}")
        self.embeddings = self._append(self.embeddings, vectors)
        self.doc_key_ids = self._append_ids(self.doc_key_ids, batch.get("doc_key_id"), device=vectors.device)
        self.content_key_ids = self._append_ids(self.content_key_ids, batch.get("content_key_id"), device=vectors.device)
        self.query_key_ids = self._append_ids(self.query_key_ids, batch.get("query_key_id"), device=vectors.device)

    def _append(self, previous: torch.Tensor | None, current: torch.Tensor) -> torch.Tensor:
        if previous is None:
            merged = current
        else:
            merged = torch.cat([previous.to(device=current.device, dtype=current.dtype), current], dim=0)
        return merged[-self.size :].detach()

    def _append_ids(
        self,
        previous: torch.Tensor | None,
        current: torch.Tensor | None,
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        if current is None:
            return None
        current = current.detach().clone().to(device=device).view(-1)
        if previous is None:
            merged = current
        else:
            merged = torch.cat([previous.to(device=device), current], dim=0)
        return merged[-self.size :].detach()

    def _scheduled_hard_k(self, step: int) -> int:
        if self.hard_negatives <= 0 or step < self.hard_warmup_steps:
            return 0
        progress = min(max((step - self.hard_warmup_steps) / float(self.hard_ramp_steps), 0.0), 1.0)
        return int(round(float(self.hard_negatives) * progress))

    @staticmethod
    def _topk_mask(scores: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
        k = max(0, min(int(k), int(scores.shape[1])))
        if k == 0:
            return torch.zeros_like(valid)
        masked_scores = scores.masked_fill(~valid, float("-inf"))
        values, indices = masked_scores.topk(k, dim=1)
        active = torch.zeros_like(valid)
        active.scatter_(1, indices, torch.isfinite(values))
        return active & valid

    @staticmethod
    def _random_mask(valid: torch.Tensor, k: int) -> torch.Tensor:
        k = max(0, min(int(k), int(valid.shape[1])))
        if k == 0:
            return torch.zeros_like(valid)
        random_scores = torch.rand(valid.shape, device=valid.device).masked_fill(~valid, float("-inf"))
        values, indices = random_scores.topk(k, dim=1)
        active = torch.zeros_like(valid)
        active.scatter_(1, indices, torch.isfinite(values))
        return active & valid


_ContrastiveMemoryBank = ContrastiveMemoryBank

__all__ = ["ContrastiveMemoryBank", "_ContrastiveMemoryBank"]
