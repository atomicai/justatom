from __future__ import annotations

import torch
import torch.nn.functional as F
from loguru import logger


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
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, float]]:
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
        }
        if not self.enabled or self.embeddings is None or self.current_size == 0 or int(step) < self.warmup_steps:
            metrics["MemoryBankValidNegativesMean"] = 0.0
            metrics["MemoryBankValidNegativesMin"] = 0.0
            metrics["MemoryBankActiveNegativesMean"] = 0.0
            metrics["MemoryBankActiveHardK"] = 0.0
            metrics["MemoryBankActiveRandomK"] = 0.0
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

        bank_sim = None
        if query_vectors is not None:
            q_norm = F.normalize(query_vectors.detach(), p=2, dim=-1, eps=1e-8)
            emb_norm = F.normalize(embeddings, p=2, dim=-1, eps=1e-8)
            bank_sim = q_norm @ emb_norm.T
            if positive_vectors is not None and self.too_hard_margin is not None:
                pos_sim = (q_norm * F.normalize(positive_vectors.detach(), p=2, dim=-1, eps=1e-8)).sum(
                    dim=1,
                    keepdim=True,
                )
                valid &= bank_sim <= (pos_sim - float(self.too_hard_margin))

        hard_k = self._scheduled_hard_k(int(step))
        random_k = self.random_negatives
        if self.mining_mode == "all":
            active = valid
            hard_k = 0
            random_k = 0
        else:
            active = torch.zeros_like(valid)
            if self.mining_mode in {"hard", "mixed"} and hard_k > 0 and bank_sim is not None:
                active |= self._topk_mask(bank_sim, valid, hard_k)
            if self.mining_mode in {"random", "mixed"} and random_k > 0:
                active |= self._random_mask(valid, random_k)
            active &= valid

        with torch.no_grad():
            valid_counts = valid.float().sum(dim=1)
            metrics["MemoryBankValidNegativesMean"] = valid_counts.mean().detach()
            metrics["MemoryBankValidNegativesMin"] = valid_counts.min().detach()
            active_counts = active.float().sum(dim=1)
            metrics["MemoryBankActiveNegativesMean"] = active_counts.mean().detach()
            metrics["MemoryBankActiveNegativesMin"] = active_counts.min().detach()
            metrics["MemoryBankActiveHardK"] = float(hard_k)
            metrics["MemoryBankActiveRandomK"] = float(random_k)
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
