from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from justatom.training.config import AnchorBankConfig


@dataclass(frozen=True)
class AnchorSelection:
    """Frozen-base query/document anchors visible to the current batch."""

    queries: torch.Tensor
    documents: torch.Tensor
    valid_mask: torch.Tensor
    metrics: dict[str, float | torch.Tensor]


class GeometryAnchorBank:
    """FIFO bank that preserves neighbourhoods of the unadapted encoder.

    The bank contains normalized embeddings produced with the LoRA adapter
    disabled. It never stores trainable or stale student embeddings.
    """

    def __init__(self, config: AnchorBankConfig):
        if not isinstance(config, AnchorBankConfig):
            raise TypeError("GeometryAnchorBank requires AnchorBankConfig")
        self.config = config
        self.queries: torch.Tensor | None = None
        self.documents: torch.Tensor | None = None
        self.doc_key_ids: torch.Tensor | None = None
        self.content_key_ids: torch.Tensor | None = None
        self.query_key_ids: torch.Tensor | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.config.size > 0

    @property
    def current_size(self) -> int:
        return 0 if self.documents is None else int(self.documents.shape[0])

    @torch.no_grad()
    def select(
        self,
        batch: dict[str, Any],
        *,
        step: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> AnchorSelection | None:
        if (
            not self.enabled
            or self.queries is None
            or self.documents is None
            or self.current_size == 0
            or int(step) < self.config.warmup_steps
        ):
            return None

        queries = self.queries.to(device=device, dtype=dtype)
        documents = self.documents.to(device=device, dtype=dtype)
        batch_size = self._batch_size(batch)
        valid = torch.ones((batch_size, self.current_size), dtype=torch.bool, device=device)
        for name, bank_values in (
            ("doc_key_id", self.doc_key_ids),
            ("content_key_id", self.content_key_ids),
            ("query_key_id", self.query_key_ids),
        ):
            current_values = batch.get(name)
            if not isinstance(current_values, torch.Tensor) or bank_values is None:
                continue
            valid &= current_values.to(device).view(-1, 1) != bank_values.to(device).view(1, -1)

        valid_per_row = valid.float().sum(dim=-1)
        metrics: dict[str, float | torch.Tensor] = {
            "anchor/capacity": float(self.config.size),
            "anchor/size": float(self.current_size),
            "anchor/valid_mean": valid_per_row.mean(),
            "anchor/valid_min": valid_per_row.min(),
        }
        return AnchorSelection(
            queries=queries,
            documents=documents,
            valid_mask=valid,
            metrics=metrics,
        )

    def geometry_loss(
        self,
        *,
        student_queries: torch.Tensor,
        student_documents: torch.Tensor,
        base_queries: torch.Tensor,
        base_documents: torch.Tensor,
        selection: AnchorSelection,
    ) -> tuple[torch.Tensor | None, dict[str, float | torch.Tensor]]:
        """Preserve frozen-base rankings in both retrieval directions."""

        if student_queries.shape != base_queries.shape:
            raise ValueError("student/base query embeddings must have matching shapes")
        if student_documents.shape != base_documents.shape:
            raise ValueError("student/base document embeddings must have matching shapes")
        if student_queries.shape != student_documents.shape:
            raise ValueError("query/document embeddings must have matching shapes")
        if selection.valid_mask.shape != (student_queries.shape[0], selection.documents.shape[0]):
            raise ValueError("anchor valid mask has incompatible shape")

        with torch.autocast(device_type=student_queries.device.type, enabled=False):
            student_queries_float = F.normalize(student_queries.float(), p=2, dim=-1, eps=1e-8)
            student_documents_float = F.normalize(student_documents.float(), p=2, dim=-1, eps=1e-8)
            base_queries_float = F.normalize(base_queries.detach().float(), p=2, dim=-1, eps=1e-8)
            base_documents_float = F.normalize(base_documents.detach().float(), p=2, dim=-1, eps=1e-8)
            anchor_queries = F.normalize(selection.queries.detach().float(), p=2, dim=-1, eps=1e-8)
            anchor_documents = F.normalize(selection.documents.detach().float(), p=2, dim=-1, eps=1e-8)
            valid = selection.valid_mask.to(device=student_queries.device, dtype=torch.bool)

            forward, forward_entropy, forward_rows = self._directional_kl(
                student_queries_float @ anchor_documents.T,
                base_queries_float @ anchor_documents.T,
                valid,
            )
            reverse, reverse_entropy, reverse_rows = self._directional_kl(
                student_documents_float @ anchor_queries.T,
                base_documents_float @ anchor_queries.T,
                valid,
            )
            active_terms = [loss for loss in (forward, reverse) if loss is not None]
            if not active_terms:
                return None, {
                    "loss/anchor_geometry": 0.0,
                    "anchor/active_rows": 0.0,
                }
            loss = torch.stack(active_terms).mean()

        metrics: dict[str, float | torch.Tensor] = {
            "loss/anchor_geometry": loss.detach(),
            "anchor/forward_kl": 0.0 if forward is None else forward.detach(),
            "anchor/reverse_kl": 0.0 if reverse is None else reverse.detach(),
            "anchor/teacher_entropy_forward": forward_entropy,
            "anchor/teacher_entropy_reverse": reverse_entropy,
            "anchor/active_rows": float(max(forward_rows, reverse_rows)),
        }
        return loss, metrics

    def _directional_kl(
        self,
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor, int]:
        active_rows = valid.any(dim=-1)
        row_count = int(active_rows.sum().item())
        if row_count == 0:
            return None, student_scores.new_zeros(()), 0

        floor = torch.finfo(student_scores.dtype).min
        temperature = float(self.config.temperature)
        student_logits = (student_scores / temperature).masked_fill(~valid, floor)
        teacher_logits = (teacher_scores / temperature).masked_fill(~valid, floor)
        teacher_log_probabilities = F.log_softmax(teacher_logits[active_rows], dim=-1)
        teacher_probabilities = teacher_log_probabilities.exp()
        student_log_probabilities = F.log_softmax(student_logits[active_rows], dim=-1)
        per_row = (teacher_probabilities * (teacher_log_probabilities - student_log_probabilities)).sum(dim=-1)
        entropy = -(teacher_probabilities * teacher_log_probabilities).sum(dim=-1).mean().detach()
        return per_row.mean(), entropy, row_count

    @torch.no_grad()
    def enqueue(
        self,
        base_queries: torch.Tensor,
        base_documents: torch.Tensor,
        batch: dict[str, Any],
    ) -> None:
        if not self.enabled or base_queries.numel() == 0:
            return
        if base_queries.ndim != 2 or base_documents.shape != base_queries.shape:
            raise ValueError("base query/document embeddings must have matching [batch, dim] shapes")
        base_queries = F.normalize(base_queries.detach().clone(), p=2, dim=-1, eps=1e-8)
        base_documents = F.normalize(base_documents.detach().clone(), p=2, dim=-1, eps=1e-8)
        self.queries = self._append(self.queries, base_queries)
        self.documents = self._append(self.documents, base_documents)
        self.doc_key_ids = self._append_ids(self.doc_key_ids, batch.get("doc_key_id"), base_queries.device)
        self.content_key_ids = self._append_ids(
            self.content_key_ids,
            batch.get("content_key_id"),
            base_queries.device,
        )
        self.query_key_ids = self._append_ids(
            self.query_key_ids,
            batch.get("query_key_id"),
            base_queries.device,
        )

    def _append(self, previous: torch.Tensor | None, current: torch.Tensor) -> torch.Tensor:
        merged = current if previous is None else torch.cat([previous.to(current), current], dim=0)
        return merged[-self.config.size :].detach()

    def _append_ids(
        self,
        previous: torch.Tensor | None,
        current: Any,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not isinstance(current, torch.Tensor):
            return None
        current = current.detach().clone().to(device).view(-1)
        merged = current if previous is None else torch.cat([previous.to(device), current], dim=0)
        return merged[-self.config.size :].detach()

    @staticmethod
    def _batch_size(batch: dict[str, Any]) -> int:
        for name in ("doc_key_id", "query_key_id", "input_ids", "pos_input_ids"):
            value = batch.get(name)
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
        raise ValueError("cannot infer batch size for anchor-bank selection")


__all__ = ["AnchorSelection", "GeometryAnchorBank"]
