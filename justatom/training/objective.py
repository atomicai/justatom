from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from justatom.training.config import MarginConfig, ObjectiveConfig
from justatom.training.loss import ContrastiveLoss
from justatom.training.memory_bank import MemorySelection


@dataclass(frozen=True)
class ObjectiveInputs:
    queries: torch.Tensor
    positives: torch.Tensor
    query_alt: torch.Tensor | None = None
    alpha_logits: torch.Tensor | None = None
    memory: MemorySelection | None = None
    margin: torch.Tensor | None = None
    raw_margin: torch.Tensor | None = None
    alpha_supervision_weight: float = 0.0


@dataclass(frozen=True)
class ObjectiveOutput:
    loss: torch.Tensor
    primary_loss: torch.Tensor
    memory_loss: torch.Tensor | None
    main_per_row: torch.Tensor
    memory_per_row: torch.Tensor | None
    simcse_per_row: torch.Tensor | None
    soft_fn_per_row: torch.Tensor | None
    alpha_target: torch.Tensor | None
    alpha_supervision_per_row: torch.Tensor | None
    metrics: dict[str, float | torch.Tensor]


class ContrastiveObjective(nn.Module):
    """Compose the three supported methods around one contrastive kernel."""

    def __init__(self, config: ObjectiveConfig):
        super().__init__()
        self.config = config
        self.kernel = ContrastiveLoss(
            temperature=config.temperature,
            reduction="none",
            learnable_temperature=config.learnable_temperature,
            decoupled=config.decoupled,
        )

    def forward(
        self,
        inputs: ObjectiveInputs,
        *,
        margin_config: MarginConfig | None = None,
    ) -> ObjectiveOutput:
        if inputs.alpha_logits is not None and inputs.query_alt is None and self.config.simcse_dropout_weight > 0.0:
            raise ValueError("alpha(q) requires query_alt when SimCSE auxiliary pressure is enabled")

        memory = inputs.memory
        main = self.kernel.info_nce(
            inputs.queries,
            inputs.positives,
            reduction="none",
        )
        memory_per_row = None
        has_memory = (
            memory is not None
            and memory.embeddings is not None
            and memory.active_mask is not None
            and bool(memory.active_mask.any().item())
        )
        if has_memory:
            assert memory is not None
            augmented = self.kernel.info_nce(
                inputs.queries,
                inputs.positives,
                reduction="none",
                memory_negatives=memory.embeddings,
                memory_negative_mask=memory.active_mask,
                memory_log_weights=memory.log_weights,
                memory_margin=inputs.margin,
                memory_soft_beta=None if margin_config is None else margin_config.admission_beta,
            )
            memory_per_row = augmented - main

        simcse = None
        soft_fn = None
        auxiliary = main.new_zeros(main.shape)
        if inputs.query_alt is not None and self.config.simcse_dropout_weight > 0.0:
            simcse = self.kernel.simcse_term(inputs.queries, inputs.query_alt, reduction="none")
            auxiliary = auxiliary + self.config.simcse_dropout_weight * simcse
        if self.config.soft_fn_attract_weight > 0.0 and inputs.queries.shape[0] > 1:
            soft_fn = self.kernel.soft_fn_term(
                inputs.queries,
                inputs.positives,
                topk=self.config.soft_fn_topk,
                reduction="none",
            )
            auxiliary = auxiliary + self.config.soft_fn_attract_weight * soft_fn

        per_row = main + auxiliary
        alpha_target = None
        alpha_supervision = None
        if inputs.alpha_logits is not None:
            alpha_logits = inputs.alpha_logits.view(-1)
            if alpha_logits.shape != main.shape:
                raise ValueError(f"alpha logits must have shape {tuple(main.shape)}, got {tuple(alpha_logits.shape)}")
            alpha = torch.sigmoid(alpha_logits)
            auxiliary_weight = 1.0 - alpha.detach()
            per_row = main + auxiliary_weight * auxiliary
            confidence_logits = inputs.queries.detach() @ inputs.positives.detach().T
            confidence_logits = confidence_logits / self.kernel.tau.detach()
            alpha_target = torch.softmax(confidence_logits, dim=-1).diagonal()
            alpha_supervision = F.binary_cross_entropy_with_logits(alpha_logits, alpha_target, reduction="none")
            if inputs.alpha_supervision_weight != 0.0:
                per_row = per_row + inputs.alpha_supervision_weight * alpha_supervision
        if memory_per_row is not None:
            per_row = per_row + memory_per_row
        loss = per_row.mean()
        memory_loss = None if memory_per_row is None else memory_per_row.mean()

        active_negatives = 0.0
        if memory is not None and memory.active_mask is not None:
            active_negatives = float(memory.active_mask.float().sum(dim=1).mean().item())
        metrics: dict[str, float | torch.Tensor] = {
            "loss/main": main.detach().mean(),
            "loss/memory": 0.0 if memory_per_row is None else memory_per_row.detach().mean(),
            "loss/alpha_aux": 0.0 if simcse is None else simcse.detach().mean(),
            "loss/alpha_supervision": 0.0 if alpha_supervision is None else alpha_supervision.detach().mean(),
            "loss/soft_fn": 0.0 if soft_fn is None else soft_fn.detach().mean(),
            "loss/memory_margin_regularization": 0.0,
            "memory/active_negatives_mean": active_negatives,
            "temperature": self.kernel.tau.detach(),
        }

        if margin_config is not None and inputs.raw_margin is not None:
            regularization = margin_config.regularization_weight * (inputs.raw_margin - margin_config.base).pow(2).mean()
            loss = loss + regularization
            memory_loss = regularization if memory_loss is None else memory_loss + regularization
            metrics["loss/memory_margin_regularization"] = regularization.detach()
            metrics["loss/memory_margin_regularization_tensor"] = regularization

        primary_loss = loss if memory_loss is None else loss - memory_loss
        return ObjectiveOutput(
            loss=loss,
            primary_loss=primary_loss,
            memory_loss=memory_loss,
            main_per_row=main,
            memory_per_row=memory_per_row,
            simcse_per_row=simcse,
            soft_fn_per_row=soft_fn,
            alpha_target=alpha_target,
            alpha_supervision_per_row=alpha_supervision,
            metrics=metrics,
        )
