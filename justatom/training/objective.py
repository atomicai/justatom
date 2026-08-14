from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from justatom.training.config import MarginConfig, ObjectiveConfig
from justatom.training.loss import ContrastiveLoss
from justatom.training.memory_bank import MemorySelection


@dataclass(frozen=True)
class ObjectiveInputs:
    queries: torch.Tensor
    positives: torch.Tensor
    query_alt: torch.Tensor | None = None
    alpha: torch.Tensor | None = None
    memory: MemorySelection | None = None
    margin: torch.Tensor | None = None
    raw_margin: torch.Tensor | None = None
    semantic_pair_scores: torch.Tensor | None = None
    lexical_pair_scores: torch.Tensor | None = None
    alpha_mix_weight: float = 0.0
    alpha_entropy_weight: float = 0.0


@dataclass(frozen=True)
class ObjectiveOutput:
    loss: torch.Tensor
    primary_loss: torch.Tensor
    memory_loss: torch.Tensor | None
    main_per_row: torch.Tensor
    memory_per_row: torch.Tensor | None
    simcse_per_row: torch.Tensor | None
    soft_fn_per_row: torch.Tensor | None
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
        if inputs.alpha is not None and inputs.query_alt is None and self.config.simcse_dropout_weight > 0.0:
            raise ValueError("alpha(q) requires query_alt when SimCSE auxiliary pressure is enabled")
        if (inputs.semantic_pair_scores is None) != (inputs.lexical_pair_scores is None):
            raise ValueError("semantic_pair_scores and lexical_pair_scores must be supplied together")

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
        if inputs.alpha is not None:
            alpha = inputs.alpha.view(-1)
            if alpha.shape != main.shape:
                raise ValueError(f"alpha must have shape {tuple(main.shape)}, got {tuple(alpha.shape)}")
            auxiliary_weight = 1.0 - alpha.detach()
            per_row = main + auxiliary_weight * auxiliary
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
            "loss/soft_fn": 0.0 if soft_fn is None else soft_fn.detach().mean(),
            "loss/lexical_mix": 0.0,
            "loss/alpha_entropy_bonus": 0.0,
            "loss/memory_margin_regularization": 0.0,
            "memory/active_negatives_mean": active_negatives,
            "temperature": self.kernel.tau.detach(),
        }

        if inputs.alpha is not None and inputs.semantic_pair_scores is not None and inputs.lexical_pair_scores is not None:
            alpha_column = inputs.alpha.view(-1, 1)
            mixed_pair = alpha_column * inputs.semantic_pair_scores + (1.0 - alpha_column) * inputs.lexical_pair_scores
            if mixed_pair.ndim != 2 or mixed_pair.shape[1] != 2:
                raise ValueError("pair scores must have shape [batch, 2] for positive and negative")
            positive_distance = 1.0 - mixed_pair[:, 0]
            negative_distance = 1.0 - mixed_pair[:, 1]
            positive_loss = 0.5 * positive_distance.pow(2)
            negative_loss = 0.5 * torch.relu(self.config.pairwise_margin - negative_distance).pow(2)
            mix_loss = (positive_loss + negative_loss).mean()
            loss = loss + inputs.alpha_mix_weight * mix_loss
            metrics["loss/lexical_mix"] = mix_loss.detach()

        if inputs.alpha is not None and inputs.alpha_entropy_weight > 0.0:
            alpha_safe = inputs.alpha.clamp(1e-6, 1.0 - 1e-6)
            entropy = -(alpha_safe * alpha_safe.log() + (1.0 - alpha_safe) * (1.0 - alpha_safe).log()).mean()
            entropy_bonus = inputs.alpha_entropy_weight * entropy
            loss = loss - entropy_bonus
            metrics["loss/alpha_entropy_bonus"] = entropy_bonus.detach()

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
            metrics=metrics,
        )
