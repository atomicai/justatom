from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

GradientList = Sequence[torch.Tensor | None]


@dataclass(frozen=True)
class ProjectionStats:
    primary_norm: float
    memory_norm: float
    projected_memory_norm: float
    dot: float
    cosine: float
    conflict: bool
    coefficient: float

    def metrics(self) -> dict[str, float]:
        return {
            "gradient/primary_norm": self.primary_norm,
            "gradient/memory_norm": self.memory_norm,
            "gradient/projected_memory_norm": self.projected_memory_norm,
            "gradient/dot": self.dot,
            "gradient/cosine": self.cosine,
            "gradient/conflict": float(self.conflict),
            "gradient/projection_coefficient": self.coefficient,
        }


@dataclass(frozen=True)
class ConstraintProjectionStats:
    constraint_norm: float
    update_norm: float
    projected_update_norm: float
    dot: float
    cosine: float
    active: bool
    coefficient: float

    def metrics(self) -> dict[str, float]:
        return {
            "gradient_anchor/constraint_norm": self.constraint_norm,
            "gradient_anchor/update_norm": self.update_norm,
            "gradient_anchor/projected_update_norm": self.projected_update_norm,
            "gradient_anchor/dot": self.dot,
            "gradient_anchor/cosine": self.cosine,
            "gradient_anchor/active": float(self.active),
            "gradient_anchor/projection_coefficient": self.coefficient,
        }


def project_conflicting_gradients(
    primary: GradientList,
    memory: GradientList,
    *,
    eps: float = 1e-12,
) -> tuple[list[torch.Tensor | None], ProjectionStats]:
    """Project a conflicting memory gradient off the protected primary gradient.

    The operation is deliberately asymmetric. ``primary`` is never changed.
    When their shared-parameter dot product is negative, the component of the
    memory gradient that opposes the primary gradient is removed. Gradients for
    memory-only parameters are retained unchanged.
    """
    if len(primary) != len(memory):
        raise ValueError("primary and memory gradient lists must have equal length")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    reference = next(
        (gradient for gradient in (*primary, *memory) if gradient is not None),
        None,
    )
    if reference is None:
        zero_stats = ProjectionStats(0.0, 0.0, 0.0, 0.0, 0.0, False, 0.0)
        return [None for _ in memory], zero_stats

    device = reference.device
    dot = torch.zeros((), device=device, dtype=torch.float32)
    primary_sq = torch.zeros_like(dot)
    memory_sq = torch.zeros_like(dot)
    for primary_gradient, memory_gradient in zip(primary, memory):
        if primary_gradient is not None:
            primary_float = primary_gradient.detach().float()
            primary_sq = primary_sq + primary_float.square().sum()
        if memory_gradient is not None:
            memory_float = memory_gradient.detach().float()
            memory_sq = memory_sq + memory_float.square().sum()
        if primary_gradient is not None and memory_gradient is not None:
            dot = dot + (primary_gradient.detach().float() * memory_gradient.detach().float()).sum()

    conflict = bool(dot.item() < 0.0 and primary_sq.item() > eps)
    coefficient = dot / primary_sq.clamp_min(eps) if conflict else torch.zeros_like(dot)
    projected: list[torch.Tensor | None] = []
    projected_sq = torch.zeros_like(dot)
    for primary_gradient, memory_gradient in zip(primary, memory):
        if memory_gradient is None and (not conflict or primary_gradient is None):
            projected.append(None)
            continue
        if memory_gradient is None:
            assert primary_gradient is not None
            value = -coefficient.to(dtype=primary_gradient.dtype) * primary_gradient.detach()
        else:
            value = memory_gradient.detach().clone()
            if conflict and primary_gradient is not None:
                value = value - coefficient.to(dtype=value.dtype) * primary_gradient.detach()
        projected.append(value)
        projected_sq = projected_sq + value.float().square().sum()

    primary_norm = primary_sq.sqrt()
    memory_norm = memory_sq.sqrt()
    cosine_denominator = (primary_norm * memory_norm).clamp_min(eps)
    cosine = dot / cosine_denominator if primary_sq.item() > eps and memory_sq.item() > eps else torch.zeros_like(dot)
    stats = ProjectionStats(
        primary_norm=float(primary_norm.item()),
        memory_norm=float(memory_norm.item()),
        projected_memory_norm=float(projected_sq.sqrt().item()),
        dot=float(dot.item()),
        cosine=float(cosine.item()),
        conflict=conflict,
        coefficient=float(coefficient.item()),
    )
    return projected, stats


def project_update_against_constraint(
    update: GradientList,
    constraint: GradientList,
    *,
    eps: float = 1e-12,
) -> tuple[list[torch.Tensor | None], ConstraintProjectionStats]:
    """Remove only an update component that would increase a constraint.

    Parameters follow ``theta <- theta - eta * update``. A negative dot
    product with the constraint gradient therefore increases the constraint to
    first order. The constraint itself is never added as an auxiliary loss.
    """

    projected, raw = project_conflicting_gradients(constraint, update, eps=eps)
    return projected, ConstraintProjectionStats(
        constraint_norm=raw.primary_norm,
        update_norm=raw.memory_norm,
        projected_update_norm=raw.projected_memory_norm,
        dot=raw.dot,
        cosine=raw.cosine,
        active=raw.conflict,
        coefficient=raw.coefficient,
    )
