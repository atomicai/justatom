from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from justatom.training.config import AuxiliaryGradientMode

GradientList = Sequence[torch.Tensor | None]


@dataclass(frozen=True)
class AuxiliaryGradientStats:
    retrieval_norm: float
    auxiliary_norm: float
    auxiliary_controlled_norm: float
    dot: float
    cosine: float
    compatible: bool
    cosine_scale: float
    norm_scale: float
    total_scale: float

    @property
    def auxiliary_dot(self) -> float:
        return self.dot

    @property
    def auxiliary_cosine(self) -> float:
        return self.cosine

    @property
    def auxiliary_compatible(self) -> bool:
        return self.compatible

    def metrics(self) -> dict[str, float]:
        return {
            "gradient/retrieval_norm": self.retrieval_norm,
            "gradient/auxiliary_norm": self.auxiliary_norm,
            "gradient/auxiliary_controlled_norm": self.auxiliary_controlled_norm,
            "gradient/auxiliary_dot": self.dot,
            "gradient/auxiliary_cosine": self.cosine,
            "gradient/auxiliary_compatible": float(self.compatible),
            "gradient/auxiliary_cosine_scale": self.cosine_scale,
            "gradient/auxiliary_norm_scale": self.norm_scale,
            "gradient/auxiliary_total_scale": self.total_scale,
        }


def control_auxiliary_gradients(
    primary: GradientList,
    auxiliary: GradientList,
    *,
    mode: AuxiliaryGradientMode,
    max_norm_ratio: float,
    eps: float,
) -> tuple[list[torch.Tensor | None], AuxiliaryGradientStats]:
    """Control an auxiliary parameter-gradient list against a primary list."""
    if len(primary) != len(auxiliary):
        raise ValueError("primary and auxiliary gradient lists must have equal length")
    if not math.isfinite(float(max_norm_ratio)) or max_norm_ratio < 0.0:
        raise ValueError("max_norm_ratio must be finite and non-negative")
    if not math.isfinite(float(eps)) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    if not isinstance(mode, AuxiliaryGradientMode):
        raise ValueError("mode must be an AuxiliaryGradientMode")

    reference = next(
        (gradient for gradient in (*primary, *auxiliary) if gradient is not None),
        None,
    )
    if reference is None:
        passthrough = mode in (AuxiliaryGradientMode.OFF, AuxiliaryGradientMode.OBSERVE)
        zero_stats = AuxiliaryGradientStats(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            False,
            float(passthrough),
            float(passthrough),
            float(passthrough),
        )
        return [None for _ in auxiliary], zero_stats

    zero = torch.zeros((), device=reference.device, dtype=torch.float32)
    dot = zero.clone()
    primary_sq = zero.clone()
    auxiliary_sq = zero.clone()
    for primary_gradient, auxiliary_gradient in zip(primary, auxiliary):
        if primary_gradient is not None:
            primary_float = primary_gradient.detach().float()
            primary_sq = primary_sq + primary_float.square().sum()
        if auxiliary_gradient is not None:
            auxiliary_float = auxiliary_gradient.detach().float()
            auxiliary_sq = auxiliary_sq + auxiliary_float.square().sum()
        if primary_gradient is not None and auxiliary_gradient is not None:
            dot = dot + (primary_gradient.detach().float() * auxiliary_gradient.detach().float()).sum()

    primary_norm = primary_sq.sqrt()
    auxiliary_norm = auxiliary_sq.sqrt()
    cosine = dot / (primary_norm * auxiliary_norm).clamp_min(eps)

    if not torch.isfinite(torch.stack((dot, primary_norm, auxiliary_norm, cosine))).all().item():
        raise ValueError("non-finite auxiliary gradient statistics")

    if primary_norm.item() <= eps or auxiliary_norm.item() <= eps:
        dot = zero.clone()
        cosine = zero.clone()

    if mode in (AuxiliaryGradientMode.OFF, AuxiliaryGradientMode.OBSERVE):
        cosine_scale = 1.0
        norm_scale = 1.0
    else:
        cosine_scale = float(cosine.clamp_min(0.0).item())
        candidate_norm = cosine_scale * float(auxiliary_norm.item())
        norm_scale = min(1.0, max_norm_ratio * float(primary_norm.item()) / (candidate_norm + eps))
        if dot.item() <= 0.0 or primary_norm.item() <= eps or auxiliary_norm.item() <= eps:
            cosine_scale = 0.0
            norm_scale = 0.0

    total_scale = cosine_scale * norm_scale
    controlled = [None if gradient is None else gradient.detach().clone() * total_scale for gradient in auxiliary]
    controlled_sq = zero.clone()
    for gradient in controlled:
        if gradient is not None:
            controlled_sq = controlled_sq + gradient.detach().float().square().sum()
    controlled_norm = controlled_sq.sqrt()

    if not torch.isfinite(torch.stack((controlled_norm,))).all().item():
        raise ValueError("non-finite auxiliary gradient statistics")

    stats = AuxiliaryGradientStats(
        retrieval_norm=float(primary_norm.item()),
        auxiliary_norm=float(auxiliary_norm.item()),
        auxiliary_controlled_norm=float(controlled_norm.item()),
        dot=float(dot.item()),
        cosine=float(cosine.item()),
        compatible=bool(dot.item() > 0.0),
        cosine_scale=cosine_scale,
        norm_scale=norm_scale,
        total_scale=total_scale,
    )
    return controlled, stats
