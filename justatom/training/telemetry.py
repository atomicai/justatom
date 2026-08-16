from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn


@torch.no_grad()
def scalar_distribution(prefix: str, values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().reshape(-1).cpu()
    values = values[torch.isfinite(values)]
    names = ("mean", "std", "min", "p05", "p50", "p95", "max")
    if values.numel() == 0:
        return {f"{prefix}/{name}": float("nan") for name in names}

    quantiles = torch.quantile(values, torch.tensor([0.05, 0.50, 0.95], dtype=values.dtype))
    return {
        f"{prefix}/mean": float(values.mean().item()),
        f"{prefix}/std": float(values.std(unbiased=False).item()),
        f"{prefix}/min": float(values.min().item()),
        f"{prefix}/p05": float(quantiles[0].item()),
        f"{prefix}/p50": float(quantiles[1].item()),
        f"{prefix}/p95": float(quantiles[2].item()),
        f"{prefix}/max": float(values.max().item()),
    }


@torch.no_grad()
def batch_retrieval_metrics(scores: torch.Tensor) -> dict[str, float]:
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"scores must be a square [batch, batch] matrix, got {tuple(scores.shape)}")
    batch_size = int(scores.shape[0])
    sorted_indices = torch.argsort(scores, dim=1, descending=True)
    targets = torch.arange(batch_size, device=scores.device).unsqueeze(1)
    ranks = (sorted_indices == targets).nonzero(as_tuple=False)[:, 1] + 1
    rank_float = ranks.float()
    return {
        "batch/size": float(batch_size),
        "batch/errors_at_1": float((ranks > 1).sum().item()),
        "batch/error_rate_at_1": float((ranks > 1).float().mean().item()),
        "batch/hit_rate_at_1": float((ranks <= 1).float().mean().item()),
        "batch/hit_rate_at_3": float((ranks <= min(3, batch_size)).float().mean().item()),
        "batch/hit_rate_at_5": float((ranks <= min(5, batch_size)).float().mean().item()),
        "batch/mrr": float((1.0 / rank_float).mean().item()),
        "batch/mean_rank": float(rank_float.mean().item()),
        "batch/median_rank": float(rank_float.median().item()),
    }


@torch.no_grad()
def retrieval_metrics_by_confidence(
    scores: torch.Tensor,
    confidence: torch.Tensor,
) -> dict[str, float]:
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"scores must be square, got {tuple(scores.shape)}")
    confidence = confidence.detach().reshape(-1).to(scores.device)
    if confidence.shape[0] != scores.shape[0]:
        raise ValueError("confidence batch dimension must match scores")
    targets = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1)
    ranks = (torch.argsort(scores, dim=1, descending=True) == targets).nonzero(as_tuple=False)[:, 1] + 1
    result: dict[str, float] = {}
    buckets = (
        ("low", 0.0, 1.0 / 3.0, False),
        ("medium", 1.0 / 3.0, 2.0 / 3.0, False),
        ("high", 2.0 / 3.0, 1.0, True),
    )
    for name, lower, upper, include_upper in buckets:
        mask = (confidence >= lower) & ((confidence <= upper) if include_upper else (confidence < upper))
        prefix = f"alpha_target_bucket/{name}"
        result[f"{prefix}/count"] = float(mask.sum().item())
        result[f"{prefix}/hit_rate_at_1"] = float((ranks[mask] <= 1).float().mean().item()) if mask.any() else float("nan")
        result[f"{prefix}/mrr"] = float((1.0 / ranks[mask].float()).mean().item()) if mask.any() else float("nan")
    return result


def grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    gradients = [parameter.grad.detach().float().norm(2) for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    return float(torch.stack(gradients).norm(2).item())


def resolve_metric_tensors(metrics: Mapping[str, Any]) -> dict[str, Any]:
    resolved = dict(metrics)
    tensor_keys = [key for key, value in resolved.items() if isinstance(value, torch.Tensor)]
    if not tensor_keys:
        return resolved
    for key in tensor_keys:
        if resolved[key].numel() != 1:
            raise ValueError(f"metric tensor {key!r} must be scalar")
    stacked = torch.stack([resolved[key].detach().reshape(()).float() for key in tensor_keys])
    for key, value in zip(tensor_keys, stacked.cpu().tolist(), strict=True):
        resolved[key] = float(value)
    return resolved
