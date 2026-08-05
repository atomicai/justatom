from __future__ import annotations

import torch

from justatom.training.telemetry import batch_retrieval_metrics, grad_norm, resolve_metric_tensors, scalar_distribution


def test_scalar_distribution_has_stable_quantiles():
    metrics = scalar_distribution("alpha", torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))

    assert metrics["alpha/mean"] == 0.5
    assert metrics["alpha/p50"] == 0.5
    assert metrics["alpha/min"] == 0.0
    assert metrics["alpha/max"] == 1.0


def test_scalar_distribution_ignores_nonfinite_values():
    metrics = scalar_distribution("margin", torch.tensor([0.0, float("nan"), 1.0]))

    assert metrics["margin/mean"] == 0.5


def test_batch_retrieval_metrics_uses_diagonal_as_positive():
    scores = torch.tensor([[2.0, 1.0], [0.0, 3.0]])

    metrics = batch_retrieval_metrics(scores)

    assert metrics["batch/hit_rate_at_1"] == 1.0
    assert metrics["batch/mrr"] == 1.0


def test_grad_norm_returns_zero_without_gradients():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    assert grad_norm([parameter]) == 0.0


def test_resolve_metric_tensors_batches_scalar_conversion():
    metrics = resolve_metric_tensors({"loss": torch.tensor(1.5), "method": "atomic"})

    assert metrics == {"loss": 1.5, "method": "atomic"}
