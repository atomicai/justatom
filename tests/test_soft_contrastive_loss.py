"""Focused tests for reusable contrastive-loss kernels."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from justatom.training.loss import ContrastiveLoss, FocalLoss, SoftContrastiveLoss


def test_soft_contrastive_loss_matches_pairwise_formula():
    loss_fn = SoftContrastiveLoss(margin=0.5, size_average=True)
    anchors = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    others = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([1.0, 0.0])

    loss = loss_fn(anchors, others, labels)

    expected = torch.tensor((0.5 * 0.0**2 + 0.5 * max(0.0, 0.5 - 1.0) ** 2) / 2.0)
    torch.testing.assert_close(loss, expected)


def test_soft_contrastive_loss_applies_temperature_to_similarity():
    loss_fn = SoftContrastiveLoss(margin=0.5, temperature=10.0)
    anchors = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    others = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([1.0, 0.0])

    loss = loss_fn(anchors, others, labels)

    positive_distance = 1.0 - 0.1
    negative_distance = 1.0
    expected = torch.tensor(
        (0.5 * positive_distance**2 + 0.5 * max(0.0, 0.5 - negative_distance) ** 2) / 2.0
    )
    torch.testing.assert_close(loss, expected)


def test_focal_loss_targets_the_requested_class():
    loss_fn = FocalLoss(gamma=2.0, reduction="mean")
    logits = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    targets = torch.zeros(2, dtype=torch.long)

    loss = loss_fn(logits, targets)

    probabilities = F.softmax(logits, dim=1)[:, 0]
    log_probabilities = F.log_softmax(logits, dim=1)[:, 0]
    expected = (-(1.0 - probabilities).pow(2.0) * log_probabilities).mean()
    torch.testing.assert_close(loss, expected)


def test_contrastive_loss_uses_temperature_scaled_in_batch_infonce():
    loss_fn = ContrastiveLoss(
        temperature=0.1,
        reduction="mean",
        learnable_temperature=False,
        decoupled=False,
    )
    embeddings = torch.eye(2)

    loss = loss_fn(embeddings, embeddings)

    expected = F.cross_entropy(torch.tensor([[10.0, 0.0], [0.0, 10.0]]), torch.tensor([0, 1]))
    torch.testing.assert_close(loss, expected)


def test_learnable_temperature_is_registered_and_clamped():
    loss_fn = ContrastiveLoss(temperature=0.05, learnable_temperature=True)
    assert isinstance(loss_fn.log_tau, nn.Parameter)
    assert math.isclose(float(loss_fn.tau.item()), 0.05, rel_tol=1e-6)

    with torch.no_grad():
        loss_fn.log_tau.fill_(float("nan"))

    assert loss_fn.clamp_temperature_()
    assert torch.isfinite(loss_fn.log_tau)
    assert math.isclose(float(loss_fn.tau.item()), 0.05, rel_tol=1e-6)


def test_masked_memory_backward_keeps_all_gradients_finite():
    loss_fn = ContrastiveLoss(
        temperature=0.05,
        learnable_temperature=True,
        decoupled=True,
    )
    torch.manual_seed(0)
    queries = F.normalize(torch.randn(4, 8), dim=-1).requires_grad_()
    positives = F.normalize(torch.randn(4, 8), dim=-1).requires_grad_()
    memory = F.normalize(torch.randn(32, 8), dim=-1)
    mask = torch.zeros(4, 32, dtype=torch.bool)
    mask[:, :8] = True

    loss = loss_fn.info_nce(
        queries,
        positives,
        memory_negatives=memory,
        memory_negative_mask=mask,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert loss_fn.log_tau.grad is not None and torch.isfinite(loss_fn.log_tau.grad).all()
    assert queries.grad is not None and torch.isfinite(queries.grad).all()
    assert positives.grad is not None and torch.isfinite(positives.grad).all()


def test_single_row_decoupled_loss_without_negatives_is_zero():
    loss_fn = ContrastiveLoss(
        temperature=1.0,
        learnable_temperature=False,
        decoupled=True,
        reduction="none",
    )
    embedding = torch.tensor([[1.0, 0.0]])

    per_row = loss_fn.info_nce(embedding, embedding)

    torch.testing.assert_close(per_row, torch.zeros(1))


def test_simcse_term_reuses_infonce_kernel():
    loss_fn = ContrastiveLoss(
        temperature=0.1,
        learnable_temperature=False,
        decoupled=True,
    )
    torch.manual_seed(1)
    queries = F.normalize(torch.randn(3, 5), dim=-1)
    alternate = F.normalize(torch.randn(3, 5), dim=-1)

    torch.testing.assert_close(
        loss_fn.simcse_term(queries, alternate),
        loss_fn.info_nce(queries, alternate),
    )
