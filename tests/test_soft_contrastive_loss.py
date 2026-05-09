"""Tests for the InfoNCE / SimCSE / soft-FN ContrastiveLoss family."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from justatom.running.trainer import _pairwise_focal_loss, _sample_negative_derangement
from justatom.training.loss import ContrastiveLoss, FocalLoss, SoftContrastiveLoss


def test_soft_contrastive_loss_matches_expected_formula():
    loss_fn = SoftContrastiveLoss(margin=0.5, size_average=True)
    rep_anchor = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    rep_other = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([1.0, 0.0], dtype=torch.float32)
    loss = loss_fn(rep_anchor, rep_other, labels)
    expected_positive = 0.5 * (0.0**2)
    expected_negative = 0.5 * max(0.0, 0.5 - 1.0) ** 2
    expected = (expected_positive + expected_negative) / 2.0
    assert torch.isclose(loss, torch.tensor(expected, dtype=torch.float32))


def test_soft_contrastive_loss_applies_temperature_to_similarity():
    loss_fn = SoftContrastiveLoss(margin=0.5, size_average=True, temperature=10.0)
    rep_anchor = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    rep_other = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([1.0, 0.0], dtype=torch.float32)
    loss = loss_fn(rep_anchor, rep_other, labels)
    positive_distance = 1.0 - (1.0 / 10.0)
    negative_distance = 1.0 - (0.0 / 10.0)
    expected_positive = 0.5 * positive_distance**2
    expected_negative = 0.5 * max(0.0, 0.5 - negative_distance) ** 2
    expected = (expected_positive + expected_negative) / 2.0
    assert torch.isclose(loss, torch.tensor(expected, dtype=torch.float32))


def test_soft_contrastive_negative_sampler_returns_derangements():
    torch.manual_seed(0)
    observed = set()
    for _ in range(12):
        permutation = _sample_negative_derangement(batch_size=8, device=torch.device("cpu"))
        assert permutation.shape == (8,)
        assert sorted(permutation.tolist()) == list(range(8))
        assert all(idx != value for idx, value in enumerate(permutation.tolist()))
        observed.add(tuple(permutation.tolist()))
    assert len(observed) > 1


def test_contrastive_loss_uses_temperature_scaled_in_batch_infonce():
    loss_fn = ContrastiveLoss(temperature=0.1, reduction="mean", learnable_temperature=False, decoupled=False)
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    positives = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    loss = loss_fn(queries, positives)
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    expected = F.cross_entropy(logits, labels)
    assert torch.isclose(loss, expected, atol=1e-6)


def test_contrastive_loss_learnable_temperature_is_a_parameter():
    loss_fn = ContrastiveLoss(temperature=0.05, learnable_temperature=True)
    assert isinstance(loss_fn.log_tau, nn.Parameter)
    assert loss_fn.log_tau.requires_grad
    assert math.isclose(float(loss_fn.tau.item()), 0.05, rel_tol=1e-6)
    frozen = ContrastiveLoss(temperature=0.05, learnable_temperature=False)
    assert not isinstance(frozen.log_tau, nn.Parameter)
    assert math.isclose(float(frozen.tau.item()), 0.05, rel_tol=1e-6)


def test_contrastive_loss_decoupled_matches_closed_form():
    loss_fn = ContrastiveLoss(temperature=1.0, learnable_temperature=False, decoupled=True, reduction="none")
    torch.manual_seed(0)
    q = F.normalize(torch.randn(4, 8), dim=-1)
    p = F.normalize(torch.randn(4, 8), dim=-1)
    per_row = loss_fn(q, p)
    sim = q @ p.t()
    eye = torch.eye(4, dtype=torch.bool)
    pos = sim.diagonal()
    neg = sim.masked_fill(eye, float("-inf"))
    expected = -pos + torch.logsumexp(neg, dim=-1)
    assert torch.allclose(per_row, expected, atol=1e-6)


def test_contrastive_loss_simcse_term_equals_info_nce_on_alt_view():
    loss_fn = ContrastiveLoss(temperature=0.1, learnable_temperature=False, decoupled=True, reduction="mean")
    torch.manual_seed(1)
    q = F.normalize(torch.randn(3, 5), dim=-1)
    q_alt = F.normalize(torch.randn(3, 5), dim=-1)
    simcse = loss_fn.simcse_term(q, q_alt)
    info_nce = loss_fn.info_nce(q, q_alt)
    assert torch.isclose(simcse, info_nce, atol=1e-6)


def test_contrastive_loss_soft_fn_term_attracts_top_k():
    loss_fn = ContrastiveLoss(temperature=1.0, learnable_temperature=False, reduction="none")
    queries = F.normalize(
        torch.tensor([[1.0, 0.05, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), dim=-1,
    )
    positives = F.normalize(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), dim=-1,
    )
    per_row = loss_fn.soft_fn_term(queries, positives, topk=1)
    sim = queries @ positives.t()
    eye = torch.eye(3, dtype=torch.bool)
    expected_top = sim.masked_fill(eye, float("-inf")).max(dim=-1).values
    expected = -expected_top
    assert torch.allclose(per_row, expected, atol=1e-6)
    assert per_row[0] < per_row[1]


def test_pairwise_focal_loss_uses_positive_class_as_target():
    loss_fn = FocalLoss(gamma=2.0, reduction="mean")
    positive_scores = torch.tensor([2.0, 0.5], dtype=torch.float32)
    negative_scores = torch.tensor([0.5, 1.0], dtype=torch.float32)
    loss = _pairwise_focal_loss(loss_fn, positive_scores, negative_scores)
    logits = torch.stack([positive_scores, negative_scores], dim=1)
    targets = torch.zeros(logits.shape[0], dtype=torch.long)
    pt = F.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
    logpt = F.log_softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
    expected = (-(1.0 - pt).pow(2.0) * logpt).mean()
    assert torch.isclose(loss, expected)
