"""Tests for the InfoNCE / SimCSE / soft-FN ContrastiveLoss family."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from justatom.running.trainer import (
    EncoderOnlyLightningTrainer,
    _pairwise_focal_loss,
    _sample_negative_derangement,
)
from justatom.training.loss import ContrastiveLoss, FocalLoss, SoftContrastiveLoss
from justatom.training.memory_bank import ContrastiveMemoryBank


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


def test_contrastive_loss_decoupled_accepts_masked_memory_negatives():
    loss_fn = ContrastiveLoss(temperature=1.0, learnable_temperature=False, decoupled=True, reduction="none")
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    p = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    memory = torch.tensor([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]], dtype=torch.float32)
    memory_mask = torch.tensor([[True, False, False], [False, True, True]])

    per_row = loss_fn.info_nce(q, p, memory_negatives=memory, memory_negative_mask=memory_mask)

    sim = q @ p.t()
    memory_sim = (q @ memory.t()).masked_fill(~memory_mask, float("-inf"))
    eye = torch.eye(2, dtype=torch.bool)
    neg = torch.cat([sim.masked_fill(eye, float("-inf")), memory_sim], dim=1)
    expected = -sim.diagonal() + torch.logsumexp(neg, dim=-1)
    assert torch.allclose(per_row, expected, atol=1e-6)


def test_contrastive_loss_masked_memory_backward_keeps_temperature_grad_finite():
    loss_fn = ContrastiveLoss(temperature=0.05, learnable_temperature=True, decoupled=True, reduction="mean")
    torch.manual_seed(0)
    q = F.normalize(torch.randn(4, 8), dim=-1).requires_grad_()
    p = F.normalize(torch.randn(4, 8), dim=-1).requires_grad_()
    memory = F.normalize(torch.randn(128, 8), dim=-1)
    memory_mask = torch.zeros(4, 128, dtype=torch.bool)
    memory_mask[:, :8] = True

    loss = loss_fn.info_nce(q, p, memory_negatives=memory, memory_negative_mask=memory_mask)
    loss.backward()

    assert torch.isfinite(loss.detach())
    assert loss_fn.log_tau.grad is not None
    assert torch.isfinite(loss_fn.log_tau.grad).all()
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(p.grad).all()


def test_contrastive_loss_clamps_nonfinite_temperature_parameter():
    loss_fn = ContrastiveLoss(temperature=0.05, learnable_temperature=True)
    with torch.no_grad():
        loss_fn.log_tau.fill_(float("nan"))

    changed = loss_fn.clamp_temperature_()

    assert changed
    assert torch.isfinite(loss_fn.log_tau)
    assert math.isclose(float(loss_fn.tau.item()), 0.05, rel_tol=1e-6)


def test_contrastive_loss_decoupled_no_negatives_returns_zero_row():
    loss_fn = ContrastiveLoss(temperature=1.0, learnable_temperature=False, decoupled=True, reduction="none")
    q = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    p = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    per_row = loss_fn.info_nce(q, p)

    assert torch.allclose(per_row, torch.zeros(1))


def test_memory_bank_mixed_mining_warms_up_and_filters_too_hard():
    bank = ContrastiveMemoryBank(
        4,
        warmup_steps=2,
        mining_mode="mixed",
        hard_negatives=2,
        random_negatives=1,
        hard_warmup_steps=2,
        hard_ramp_steps=2,
        too_hard_margin=0.01,
    )
    bank.enqueue(
        F.normalize(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.8, 0.2],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            dim=-1,
        ),
        {
            "doc_key_id": torch.tensor([1, 2, 3, 4]),
            "content_key_id": torch.tensor([10, 20, 30, 40]),
            "query_key_id": torch.tensor([100, 200, 300, 400]),
        },
    )
    batch = {
        "input_ids": torch.ones(1, 1, dtype=torch.long),
        "doc_key_id": torch.tensor([99]),
        "content_key_id": torch.tensor([999]),
        "query_key_id": torch.tensor([9999]),
    }
    q = F.normalize(torch.tensor([[1.0, 0.0]], dtype=torch.float32), dim=-1)
    p = F.normalize(torch.tensor([[1.0, 0.0]], dtype=torch.float32), dim=-1)

    _, warmup_mask, warmup_metrics = bank.get(batch, device=torch.device("cpu"), dtype=torch.float32, step=1)
    assert warmup_mask is None
    assert warmup_metrics["MemoryBankActiveNegativesMean"] == 0.0

    _, active_mask, metrics = bank.get(
        batch,
        device=torch.device("cpu"),
        dtype=torch.float32,
        query_vectors=q,
        positive_vectors=p,
        step=4,
    )

    assert active_mask is not None
    assert not active_mask[0, 0]
    assert 1 <= int(active_mask.sum().item()) <= 3
    assert metrics["MemoryBankActiveHardK"] == 2.0


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


class _DummyEncoderRunner(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(2, 2)


def _optimizer_param_ids(optimizer):
    return {id(param) for group in optimizer.param_groups for param in group["params"]}


def test_encoder_only_optimizer_includes_learnable_temperature_parameter():
    runner = _DummyEncoderRunner()
    trainer = EncoderOnlyLightningTrainer(
        runner=runner,
        loss_name="contrastive",
        contrastive_temperature=0.05,
        contrastive_learnable_temperature=True,
        optimizer_name="adamw",
    )

    optimizer = trainer.configure_optimizers()

    assert id(trainer.loss_fn.log_tau) in _optimizer_param_ids(optimizer)


def test_manual_grad_accumulation_steps_only_on_boundary():
    trainer = EncoderOnlyLightningTrainer(
        runner=_DummyEncoderRunner(),
        loss_name="contrastive",
        grad_acc_steps=3,
        optimizer_name="adamw",
    )

    assert trainer._is_accumulation_start(0)
    assert not trainer._should_step_optimizer(0)
    assert not trainer._should_step_optimizer(1)
    assert trainer._should_step_optimizer(2)
    assert trainer._is_accumulation_start(3)


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
