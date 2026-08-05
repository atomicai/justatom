"""Tests for the InfoNCE / SimCSE / soft-FN ContrastiveLoss family."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from justatom.running.trainer import (
    BiGammaLightningTrainer,
    EncoderOnlyLightningTrainer,
    _pairwise_focal_loss,
    _sample_negative_derangement,
    _scalar_distribution_metrics,
)
from justatom.running.encoders import GammaHybridRunner
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


def test_scalar_distribution_metrics_keeps_alpha_gate_quantiles():
    values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
    metrics = _scalar_distribution_metrics(values, "Alpha")

    assert math.isclose(metrics["AlphaMean"], 0.5)
    assert math.isclose(metrics["AlphaStd"], float(values.std(unbiased=False).item()))
    assert math.isclose(metrics["AlphaMin"], 0.0)
    assert math.isclose(metrics["AlphaP50"], 0.5)
    assert math.isclose(metrics["AlphaMax"], 1.0)
    assert "AlphaP05" in metrics
    assert "AlphaP95" in metrics


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


def test_contrastive_loss_soft_memory_margin_backpropagates_to_margin():
    loss_fn = ContrastiveLoss(temperature=1.0, learnable_temperature=False, decoupled=True, reduction="mean")
    q = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32), dim=-1)
    p = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32), dim=-1)
    memory = F.normalize(torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float32), dim=-1)
    memory_mask = torch.ones(2, 2, dtype=torch.bool)
    margin = torch.full((2,), 0.05, dtype=torch.float32, requires_grad=True)

    loss = loss_fn.info_nce(
        q,
        p,
        memory_negatives=memory,
        memory_negative_mask=memory_mask,
        memory_margin=margin,
        memory_soft_beta=0.05,
    )
    loss.backward()

    assert margin.grad is not None
    assert torch.isfinite(margin.grad).all()
    assert float(margin.grad.abs().sum().item()) > 0.0


def test_contrastive_loss_applies_memory_log_weights_to_denominator():
    loss_fn = ContrastiveLoss(temperature=1.0, learnable_temperature=False, decoupled=True, reduction="none")
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    p = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    memory = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    memory_mask = torch.ones(2, 2, dtype=torch.bool)
    memory_log_weights = torch.log(torch.tensor([[0.25, 1.0], [1.0, 0.5]], dtype=torch.float32))

    per_row = loss_fn.info_nce(
        q,
        p,
        memory_negatives=memory,
        memory_negative_mask=memory_mask,
        memory_log_weights=memory_log_weights,
    )

    sim = q @ p.t()
    memory_sim = q @ memory.t() + memory_log_weights
    eye = torch.eye(2, dtype=torch.bool)
    neg = torch.cat([sim.masked_fill(eye, float("-inf")), memory_sim], dim=1)
    expected = -sim.diagonal() + torch.logsumexp(neg, dim=-1)
    assert torch.allclose(per_row, expected, atol=1e-6)


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
    assert metrics["MemoryBankActiveSimilarityMax"] <= 0.99
    assert metrics["MemoryBankActivePositiveGapMin"] >= 0.01
    assert math.isfinite(metrics["MemoryBankActiveSimilarityP95"])
    assert math.isfinite(metrics["MemoryBankValidSimilarityMax"])


def test_memory_bank_hard_similarity_cap_filters_only_hard_candidates():
    vectors = F.normalize(
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
    )
    ids = {
        "doc_key_id": torch.tensor([1, 2, 3, 4]),
        "content_key_id": torch.tensor([10, 20, 30, 40]),
        "query_key_id": torch.tensor([100, 200, 300, 400]),
    }
    batch = {
        "input_ids": torch.ones(1, 1, dtype=torch.long),
        "doc_key_id": torch.tensor([99]),
        "content_key_id": torch.tensor([999]),
        "query_key_id": torch.tensor([9999]),
    }
    q = F.normalize(torch.tensor([[1.0, 0.0]], dtype=torch.float32), dim=-1)
    p = F.normalize(torch.tensor([[1.0, 0.0]], dtype=torch.float32), dim=-1)

    hard_bank = ContrastiveMemoryBank(
        4,
        mining_mode="hard",
        hard_negatives=3,
        hard_similarity_cap=0.45,
    )
    hard_bank.enqueue(vectors, ids)
    _, hard_mask, hard_metrics = hard_bank.get(
        batch,
        device=torch.device("cpu"),
        dtype=torch.float32,
        query_vectors=q,
        positive_vectors=p,
        step=1,
    )

    assert hard_mask is not None
    assert not hard_mask[0, 0]
    assert not hard_mask[0, 1]
    assert hard_metrics["MemoryBankHardCandidateNegativesMean"] == 2.0
    assert hard_metrics["MemoryBankActiveHardSimilarityMax"] <= 0.45

    random_bank = ContrastiveMemoryBank(
        4,
        mining_mode="random",
        random_negatives=4,
        hard_similarity_cap=0.0,
    )
    random_bank.enqueue(vectors, ids)
    _, random_mask, random_metrics = random_bank.get(
        batch,
        device=torch.device("cpu"),
        dtype=torch.float32,
        query_vectors=q,
        positive_vectors=p,
        step=1,
    )

    assert random_mask is not None
    assert int(random_mask.sum().item()) == 4
    assert random_metrics["MemoryBankActiveHardNegativesMean"] == 0.0
    assert random_metrics["MemoryBankActiveSimilarityMax"] > 0.45


def test_memory_bank_adaptive_hard_suppresses_collision_rows_only():
    bank = ContrastiveMemoryBank(
        5,
        mining_mode="hard",
        hard_negatives=2,
        adaptive_hard=True,
        hard_collision_threshold=0.0,
    )
    bank.enqueue(
        F.normalize(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.95, 0.31, 0.0],
                    [0.0, 0.8, 0.6],
                    [0.0, 0.6, 0.8],
                    [-1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            dim=-1,
        ),
        {
            "doc_key_id": torch.tensor([1, 2, 3, 4, 5]),
            "content_key_id": torch.tensor([10, 20, 30, 40, 50]),
            "query_key_id": torch.tensor([100, 200, 300, 400, 500]),
        },
    )
    batch = {
        "input_ids": torch.ones(2, 1, dtype=torch.long),
        "doc_key_id": torch.tensor([99, 98]),
        "content_key_id": torch.tensor([999, 998]),
        "query_key_id": torch.tensor([9999, 9998]),
    }
    q = F.normalize(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32), dim=-1)
    p = F.normalize(torch.tensor([[0.8, 0.6, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32), dim=-1)

    _, active_mask, metrics = bank.get(
        batch,
        device=torch.device("cpu"),
        dtype=torch.float32,
        query_vectors=q,
        positive_vectors=p,
        step=1,
    )

    assert active_mask is not None
    assert int(active_mask[0].sum().item()) == 0
    assert int(active_mask[1].sum().item()) == 2
    assert metrics["MemoryBankAdaptiveHardEnabled"] == 1.0
    assert metrics["MemoryBankAdaptiveHardAllowedRows"] == 1.0
    assert metrics["MemoryBankAdaptiveHardSuppressedRows"] == 1.0
    assert metrics["MemoryBankActiveHardNegativesMean"] == 1.0


def test_memory_bank_soft_adaptive_hard_downweights_without_suppressing():
    bank = ContrastiveMemoryBank(
        5,
        mining_mode="hard",
        hard_negatives=2,
        adaptive_hard=True,
        adaptive_hard_mode="soft",
        hard_collision_threshold=0.0,
        hard_collision_beta=0.05,
    )
    bank.enqueue(
        F.normalize(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.95, 0.31, 0.0],
                    [0.0, 0.8, 0.6],
                    [0.0, 0.6, 0.8],
                    [-1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            dim=-1,
        ),
        {
            "doc_key_id": torch.tensor([1, 2, 3, 4, 5]),
            "content_key_id": torch.tensor([10, 20, 30, 40, 50]),
            "query_key_id": torch.tensor([100, 200, 300, 400, 500]),
        },
    )
    batch = {
        "input_ids": torch.ones(2, 1, dtype=torch.long),
        "doc_key_id": torch.tensor([99, 98]),
        "content_key_id": torch.tensor([999, 998]),
        "query_key_id": torch.tensor([9999, 9998]),
    }
    q = F.normalize(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32), dim=-1)
    p = F.normalize(torch.tensor([[0.8, 0.6, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32), dim=-1)

    _, active_mask, memory_log_weights, metrics = bank.get(
        batch,
        device=torch.device("cpu"),
        dtype=torch.float32,
        query_vectors=q,
        positive_vectors=p,
        step=1,
        return_log_weights=True,
    )

    assert active_mask is not None
    assert memory_log_weights is not None
    assert int(active_mask[0].sum().item()) == 2
    assert int(active_mask[1].sum().item()) == 2
    row0_active_log_weight = memory_log_weights[0][active_mask[0]].mean()
    row1_active_log_weight = memory_log_weights[1][active_mask[1]].mean()
    assert row0_active_log_weight < row1_active_log_weight
    assert row0_active_log_weight < -0.1
    assert row1_active_log_weight <= 0.0
    assert metrics["MemoryBankAdaptiveHardMode"] == "soft"
    assert 0.0 < metrics["MemoryBankAdaptiveHardWeightMean"] < 1.0


def test_memory_bank_collision_g_uses_pre_margin_candidates():
    bank = ContrastiveMemoryBank(
        3,
        mining_mode="all",
        too_hard_margin=0.05,
    )
    vectors = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.7, 0.3],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    bank.enqueue(
        vectors,
        {
            "doc_key_id": torch.tensor([1, 2, 3]),
            "content_key_id": torch.tensor([10, 20, 30]),
            "query_key_id": torch.tensor([100, 200, 300]),
        },
    )
    batch = {
        "input_ids": torch.ones(1, 1, dtype=torch.long),
        "doc_key_id": torch.tensor([99]),
        "content_key_id": torch.tensor([999]),
        "query_key_id": torch.tensor([9999]),
    }
    q = F.normalize(torch.tensor([[1.0, 0.0]], dtype=torch.float32), dim=-1)
    p = F.normalize(torch.tensor([[0.9, 0.1]], dtype=torch.float32), dim=-1)

    _, active_mask, metrics = bank.get(
        batch,
        device=torch.device("cpu"),
        dtype=torch.float32,
        query_vectors=q,
        positive_vectors=p,
        step=1,
    )

    assert active_mask is not None
    assert not active_mask[0, 0]
    assert metrics["MemoryBankCollisionGMax"] > 0.0
    assert metrics["MemoryBankValidSimilarityMax"] < metrics["MemoryBankCollisionBankMaxSimilarityMax"]


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


class _TinyLanguageModel(nn.Module):
    def __init__(self, output_dims: int = 4):
        super().__init__()
        self.output_dims = output_dims
        self.proj = nn.Linear(output_dims, output_dims)


def _optimizer_param_ids(optimizer):
    return {id(param) for group in optimizer.param_groups for param in group["params"]}


def test_gamma_runner_margin_head_is_zero_init_and_part_of_mixing_parameters():
    runner = GammaHybridRunner(
        model=_TinyLanguageModel(output_dims=4),
        prediction_heads=[],
        gamma_joint=True,
        margin_query_conditional=True,
        margin_base=0.05,
        margin_scale=0.02,
    )
    q = F.normalize(torch.randn(3, 4), dim=-1)

    margins = runner.margin_weights(q)

    assert torch.allclose(margins, torch.full((3,), 0.05), atol=1e-7)
    margin_param_ids = {id(param) for param in runner.margin_parameters()}
    mixing_param_ids = {id(param) for param in runner.mixing_parameters()}
    assert margin_param_ids
    assert margin_param_ids.issubset(mixing_param_ids)


def test_gamma_trainer_margin_regularization_penalizes_margin_drift_and_logs():
    runner = GammaHybridRunner(
        model=_TinyLanguageModel(output_dims=4),
        processor=None,
        prediction_heads=[],
        device="cpu",
        gamma_joint=True,
        margin_query_conditional=True,
        margin_base=0.05,
        margin_scale=0.02,
    )
    with torch.no_grad():
        runner.margin_head[-1].bias.fill_(1.0)
    trainer = BiGammaLightningTrainer(
        runner=runner,
        freeze_encoder=False,
        loss_name="contrastive",
        contrastive_temperature=1.0,
        contrastive_learnable_temperature=False,
        contrastive_decoupled=True,
        memory_bank_soft_mode="soft",
        memory_bank_too_hard_margin=0.05,
        memory_bank_margin_regularization_weight=10.0,
    )
    q = F.normalize(torch.eye(4, dtype=torch.float32), dim=-1)
    p = q.clone()
    batch = {
        "doc_key_id": torch.arange(4),
        "content_key_id": torch.arange(10, 14),
        "query_key_id": torch.arange(20, 24),
    }
    metrics: dict[str, float] = {}

    loss = trainer._compute_contrastive_loss(batch=batch, q_vecs=q, d_vecs=p, metrics=metrics)

    margins = runner.margin_weights(q)
    expected_reg = 10.0 * (margins - 0.05).pow(2).mean()
    base_loss = trainer.loss_fn.info_nce(q, p, reduction="none").mean()
    assert torch.allclose(loss, base_loss + expected_reg, atol=1e-6)
    assert torch.allclose(metrics["ContrastiveMemoryMarginRegLoss"], expected_reg.detach(), atol=1e-6)
    assert metrics["ContrastiveMemoryMarginRegWeight"] == 10.0
    assert metrics["ContrastiveMemoryMarginRegTarget"] == 0.05


def test_gamma_trainer_margin_regularization_uses_unclamped_margin():
    runner = GammaHybridRunner(
        model=_TinyLanguageModel(output_dims=4),
        processor=None,
        prediction_heads=[],
        device="cpu",
        gamma_joint=True,
        margin_query_conditional=True,
        margin_base=0.05,
        margin_scale=0.10,
        margin_max=0.07,
    )
    with torch.no_grad():
        runner.margin_head[-1].bias.fill_(8.0)
    trainer = BiGammaLightningTrainer(
        runner=runner,
        freeze_encoder=False,
        loss_name="contrastive",
        contrastive_temperature=1.0,
        contrastive_learnable_temperature=False,
        contrastive_decoupled=True,
        memory_bank_soft_mode="soft",
        memory_bank_too_hard_margin=0.05,
        memory_bank_margin_regularization_weight=10.0,
    )
    q = F.normalize(torch.eye(4, dtype=torch.float32), dim=-1)
    p = q.clone()
    batch = {
        "doc_key_id": torch.arange(4),
        "content_key_id": torch.arange(10, 14),
        "query_key_id": torch.arange(20, 24),
    }
    metrics: dict[str, float] = {}

    loss = trainer._compute_contrastive_loss(batch=batch, q_vecs=q, d_vecs=p, metrics=metrics)

    raw_margins = runner.margin_raw_weights(q)
    clamped_margins = runner.margin_weights(q)
    expected_reg = 10.0 * (raw_margins - 0.05).pow(2).mean()
    base_loss = trainer.loss_fn.info_nce(q, p, reduction="none").mean()
    assert float(raw_margins.mean().item()) > float(clamped_margins.mean().item())
    assert torch.allclose(loss, base_loss + expected_reg, atol=1e-6)
    assert torch.allclose(metrics["ContrastiveMemoryMarginRegLoss"], expected_reg.detach(), atol=1e-6)
    assert math.isclose(metrics["ContrastiveMemoryMarginMean"], 0.07, rel_tol=0.0, abs_tol=1e-6)
    assert metrics["ContrastiveMemoryMarginRawMean"] > metrics["ContrastiveMemoryMarginMean"]


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
