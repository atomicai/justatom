from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from justatom.training.config import MarginConfig, MarginMode, ObjectiveConfig
from justatom.training.memory_bank import MemorySelection
from justatom.training.objective import ContrastiveObjective, ObjectiveInputs


def test_objective_vanilla_has_no_auxiliary_components():
    objective = ContrastiveObjective(ObjectiveConfig(temperature=1.0, learnable_temperature=False))
    queries = F.normalize(torch.eye(3), dim=-1).requires_grad_()
    positives = queries.detach().clone().requires_grad_()

    output = objective(ObjectiveInputs(queries=queries, positives=positives))

    assert output.loss.ndim == 0
    torch.testing.assert_close(output.primary_loss, output.loss)
    assert output.memory_loss is None
    assert output.simcse_per_row is None
    assert output.metrics["loss/alpha_aux"] == 0.0
    assert output.metrics["loss/memory_margin_regularization"] == 0.0


def test_objective_atom_gate_detaches_alpha_only_from_auxiliary_gradient():
    torch.manual_seed(1)
    objective = ContrastiveObjective(
        ObjectiveConfig(
            temperature=1.0,
            learnable_temperature=False,
            decoupled=False,
            simcse_dropout_weight=0.1,
        )
    )
    queries = F.normalize(torch.randn(3, 4), dim=-1).requires_grad_()
    positives = F.normalize(torch.randn(3, 4), dim=-1).requires_grad_()
    alternate_queries = F.normalize(torch.randn(3, 4), dim=-1).requires_grad_()
    alpha_logits = torch.tensor([-1.3862944, 0.0, 1.3862944], requires_grad=True)

    output = objective(
        ObjectiveInputs(
            queries=queries,
            positives=positives,
            query_alt=alternate_queries,
            alpha_logits=alpha_logits,
            alpha_supervision_weight=0.0,
        )
    )

    assert output.simcse_per_row is not None
    expected = (output.main_per_row + (1.0 - torch.sigmoid(alpha_logits).detach()) * 0.1 * output.simcse_per_row).mean()
    torch.testing.assert_close(output.loss, expected)

    high_alpha = objective(
        ObjectiveInputs(
            queries=queries,
            positives=positives,
            query_alt=alternate_queries,
            alpha_logits=torch.full_like(alpha_logits, torch.logit(torch.tensor(0.8))),
            alpha_supervision_weight=0.0,
        )
    )
    low_alpha = objective(
        ObjectiveInputs(
            queries=queries,
            positives=positives,
            query_alt=alternate_queries,
            alpha_logits=torch.full_like(alpha_logits, torch.logit(torch.tensor(0.2))),
            alpha_supervision_weight=0.0,
        )
    )
    assert low_alpha.loss.detach() > high_alpha.loss.detach()

    auxiliary_only = output.loss - output.main_per_row.mean()
    auxiliary_only.backward()
    assert alpha_logits.grad is None
    assert queries.grad is not None and float(queries.grad.abs().sum()) > 0.0
    assert alternate_queries.grad is not None and float(alternate_queries.grad.abs().sum()) > 0.0


def test_atom_gate_uses_detached_positive_confidence_with_learnable_temperature():
    objective = ContrastiveObjective(
        ObjectiveConfig(temperature=0.7, learnable_temperature=True, decoupled=False)
    )
    q = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1).requires_grad_()
    p = F.normalize(torch.tensor([[1.0, 0.0], [0.6, 0.8]]), dim=-1).requires_grad_()
    alpha_logits = torch.tensor([-0.4, 0.7], requires_grad=True)
    output = objective(
        ObjectiveInputs(
            queries=q,
            positives=p,
            alpha_logits=alpha_logits,
            alpha_supervision_weight=0.3,
        )
    )
    target = torch.softmax((q.detach() @ p.detach().T) / objective.kernel.tau.detach(), dim=-1).diagonal()
    bce = F.binary_cross_entropy_with_logits(alpha_logits, target, reduction="none")
    torch.testing.assert_close(output.alpha_target, target)
    torch.testing.assert_close(output.alpha_supervision_per_row, bce)
    torch.testing.assert_close(output.loss, output.main_per_row.mean() + 0.3 * bce.mean())


def test_alpha_bce_does_not_update_retrieval_embeddings_or_temperature():
    objective = ContrastiveObjective(
        ObjectiveConfig(temperature=0.7, learnable_temperature=True, decoupled=False)
    )
    q = F.normalize(torch.randn(3, 4), dim=-1).requires_grad_()
    p = F.normalize(torch.randn(3, 4), dim=-1).requires_grad_()
    alpha_logits = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    output = objective(
        ObjectiveInputs(queries=q, positives=p, alpha_logits=alpha_logits, alpha_supervision_weight=1.0)
    )
    output.alpha_supervision_per_row.mean().backward()
    assert alpha_logits.grad is not None and torch.isfinite(alpha_logits.grad).all()
    assert float(alpha_logits.grad.abs().sum()) > 0.0
    assert q.grad is None and p.grad is None
    assert objective.kernel.log_tau.grad is None


def test_alpha_bce_is_minimized_at_soft_target():
    target = torch.tensor([0.2, 0.8])
    at_target = F.binary_cross_entropy_with_logits(torch.logit(target), target)
    away_from_target = F.binary_cross_entropy_with_logits(torch.logit(torch.tensor([0.4, 0.6])), target)
    assert at_target < away_from_target


def test_hard_row_receives_more_simcse_pressure_than_easy_row():
    alpha = torch.tensor([0.2, 0.8])
    simcse = torch.tensor([2.0, 2.0])
    weighted = (1.0 - alpha.detach()) * simcse
    torch.testing.assert_close(weighted, torch.tensor([1.6, 0.4]))


def test_objective_regularizes_raw_margin_to_constant_base():
    config = MarginConfig(
        mode=MarginMode.QUERY,
        base=0.05,
        scale=0.02,
        regularization_weight=50.0,
    )
    objective = ContrastiveObjective(ObjectiveConfig(temperature=1.0, learnable_temperature=False))
    queries = F.normalize(torch.randn(3, 4), dim=-1)
    positives = F.normalize(torch.randn(3, 4), dim=-1)
    raw = torch.tensor([0.04, 0.05, 0.07], requires_grad=True)

    output = objective(
        ObjectiveInputs(
            queries=queries,
            positives=positives,
            margin=raw.clamp(0.0, 0.15),
            raw_margin=raw,
        ),
        margin_config=config,
    )

    expected = 50.0 * (raw - 0.05).pow(2).mean()
    torch.testing.assert_close(
        output.metrics["loss/memory_margin_regularization_tensor"],
        expected,
    )


def test_objective_atomic_forwards_bank_columns_and_live_margin():
    objective = ContrastiveObjective(ObjectiveConfig(temperature=0.05, learnable_temperature=False))
    queries = F.normalize(torch.randn(2, 4), dim=-1).requires_grad_()
    positives = F.normalize(torch.randn(2, 4), dim=-1).requires_grad_()
    memory_vectors = F.normalize(torch.randn(3, 4), dim=-1)
    margin = torch.full((2,), 0.05, requires_grad=True)
    memory = MemorySelection(
        embeddings=memory_vectors,
        active_mask=torch.ones(2, 3, dtype=torch.bool),
        log_weights=torch.zeros(2, 3),
        collision_g=torch.tensor([-0.1, 0.1]),
        hard_weights=torch.tensor([0.8, 0.2]),
        metrics={},
    )

    output = objective(
        ObjectiveInputs(
            queries=queries,
            positives=positives,
            memory=memory,
            margin=margin,
        ),
        margin_config=MarginConfig(mode=MarginMode.QUERY, admission_beta=0.05),
    )
    assert output.memory_loss is not None
    assert output.memory_per_row is not None
    torch.testing.assert_close(output.loss, output.primary_loss + output.memory_loss)
    output.loss.backward()

    assert margin.grad is not None and float(margin.grad.abs().sum()) > 0.0
    assert output.metrics["memory/active_negatives_mean"] == 3.0


def test_normalized_bank_loss_is_invariant_to_duplicate_count():
    objective = ContrastiveObjective(
        ObjectiveConfig(temperature=1.0, learnable_temperature=False, decoupled=False)
    )
    q = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    p = q.clone()
    vector = F.normalize(torch.tensor([[1.0, 1.0]]), dim=-1)

    def output_for(count):
        weight = 0.5 * (q.shape[0] - 1) / count
        memory = MemorySelection(
            embeddings=vector.repeat(count, 1),
            active_mask=torch.ones(2, count, dtype=torch.bool),
            log_weights=torch.full((2, count), math.log(weight)),
            collision_g=None,
            hard_weights=None,
            metrics={},
        )
        return objective(ObjectiveInputs(queries=q, positives=p, memory=memory))

    one, four = output_for(1), output_for(4)

    torch.testing.assert_close(one.loss, four.loss)
    torch.testing.assert_close(one.memory_per_row, four.memory_per_row)
    torch.testing.assert_close(one.loss, one.primary_loss + one.memory_loss)
    torch.testing.assert_close(four.loss, four.primary_loss + four.memory_loss)
    assert torch.isfinite(one.loss)
    assert torch.isfinite(four.loss)


def test_row_without_valid_bank_candidates_equals_main_loss():
    objective = ContrastiveObjective(
        ObjectiveConfig(temperature=1.0, learnable_temperature=False, decoupled=False)
    )
    q = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    p = q.clone()
    memory = MemorySelection(
        embeddings=F.normalize(torch.tensor([[1.0, 1.0]]), dim=-1),
        active_mask=torch.tensor([[True], [False]]),
        log_weights=torch.zeros(2, 1),
        collision_g=None,
        hard_weights=None,
        metrics={},
    )

    plain = objective(ObjectiveInputs(queries=q, positives=p))
    augmented = objective(ObjectiveInputs(queries=q, positives=p, memory=memory))

    assert augmented.memory_per_row is not None
    assert augmented.memory_per_row[0] > 0.0
    assert torch.equal(augmented.memory_per_row[1], torch.zeros_like(augmented.memory_per_row[1]))
    torch.testing.assert_close(
        augmented.main_per_row[1] + augmented.memory_per_row[1],
        plain.main_per_row[1],
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_normalized_bank_objective_gradients_are_finite_on_mps():
    objective = ContrastiveObjective(
        ObjectiveConfig(temperature=0.05, learnable_temperature=True, decoupled=False)
    ).to("mps")
    q = F.normalize(torch.randn(2, 4, device="mps"), dim=-1).requires_grad_()
    p = F.normalize(torch.randn(2, 4, device="mps"), dim=-1).requires_grad_()
    memory = MemorySelection(
        embeddings=F.normalize(torch.randn(3, 4, device="mps"), dim=-1),
        active_mask=torch.tensor([[True, True, False], [True, False, True]], device="mps"),
        log_weights=torch.full((2, 3), math.log(0.25), device="mps"),
        collision_g=None,
        hard_weights=None,
        metrics={},
    )

    output = objective(ObjectiveInputs(queries=q, positives=p, memory=memory))
    output.loss.backward()

    assert torch.isfinite(output.loss)
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert objective.kernel.log_tau.grad is not None and torch.isfinite(objective.kernel.log_tau.grad).all()


def test_objective_rejects_alpha_without_auxiliary_view():
    objective = ContrastiveObjective(ObjectiveConfig(temperature=1.0, learnable_temperature=False, simcse_dropout_weight=0.1))
    embeddings = F.normalize(torch.eye(3), dim=-1)

    try:
        objective(ObjectiveInputs(queries=embeddings, positives=embeddings, alpha_logits=torch.zeros(3)))
    except ValueError as exc:
        assert "query_alt" in str(exc)
    else:
        raise AssertionError("alpha without query_alt must be rejected")
