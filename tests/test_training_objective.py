from __future__ import annotations

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
    alpha = torch.tensor([0.2, 0.5, 0.8], requires_grad=True)

    output = objective(
        ObjectiveInputs(
            queries=queries,
            positives=positives,
            query_alt=alternate_queries,
            alpha=alpha,
        )
    )

    assert output.simcse_per_row is not None
    expected = (output.main_per_row + (1.0 - alpha.detach()) * 0.1 * output.simcse_per_row).mean()
    torch.testing.assert_close(output.loss, expected)

    high_alpha = objective(
        ObjectiveInputs(
            queries=queries,
            positives=positives,
            query_alt=alternate_queries,
            alpha=torch.full_like(alpha, 0.8),
        )
    )
    low_alpha = objective(
        ObjectiveInputs(
            queries=queries,
            positives=positives,
            query_alt=alternate_queries,
            alpha=torch.full_like(alpha, 0.2),
        )
    )
    assert low_alpha.loss.detach() > high_alpha.loss.detach()

    auxiliary_only = output.loss - output.main_per_row.mean()
    auxiliary_only.backward()
    assert alpha.grad is None
    assert queries.grad is not None and float(queries.grad.abs().sum()) > 0.0
    assert alternate_queries.grad is not None and float(alternate_queries.grad.abs().sum()) > 0.0


def test_objective_atom_gate_keeps_alpha_live_for_pair_supervision():
    objective = ContrastiveObjective(
        ObjectiveConfig(
            temperature=1.0,
            learnable_temperature=False,
            decoupled=False,
        )
    )
    queries = F.normalize(torch.eye(2, 4), dim=-1)
    positives = queries.clone()
    alpha = torch.tensor([0.4, 0.6], requires_grad=True)
    semantic = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    lexical = torch.tensor([[0.1, 0.8], [0.9, 0.2]])

    output = objective(
        ObjectiveInputs(
            queries=queries,
            positives=positives,
            alpha=alpha,
            semantic_pair_scores=semantic,
            lexical_pair_scores=lexical,
            alpha_mix_weight=0.3,
        )
    )
    output.loss.backward()

    assert alpha.grad is not None
    assert torch.isfinite(alpha.grad).all()
    assert float(alpha.grad.abs().sum()) > 0.0


def test_objective_atom_gate_keeps_alpha_live_for_entropy_regularization():
    objective = ContrastiveObjective(
        ObjectiveConfig(
            temperature=1.0,
            learnable_temperature=False,
            decoupled=False,
        )
    )
    embeddings = F.normalize(torch.eye(2, 4), dim=-1)
    alpha = torch.tensor([0.2, 0.7], requires_grad=True)

    output = objective(
        ObjectiveInputs(
            queries=embeddings,
            positives=embeddings,
            alpha=alpha,
            alpha_entropy_weight=0.1,
        )
    )
    output.loss.backward()

    assert alpha.grad is not None
    assert torch.isfinite(alpha.grad).all()
    assert float(alpha.grad.abs().sum()) > 0.0


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


def test_objective_rejects_alpha_without_auxiliary_view():
    objective = ContrastiveObjective(ObjectiveConfig(temperature=1.0, learnable_temperature=False, simcse_dropout_weight=0.1))
    embeddings = F.normalize(torch.eye(3), dim=-1)

    try:
        objective(ObjectiveInputs(queries=embeddings, positives=embeddings, alpha=torch.full((3,), 0.5)))
    except ValueError as exc:
        assert "query_alt" in str(exc)
    else:
        raise AssertionError("alpha without query_alt must be rejected")
