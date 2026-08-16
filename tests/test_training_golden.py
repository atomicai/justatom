from __future__ import annotations

import torch
import torch.nn.functional as F

from justatom.training.config import ObjectiveConfig
from justatom.training.loss import ContrastiveLoss
from justatom.training.objective import ContrastiveObjective, ObjectiveInputs


def golden_embeddings() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    queries = F.normalize(
        torch.tensor(
            [[1.0, 0.2, 0.0], [0.1, 1.0, 0.2], [0.0, 0.2, 1.0]],
            dtype=torch.float32,
        ),
        dim=-1,
    ).requires_grad_()
    positives = F.normalize(
        torch.tensor(
            [[0.9, 0.1, 0.0], [0.2, 0.9, 0.1], [0.1, 0.1, 0.9]],
            dtype=torch.float32,
        ),
        dim=-1,
    ).requires_grad_()
    bank = F.normalize(
        torch.tensor(
            [[0.8, 0.2, 0.0], [0.0, 0.8, 0.2], [0.2, 0.0, 0.8], [-1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    return queries, positives, bank


def test_golden_dcl_ablation_matches_decoupled_closed_form():
    queries, positives, _ = golden_embeddings()
    loss_fn = ContrastiveLoss(
        temperature=0.05,
        learnable_temperature=False,
        decoupled=True,
        reduction="none",
    )

    actual = loss_fn.info_nce(queries, positives)

    query_norm = F.normalize(queries, dim=-1)
    positive_norm = F.normalize(positives, dim=-1)
    similarities = query_norm @ positive_norm.T / 0.05
    diagonal = torch.eye(similarities.shape[0], dtype=torch.bool)
    expected = -similarities.diagonal() + torch.logsumexp(
        similarities.masked_fill(diagonal, -1e9),
        dim=-1,
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    actual.mean().backward()
    assert queries.grad is not None and torch.isfinite(queries.grad).all()
    assert positives.grad is not None and torch.isfinite(positives.grad).all()


def test_golden_atom_gate_uses_query_alpha_for_auxiliary_pressure():
    queries, positives, _ = golden_embeddings()
    alternate_queries = F.normalize(queries.detach() + 0.03, dim=-1).requires_grad_()
    alpha_logits = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    alpha = torch.sigmoid(alpha_logits)
    loss_fn = ContrastiveLoss(
        temperature=0.05,
        learnable_temperature=False,
        decoupled=False,
    )

    main = loss_fn.info_nce(queries, positives, reduction="none")
    simcse = loss_fn.simcse_term(queries, alternate_queries, reduction="none")
    actual = (main + (1.0 - alpha.detach()) * 0.1 * simcse).mean()
    expected = torch.mean(main + (1.0 - torch.sigmoid(alpha_logits).detach()) * 0.1 * simcse)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    actual.backward()
    assert alpha_logits.grad is None
    assert queries.grad is not None and torch.isfinite(queries.grad).all()
    assert alternate_queries.grad is not None and torch.isfinite(alternate_queries.grad).all()


def test_golden_objective_decomposes_atom_gate_terms_exactly():
    queries, positives, _ = golden_embeddings()
    alternate_queries = F.normalize(queries.detach() + 0.03, dim=-1).requires_grad_()
    alpha_logits = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    objective = ContrastiveObjective(
        ObjectiveConfig(
            temperature=0.05,
            learnable_temperature=False,
            decoupled=False,
            simcse_dropout_weight=0.1,
        )
    )

    output = objective(
        ObjectiveInputs(
            queries=queries,
            positives=positives,
            query_alt=alternate_queries,
            alpha_logits=alpha_logits,
            alpha_supervision_weight=0.3,
        )
    )

    assert output.simcse_per_row is not None
    assert output.alpha_supervision_per_row is not None
    expected_retrieval = output.main_per_row.mean()
    expected_auxiliary = (0.1 * (1.0 - torch.sigmoid(alpha_logits).detach()) * output.simcse_per_row).mean()
    expected_head = 0.3 * output.alpha_supervision_per_row.mean()
    torch.testing.assert_close(output.retrieval_loss, expected_retrieval, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output.auxiliary_loss, expected_auxiliary, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output.head_loss, expected_head, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        output.primary_loss,
        output.retrieval_loss + output.auxiliary_loss + output.head_loss,
        atol=1e-6,
        rtol=1e-6,
    )


def test_golden_atomic_adds_weighted_bank_logits_and_live_margin_gradient():
    queries, positives, bank = golden_embeddings()
    margin = torch.full((queries.shape[0],), 0.05, requires_grad=True)
    hard_weight = torch.tensor([0.25, 0.5, 1.0])
    mask = torch.ones(queries.shape[0], bank.shape[0], dtype=torch.bool)
    log_weights = torch.log(hard_weight).view(-1, 1).expand(-1, bank.shape[0])
    loss_fn = ContrastiveLoss(
        temperature=0.05,
        learnable_temperature=False,
        decoupled=False,
        reduction="none",
    )

    actual = loss_fn.info_nce(
        queries,
        positives,
        memory_negatives=bank,
        memory_negative_mask=mask,
        memory_log_weights=log_weights,
        memory_margin=margin,
        memory_soft_beta=0.05,
    )

    query_norm = F.normalize(queries, dim=-1)
    positive_norm = F.normalize(positives, dim=-1)
    bank_norm = F.normalize(bank, dim=-1)
    current = query_norm @ positive_norm.T / 0.05
    positive_cosine = (query_norm * positive_norm).sum(dim=-1, keepdim=True)
    bank_cosine = query_norm @ bank_norm.T
    admission = torch.sigmoid((positive_cosine - margin.view(-1, 1) - bank_cosine) / 0.05)
    bank_logits = bank_cosine / 0.05 + log_weights + torch.log(admission.clamp_min(1e-8))
    logits = torch.cat([current, bank_logits], dim=1)
    expected = -current.diagonal() + torch.logsumexp(logits, dim=-1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    actual.mean().backward()
    assert margin.grad is not None and torch.isfinite(margin.grad).all()
    assert float(margin.grad.abs().sum()) > 0.0
