from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from justatom.training.config import (
    AdaptiveBankConfig,
    MarginConfig,
    MarginMode,
    MemoryBankConfig,
)
from justatom.training.memory_bank import (
    ContrastiveMemoryBank,
    MemorySelection,
    QueryMarginHead,
)


def test_memory_bank_requires_typed_configuration():
    with pytest.raises(TypeError, match="MemoryBankConfig"):
        ContrastiveMemoryBank(8)


def test_query_margin_starts_at_base_and_has_live_gradient():
    head = QueryMarginHead(
        embedding_dim=4,
        config=MarginConfig(
            mode=MarginMode.QUERY,
            base=0.05,
            scale=0.02,
            minimum=0.0,
            maximum=0.15,
            regularization_weight=50.0,
        ),
    )
    queries = F.normalize(torch.randn(3, 4), dim=-1).requires_grad_()

    raw, margin = head(queries)

    torch.testing.assert_close(raw, torch.full((3,), 0.05), atol=1e-7, rtol=0.0)
    torch.testing.assert_close(margin, torch.full((3,), 0.05), atol=1e-7, rtol=0.0)
    (margin.sum() + raw.sum()).backward()
    assert queries.grad is not None
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_memory_bank_returns_noop_selection_before_warmup():
    bank = ContrastiveMemoryBank(MemoryBankConfig(enabled=True, size=8, warmup_steps=2))

    selection = bank.select(
        batch={"doc_key_id": torch.tensor([1, 2])},
        query_vectors=F.normalize(torch.eye(2), dim=-1),
        positive_vectors=F.normalize(torch.eye(2), dim=-1),
        step=0,
    )

    assert isinstance(selection, MemorySelection)
    assert selection.embeddings is None
    assert selection.active_mask is None
    assert selection.log_weights is None


def test_memory_bank_enqueue_always_detaches_graph():
    bank = ContrastiveMemoryBank(MemoryBankConfig(enabled=True, size=4))
    vectors = torch.randn(2, 3, requires_grad=True)

    bank.enqueue(vectors, {"doc_key_id": torch.tensor([1, 2])})

    assert bank.embeddings is not None
    assert not bank.embeddings.requires_grad
    assert bank.embeddings.grad_fn is None


def test_memory_bank_selection_reports_collision_and_soft_hard_weight():
    bank = ContrastiveMemoryBank(
        MemoryBankConfig(
            enabled=True,
            size=4,
            mining="hard",
            hard_negatives=2,
            adaptive=AdaptiveBankConfig(
                enabled=True,
                collision_threshold=0.0,
                collision_beta=0.05,
            ),
        )
    )
    bank.enqueue(
        F.normalize(
            torch.tensor(
                [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]],
                dtype=torch.float32,
            ),
            dim=-1,
        ),
        {"doc_key_id": torch.tensor([1, 2, 3, 4])},
    )
    query = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32), dim=-1).requires_grad_()
    positive = F.normalize(torch.tensor([[0.8, 0.6], [0.8, 0.6]], dtype=torch.float32), dim=-1).requires_grad_()

    selection = bank.select(
        batch={"doc_key_id": torch.tensor([99, 98])},
        query_vectors=query,
        positive_vectors=positive,
        step=1,
    )

    assert selection.embeddings is not None
    assert selection.active_mask is not None
    assert selection.log_weights is not None
    assert selection.collision_g is not None
    assert selection.hard_weights is not None
    assert selection.collision_g[0].item() > 0.0
    assert 0.0 < selection.hard_weights[0].item() < 0.5
    assert not selection.collision_g.requires_grad
    assert not selection.log_weights.requires_grad
    assert int(selection.active_mask.sum()) == 4


def test_memory_bank_filters_same_document_ids():
    bank = ContrastiveMemoryBank(MemoryBankConfig(enabled=True, size=3, mining="all"))
    bank.enqueue(F.normalize(torch.eye(3), dim=-1), {"doc_key_id": torch.tensor([10, 20, 30])})

    selection = bank.select(
        batch={"doc_key_id": torch.tensor([20, 99])},
        query_vectors=F.normalize(torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]), dim=-1),
        positive_vectors=F.normalize(torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]), dim=-1),
        step=0,
    )

    assert selection.active_mask is not None
    assert selection.active_mask.tolist() == [[True, False, True], [True, True, True]]


def test_memory_mass_ramps_after_warmup():
    bank = ContrastiveMemoryBank(
        MemoryBankConfig(
            enabled=True,
            size=32,
            warmup_steps=50,
            mass_ratio=0.5,
            mass_ramp_steps=20,
        )
    )
    assert bank._mass_progress(49) == pytest.approx(0.0)
    assert bank._mass_progress(50) == pytest.approx(0.05)
    assert bank._mass_progress(59) == pytest.approx(0.5)
    assert bank._mass_progress(69) == pytest.approx(1.0)


def test_n8_k12_candidate_weight_is_seven_over_twenty_four():
    bank = ContrastiveMemoryBank(
        MemoryBankConfig(
            enabled=True,
            size=32,
            mass_ratio=0.5,
            mass_ramp_steps=1,
        )
    )
    active = torch.ones(8, 12, dtype=torch.bool)
    log_weights, metrics = bank._normalized_log_weights(active, None, step=0)
    torch.testing.assert_close(log_weights.exp(), torch.full((8, 12), 7.0 / 24.0))
    assert metrics["memory/effective_mass_ratio"] == pytest.approx(0.5)


def test_candidate_weights_compose_and_empty_rows_stay_finite():
    bank = ContrastiveMemoryBank(
        MemoryBankConfig(enabled=True, size=8, mass_ratio=0.5, mass_ramp_steps=1)
    )
    active = torch.tensor([[True, True, False], [False, False, False]])
    candidate = torch.log(torch.tensor([[0.5, 0.25, 1.0], [1.0, 1.0, 1.0]]))
    log_weights, _ = bank._normalized_log_weights(active, candidate, step=0)
    torch.testing.assert_close(log_weights[0, :2].exp(), torch.tensor([0.125, 0.0625]))
    assert torch.isfinite(log_weights).all()
    assert log_weights[1].eq(0.0).all()


def test_normalization_rejects_single_row_contrastive_batch():
    bank = ContrastiveMemoryBank(
        MemoryBankConfig(enabled=True, size=8, mass_ratio=0.5, mass_ramp_steps=1)
    )
    with pytest.raises(ValueError, match="contrastive batch size >= 2"):
        bank._normalized_log_weights(torch.ones(1, 2, dtype=torch.bool), None, step=0)


def test_metric_columns_are_stable_across_mass_warmup():
    bank = ContrastiveMemoryBank(
        MemoryBankConfig(
            enabled=True,
            size=8,
            warmup_steps=1,
            mining="random",
            random_negatives=2,
            mass_ratio=0.5,
            mass_ramp_steps=1,
        )
    )
    vectors = F.normalize(torch.randn(4, 3), dim=-1)
    bank.enqueue(vectors, {"doc_key_id": torch.arange(4)})
    batch = {"doc_key_id": torch.tensor([10, 11])}
    before = bank.select(batch, query_vectors=vectors[:2], positive_vectors=vectors[:2], step=0)
    after = bank.select(batch, query_vectors=vectors[:2], positive_vectors=vectors[:2], step=1)
    assert set(before.metrics) == set(after.metrics)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_normalized_weights_are_finite_on_mps():
    bank = ContrastiveMemoryBank(
        MemoryBankConfig(enabled=True, size=8, mass_ratio=0.5, mass_ramp_steps=1)
    )
    active = torch.tensor([[True, False], [False, False]], device="mps")
    log_weights, _ = bank._normalized_log_weights(active, None, step=0)
    assert torch.isfinite(log_weights).all()


def test_zero_mass_ratio_returns_noop_selection():
    bank = ContrastiveMemoryBank(
        MemoryBankConfig(
            enabled=True,
            size=8,
            warmup_steps=0,
            mining="random",
            random_negatives=2,
            mass_ratio=0.0,
            mass_ramp_steps=1,
        )
    )
    vectors = F.normalize(torch.randn(4, 3), dim=-1)
    bank.enqueue(vectors, {"doc_key_id": torch.arange(4)})
    selection = bank.select(
        {"doc_key_id": torch.tensor([10, 11])},
        query_vectors=vectors[:2],
        positive_vectors=vectors[:2],
        step=0,
    )
    assert selection.embeddings is None
    assert selection.active_mask is None
    assert selection.log_weights is None
