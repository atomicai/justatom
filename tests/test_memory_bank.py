from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from justatom.training.config import AdaptiveBankConfig, MarginConfig, MarginMode, MemoryBankConfig
from justatom.training.memory_bank import ContrastiveMemoryBank, MemorySelection, QueryMarginHead


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
    query = F.normalize(torch.tensor([[1.0, 0.0]], dtype=torch.float32), dim=-1).requires_grad_()
    positive = F.normalize(torch.tensor([[0.8, 0.6]], dtype=torch.float32), dim=-1).requires_grad_()

    selection = bank.select(
        batch={"doc_key_id": torch.tensor([99])},
        query_vectors=query,
        positive_vectors=positive,
        step=1,
    )

    assert selection.embeddings is not None
    assert selection.active_mask is not None
    assert selection.log_weights is not None
    assert selection.collision_g is not None
    assert selection.hard_weights is not None
    assert selection.collision_g.item() > 0.0
    assert 0.0 < selection.hard_weights.item() < 0.5
    assert not selection.collision_g.requires_grad
    assert not selection.log_weights.requires_grad
    assert int(selection.active_mask.sum()) == 2


def test_memory_bank_filters_same_document_ids():
    bank = ContrastiveMemoryBank(MemoryBankConfig(enabled=True, size=3, mining="all"))
    bank.enqueue(F.normalize(torch.eye(3), dim=-1), {"doc_key_id": torch.tensor([10, 20, 30])})

    selection = bank.select(
        batch={"doc_key_id": torch.tensor([20])},
        query_vectors=F.normalize(torch.tensor([[0.0, 1.0, 0.0]]), dim=-1),
        positive_vectors=F.normalize(torch.tensor([[0.0, 1.0, 0.0]]), dim=-1),
        step=0,
    )

    assert selection.active_mask is not None
    assert selection.active_mask.tolist() == [[True, False, True]]
