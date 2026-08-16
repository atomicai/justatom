from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from justatom.training.anchor_bank import GeometryAnchorBank
from justatom.training.config import AnchorBankConfig


def test_anchor_bank_requires_typed_config_and_detaches_references():
    with pytest.raises(TypeError, match="AnchorBankConfig"):
        GeometryAnchorBank(8)  # type: ignore[arg-type]

    bank = GeometryAnchorBank(AnchorBankConfig(enabled=True, size=3))
    queries = torch.randn(2, 3, requires_grad=True)
    documents = torch.randn(2, 3, requires_grad=True)
    bank.enqueue(
        queries,
        documents,
        {"query_key_id": torch.tensor([1, 2]), "doc_key_id": torch.tensor([10, 20])},
    )

    assert bank.queries is not None and bank.documents is not None
    assert not bank.queries.requires_grad
    assert not bank.documents.requires_grad
    assert bank.queries.grad_fn is None
    assert bank.documents.grad_fn is None


def test_anchor_bank_is_fifo_and_filters_identity_collisions():
    bank = GeometryAnchorBank(AnchorBankConfig(enabled=True, size=3, warmup_steps=1))
    bank.enqueue(
        F.normalize(torch.eye(4), dim=-1),
        F.normalize(torch.eye(4), dim=-1),
        {
            "query_key_id": torch.tensor([1, 2, 3, 4]),
            "doc_key_id": torch.tensor([10, 20, 30, 40]),
        },
    )

    assert bank.current_size == 3
    assert bank.query_key_ids is not None
    assert bank.query_key_ids.tolist() == [2, 3, 4]
    assert (
        bank.select(
            {"query_key_id": torch.tensor([3]), "doc_key_id": torch.tensor([99])},
            step=0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        is None
    )

    selection = bank.select(
        {"query_key_id": torch.tensor([3]), "doc_key_id": torch.tensor([99])},
        step=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert selection is not None
    assert selection.valid_mask.tolist() == [[True, False, True]]


def test_anchor_geometry_kl_is_zero_at_base_and_has_student_gradient_after_drift():
    bank = GeometryAnchorBank(AnchorBankConfig(enabled=True, size=4, warmup_steps=0, temperature=0.1))
    anchors = F.normalize(torch.eye(3), dim=-1)
    bank.enqueue(
        anchors,
        anchors,
        {"query_key_id": torch.tensor([1, 2, 3]), "doc_key_id": torch.tensor([10, 20, 30])},
    )
    selection = bank.select(
        {"query_key_id": torch.tensor([90, 91]), "doc_key_id": torch.tensor([190, 191])},
        step=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert selection is not None
    base_queries = F.normalize(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), dim=-1)
    base_documents = base_queries.clone()

    zero_loss, _ = bank.geometry_loss(
        student_queries=base_queries,
        student_documents=base_documents,
        base_queries=base_queries,
        base_documents=base_documents,
        selection=selection,
    )
    assert zero_loss is not None
    assert zero_loss.item() == pytest.approx(0.0, abs=1e-7)

    drifted_queries = base_queries.roll(1, dims=-1).clone().requires_grad_()
    drifted_documents = base_documents.roll(1, dims=-1).clone().requires_grad_()
    drift_loss, metrics = bank.geometry_loss(
        student_queries=drifted_queries,
        student_documents=drifted_documents,
        base_queries=base_queries,
        base_documents=base_documents,
        selection=selection,
    )
    assert drift_loss is not None
    assert drift_loss.item() > 0.0
    drift_loss.backward()
    assert drifted_queries.grad is not None
    assert drifted_documents.grad is not None
    assert metrics["anchor/forward_kl"] > 0.0
    assert metrics["anchor/reverse_kl"] > 0.0
