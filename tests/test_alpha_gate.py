from __future__ import annotations

import pytest
import torch

from justatom.training.alpha_gate import QueryAlphaGate
from justatom.training.config import AlphaGateConfig, AlphaHeadConfig


def test_query_alpha_gate_returns_one_value_per_query():
    gate = QueryAlphaGate(
        embedding_dim=4,
        config=AlphaGateConfig(
            enabled=True,
            head=AlphaHeadConfig(layers=1, hidden_dim=8, dropout=0.0, activation="gelu"),
        ),
    )
    queries = torch.randn(3, 4, requires_grad=True)

    alpha = gate(queries)

    assert alpha.shape == (3,)
    assert torch.all((alpha > 0.0) & (alpha < 1.0))


def test_query_alpha_gate_backpropagates_from_per_query_weights():
    torch.manual_seed(0)
    gate = QueryAlphaGate(embedding_dim=4, config=AlphaGateConfig(enabled=True))
    queries = torch.randn(3, 4, requires_grad=True)

    loss = ((1.0 - gate(queries)) * torch.tensor([1.0, 2.0, 3.0])).mean()
    loss.backward()

    assert queries.grad is not None and torch.isfinite(queries.grad).all()
    assert all(parameter.grad is not None for parameter in gate.parameters())


def test_query_alpha_gate_rejects_document_argument_by_signature():
    gate = QueryAlphaGate(embedding_dim=4, config=AlphaGateConfig(enabled=True))
    queries = torch.randn(2, 4)
    documents = torch.randn(2, 4)

    with pytest.raises(TypeError):
        gate(queries, documents)


def test_query_alpha_gate_metadata_records_resolved_hidden_dim():
    gate = QueryAlphaGate(embedding_dim=768, config=AlphaGateConfig(enabled=True))

    assert gate.metadata() == {
        "input": "query",
        "layers": 1,
        "hidden_dim": 256,
        "dropout": 0.0,
        "activation": "gelu",
    }


def test_query_alpha_gate_rejects_non_matrix_input():
    gate = QueryAlphaGate(embedding_dim=4, config=AlphaGateConfig(enabled=True))

    with pytest.raises(ValueError, match=r"\[batch, dim\]"):
        gate(torch.randn(2, 3, 4))
