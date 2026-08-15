from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn

from justatom.training.config import AlphaGateConfig

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class QueryAlphaGate(nn.Module):
    """Query-only controller for per-example auxiliary loss pressure."""

    def __init__(self, embedding_dim: int, config: AlphaGateConfig):
        super().__init__()
        if not config.enabled:
            raise ValueError("QueryAlphaGate requires alpha_gate.enabled=true")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        hidden_dim = config.head.hidden_dim or max(32, min(256, embedding_dim // 2))
        activation = _ACTIVATIONS[config.head.activation]
        layers: list[nn.Module] = []
        current_dim = embedding_dim
        for _ in range(config.head.layers):
            layers.extend((nn.Linear(current_dim, hidden_dim), activation()))
            if config.head.dropout > 0.0:
                layers.append(nn.Dropout(config.head.dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))

        self.network = nn.Sequential(*layers)
        self.config = config
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

    def logits(self, queries: torch.Tensor) -> torch.Tensor:
        if queries.ndim != 2:
            raise ValueError(f"queries must have shape [batch, dim], got {tuple(queries.shape)}")
        if queries.shape[-1] != self.embedding_dim:
            raise ValueError(f"queries embedding dimension must be {self.embedding_dim}, got {queries.shape[-1]}")
        return self.network(queries).squeeze(-1)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(queries))

    def parameters_for_optimizer(self) -> Iterator[nn.Parameter]:
        return self.parameters()

    def metadata(self) -> dict[str, object]:
        return {
            "input": "query",
            "layers": self.config.head.layers,
            "hidden_dim": self.hidden_dim,
            "dropout": self.config.head.dropout,
            "activation": self.config.head.activation,
        }
