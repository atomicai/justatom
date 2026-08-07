from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from justatom.retrieval.contracts import Embedder, EmbeddingProfile
from justatom.retrieval.embedders.huggingface import HuggingFaceEmbedder
from justatom.retrieval.errors import ConfigurationError


def _positive_integer(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{name} must be a positive integer") from error
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return parsed


@dataclass(frozen=True)
class EmbeddingServerSettings:
    model: str
    device: str
    batch_size: int
    max_length: int

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "EmbeddingServerSettings":
        values = os.environ if env is None else env
        model = values.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B").strip()
        if not model:
            raise ConfigurationError("EMBEDDING_MODEL must be non-empty")
        device = values.get("EMBEDDING_DEVICE", "cpu").strip()
        if not device:
            raise ConfigurationError("EMBEDDING_DEVICE must be non-empty")
        return cls(
            model=model,
            device=device,
            batch_size=_positive_integer(values.get("EMBEDDING_BATCH_SIZE", "8"), "EMBEDDING_BATCH_SIZE"),
            max_length=_positive_integer(values.get("EMBEDDING_MAX_LENGTH", "512"), "EMBEDDING_MAX_LENGTH"),
        )


def build_local_embedder(settings: EmbeddingServerSettings) -> Embedder:
    return HuggingFaceEmbedder(
        model=settings.model,
        device=settings.device,
        profile=EmbeddingProfile(batch_size=settings.batch_size, max_length=settings.max_length),
    )
