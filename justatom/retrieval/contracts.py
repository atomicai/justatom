from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from justatom.etc.schema import Document
from justatom.retrieval.errors import ConfigurationError, EmbeddingResponseError


@dataclass(frozen=True)
class EmbeddingProfile:
    query_prefix: str = ""
    document_prefix: str = ""
    max_length: int = 512
    batch_size: int = 64
    skip_prefix_if_present: bool = True

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ConfigurationError("max_length must be positive")
        if self.batch_size <= 0:
            raise ConfigurationError("batch_size must be positive")


class SearchMode(str, Enum):
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


def apply_prefix(text: str, prefix: str, *, skip_if_present: bool) -> str:
    if not prefix or (skip_if_present and text.startswith(prefix)):
        return text
    return f"{prefix}{text}"


def validate_embeddings(vectors: Sequence[Sequence[float]], *, expected_count: int) -> list[list[float]]:
    if len(vectors) != expected_count:
        raise EmbeddingResponseError(f"Expected {expected_count} vectors, received {len(vectors)}")
    normalized = [[float(value) for value in vector] for vector in vectors]
    dimensions = {len(vector) for vector in normalized}
    if expected_count and (0 in dimensions or len(dimensions) != 1):
        raise EmbeddingResponseError("Embedding vectors must have one non-zero dimension")
    if any(not math.isfinite(value) for vector in normalized for value in vector):
        raise EmbeddingResponseError("Embedding vectors must contain finite values")
    return normalized


@runtime_checkable
class Embedder(Protocol):
    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def close(self) -> None: ...


@runtime_checkable
class DocumentStore(Protocol):
    async def write_documents(self, documents: Sequence[Document]) -> int: ...

    async def search_vector(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        include_vectors: bool = False,
    ) -> list[list[Document]]: ...

    async def search_keywords(
        self,
        queries: Sequence[str],
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[list[Document]]: ...

    async def search_hybrid(
        self,
        queries: Sequence[str],
        vectors: Sequence[Sequence[float]],
        *,
        alpha: float,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        include_vectors: bool = False,
    ) -> list[list[Document]]: ...

    async def count_documents(self) -> int: ...

    async def clear(self) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    async def retrieve(self, query: str, *, top_k: int = 5, **kwargs: Any) -> list[Document]: ...

    async def retrieve_many(
        self,
        queries: Sequence[str],
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[list[Document]]: ...
