from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Any

from justatom.etc.schema import Document
from justatom.retrieval.contracts import DocumentStore, Embedder, validate_embeddings
from justatom.retrieval.errors import RetrievalError


def _validate_top_k(top_k: object) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return top_k


def _validate_alpha(alpha: object) -> None:
    if isinstance(alpha, bool) or not isinstance(alpha, Number):
        raise ValueError("alpha must be a finite numeric value in [0, 1]")
    try:
        numeric_alpha = float(alpha)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("alpha must be a finite numeric value in [0, 1]") from error
    if not math.isfinite(numeric_alpha) or not 0 <= numeric_alpha <= 1:
        raise ValueError("alpha must be a finite numeric value in [0, 1]")


def _validate_result_groups(results: list[list[Document]], *, expected_count: int) -> list[list[Document]]:
    if not isinstance(results, list) or len(results) != expected_count or any(not isinstance(row, list) for row in results):
        raise RetrievalError(f"Expected {expected_count} result groups from document store")
    return results


class KeywordRetriever:
    def __init__(self, store: DocumentStore):
        self.store = store

    async def retrieve(self, query: str, *, top_k: int = 5, **kwargs: Any) -> list[Document]:
        results = await self.retrieve_many([query], top_k=top_k, **kwargs)
        return results[0]

    async def retrieve_many(
        self,
        queries: Sequence[str],
        *,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> list[list[Document]]:
        top_k = _validate_top_k(top_k)
        if not queries:
            return []
        results = await self.store.search_keywords(queries, top_k=top_k, filters=filters)
        return _validate_result_groups(results, expected_count=len(queries))


class VectorRetriever:
    def __init__(self, store: DocumentStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    async def retrieve(self, query: str, *, top_k: int = 5, **kwargs: Any) -> list[Document]:
        results = await self.retrieve_many([query], top_k=top_k, **kwargs)
        return results[0]

    async def retrieve_many(
        self,
        queries: Sequence[str],
        *,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
        include_vectors: bool = False,
    ) -> list[list[Document]]:
        top_k = _validate_top_k(top_k)
        if not queries:
            return []
        vectors = validate_embeddings(await self.embedder.embed_queries(queries), expected_count=len(queries))
        results = await self.store.search_vector(
            vectors,
            top_k=top_k,
            filters=filters,
            include_vectors=include_vectors,
        )
        return _validate_result_groups(results, expected_count=len(queries))


class HybridRetriever:
    def __init__(self, store: DocumentStore, embedder: Embedder, *, alpha: float = 0.5):
        _validate_alpha(alpha)
        self.store = store
        self.embedder = embedder
        self.alpha = alpha

    async def retrieve(self, query: str, *, top_k: int = 5, **kwargs: Any) -> list[Document]:
        results = await self.retrieve_many([query], top_k=top_k, **kwargs)
        return results[0]

    async def retrieve_many(
        self,
        queries: Sequence[str],
        *,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
        include_vectors: bool = False,
        alpha: float | None = None,
    ) -> list[list[Document]]:
        top_k = _validate_top_k(top_k)
        selected_alpha = self.alpha if alpha is None else alpha
        _validate_alpha(selected_alpha)
        if not queries:
            return []
        vectors = validate_embeddings(await self.embedder.embed_queries(queries), expected_count=len(queries))
        results = await self.store.search_hybrid(
            queries,
            vectors,
            alpha=selected_alpha,
            top_k=top_k,
            filters=filters,
            include_vectors=include_vectors,
        )
        return _validate_result_groups(results, expected_count=len(queries))


__all__ = ["HybridRetriever", "KeywordRetriever", "VectorRetriever"]
