from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from typing import Any

from justatom.etc.schema import Document
from justatom.retrieval.contracts import DocumentStore, Embedder, SearchMode
from justatom.retrieval.errors import ConfigurationError
from justatom.retrieval.indexer import Indexer
from justatom.retrieval.retriever import HybridRetriever, KeywordRetriever, VectorRetriever


class RetrievalRuntime:
    def __init__(
        self,
        store: DocumentStore,
        embedder: Embedder | None,
        mode: SearchMode | str,
        alpha: float = 0.5,
    ) -> None:
        try:
            self.mode = mode if isinstance(mode, SearchMode) else SearchMode(mode)
        except ValueError as error:
            raise ConfigurationError(f"Unsupported retrieval mode: {mode!r}") from error
        if self.mode is not SearchMode.KEYWORD and embedder is None:
            raise ConfigurationError(f"{self.mode.value} retrieval requires an embedder")

        self.store = store
        self.embedder = embedder
        self.indexer = Indexer(store, embedder)
        match self.mode:
            case SearchMode.KEYWORD:
                self.retriever = KeywordRetriever(store)
            case SearchMode.VECTOR:
                self.retriever = VectorRetriever(store, embedder)
            case SearchMode.HYBRID:
                self.retriever = HybridRetriever(store, embedder, alpha=alpha)

        self._lifecycle_lock = asyncio.Lock()
        self._idle = asyncio.Event()
        self._idle.set()
        self._active_operations = 0
        self._closing = False
        self._closed = False
        self._finalization_task: asyncio.Task[None] | None = None

    async def index(
        self,
        documents: Iterable[Document | dict[str, Any]],
        batch_size: int = 64,
        max_parallel_writes: int = 1,
    ) -> int:
        return await self._delegate(
            self.indexer.index(
                documents,
                batch_size=batch_size,
                max_parallel_writes=max_parallel_writes,
            )
        )

    async def retrieve(self, query: str, *, top_k: int = 5, **kwargs: Any) -> list[Document]:
        return await self._delegate(self.retriever.retrieve(query, top_k=top_k, **kwargs))

    async def retrieve_many(
        self,
        queries: Sequence[str],
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[list[Document]]:
        return await self._delegate(self.retriever.retrieve_many(queries, top_k=top_k, **kwargs))

    async def __aenter__(self) -> RetrievalRuntime:
        async with self._lifecycle_lock:
            if self._closing or self._closed:
                raise RuntimeError("RetrievalRuntime is closed")
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            if self._finalization_task is None:
                self._closing = True
                self._finalization_task = asyncio.create_task(self._finalize_close())
            finalization_task = self._finalization_task

        await asyncio.shield(finalization_task)

    async def _delegate(self, operation: Any) -> Any:
        async with self._lifecycle_lock:
            if self._closing or self._closed:
                operation.close()
                raise RuntimeError("RetrievalRuntime is closed")
            self._active_operations += 1
            self._idle.clear()

        try:
            return await operation
        finally:
            async with self._lifecycle_lock:
                self._active_operations -= 1
                if self._active_operations == 0:
                    self._idle.set()

    async def _finalize_close(self) -> None:
        await self._idle.wait()
        first_error: BaseException | None = None
        later_error: BaseException | None = None

        try:
            if self.embedder is not None:
                try:
                    await self.embedder.close()
                except BaseException as error:
                    first_error = error
            try:
                await self.store.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
                else:
                    later_error = error
        finally:
            async with self._lifecycle_lock:
                self._closed = True

        if first_error is not None:
            if later_error is not None:
                raise first_error from later_error
            raise first_error


__all__ = ["RetrievalRuntime"]
