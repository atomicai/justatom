from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import Any

from justatom.etc.errors import DocumentStoreError
from justatom.etc.schema import Document
from justatom.retrieval.contracts import DocumentStore, Embedder, validate_embeddings
from justatom.retrieval.errors import ConfigurationError


def _batches(values: Iterable[Document | dict[str, Any]], batch_size: int) -> Iterator[list[Document | dict[str, Any]]]:
    iterator = iter(values)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def _to_document(value: Document | dict[str, Any]) -> Document:
    if isinstance(value, Document):
        return Document.from_dict(value.to_dict())
    if isinstance(value, dict):
        return Document.from_dict(value)
    raise TypeError(f"Expected Document or dict, received {type(value).__name__}")


def _validate_positive_int(value: object, parameter: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{parameter} must be a positive integer")
    return value


class Indexer:
    def __init__(self, store: DocumentStore, embedder: Embedder | None = None):
        self.store = store
        self.embedder = embedder

    async def index(
        self,
        documents: Iterable[Document | dict[str, Any]],
        batch_size: int = 64,
        max_parallel_writes: int = 1,
    ) -> int:
        batch_size = _validate_positive_int(batch_size, "batch_size")
        max_parallel_writes = _validate_positive_int(max_parallel_writes, "max_parallel_writes")
        semaphore = asyncio.Semaphore(max_parallel_writes) if max_parallel_writes > 1 else None
        pending: dict[asyncio.Task[int], int] = {}
        written = 0

        try:
            for batch_index, raw_batch in enumerate(_batches(documents, batch_size)):
                batch = [_to_document(value) for value in raw_batch]
                if self.embedder is not None:
                    embeddings = validate_embeddings(
                        await self.embedder.embed_documents([document.content for document in batch]),
                        expected_count=len(batch),
                    )
                    for document, embedding in zip(batch, embeddings, strict=True):
                        document.embedding = embedding
                else:
                    for document in batch:
                        document.embedding = None

                task = asyncio.create_task(self._write_batch(batch_index, batch, semaphore))
                pending[task] = batch_index
                if len(pending) >= max_parallel_writes:
                    written += await self._collect_completed_write(pending)

            while pending:
                written += await self._collect_completed_write(pending)
        except BaseException:
            await self._cancel_pending_writes(pending)
            raise

        return written

    async def _write_batch(
        self,
        batch_index: int,
        batch: list[Document],
        semaphore: asyncio.Semaphore | None,
    ) -> int:
        try:
            if semaphore is None:
                return await self.store.write_documents(batch)
            async with semaphore:
                return await self.store.write_documents(batch)
        except Exception as error:
            raise DocumentStoreError(f"Failed to write batch {batch_index}") from error

    @staticmethod
    async def _collect_completed_write(pending: dict[asyncio.Task[int], int]) -> int:
        done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        completed = sorted(((pending.pop(task), task) for task in done), key=lambda item: item[0])
        failures = [(batch_index, task) for batch_index, task in completed if not task.cancelled() and task.exception()]
        if failures:
            _, task = failures[0]
            return task.result()
        for _, task in completed:
            if task.cancelled():
                raise asyncio.CancelledError
        return sum(task.result() for _, task in completed)

    @staticmethod
    async def _cancel_pending_writes(pending: dict[asyncio.Task[int], int]) -> None:
        tasks = list(pending)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        pending.clear()


__all__ = ["Indexer"]
