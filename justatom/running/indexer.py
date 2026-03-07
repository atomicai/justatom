import asyncio as asio
import torch
from collections.abc import Iterable
from loguru import logger
from more_itertools import chunked
from tqdm.asyncio import tqdm_asyncio

from justatom.etc.errors import DocumentStoreError
from justatom.etc.schema import Document
from justatom.processing import igniset
from justatom.processing.loader import NamedDataLoader
from justatom.processing.mask import IProcessor
from justatom.running.encoders import EncoderRunner, BiEncoderRunner
from justatom.running.mask import IIndexerRunner
from justatom.storing.mask import INNDocStore


class NNIndexer(IIndexerRunner):
    def __init__(
        self,
        store: INNDocStore,
        runner: EncoderRunner | BiEncoderRunner,
        processor: IProcessor,
        device: str = "cpu",
    ):
        self.store = store
        self.runner = runner.eval()
        self.processor = processor
        self.device = device
        if runner.device != device:
            logger.info(
                f"Moving [{runner.__class__.__name__}] to the new device = {device}. Old device = {runner.device}"
            )
        self.runner.to(device)

    def _runner(
        self,
        documents: Iterable[dict | Document],
        batch_size: int = 512,
        device: str = None,
        flush_memory_every: int = 32,
        **props,
    ):
        device = device or self.device
        if device != self.device:
            logger.info(
                f"Moving [{self.runner.__class__.__name__}] to the new device = {device}. Old device = {self.runner.device}"
            )
            self.runner.to(device)
            self.runner = self.runner.eval()
        for i, docs_chunk in enumerate(chunked(documents, n=batch_size)):
            documents_as_dicts = [
                d.to_dict() if isinstance(d, Document) else d for d in docs_chunk
            ]
            if len(documents_as_dicts) == 0:
                continue

            dataset, tensor_names = igniset(
                dicts=documents_as_dicts,
                processor=self.processor,
                batch_size=batch_size,
            )
            loader = NamedDataLoader(
                dataset=dataset,
                tensor_names=tensor_names,
                batch_size=batch_size,
            )

            for docs, batch in zip(
                chunked(documents_as_dicts, n=batch_size), loader, strict=False
            ):
                batches = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    vectors = self.runner(batch=batches)[0].cpu().numpy()
                for doc, vec in zip(docs, vectors, strict=False):
                    doc["embedding"] = vec
                if i % flush_memory_every:
                    if device == "mps" and torch.mps.is_available():
                        torch.mps.empty_cache()
                    elif device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                yield docs

    async def _write_batch(
        self,
        batch_idx: int,
        docs_with_embeddings_batch: list[dict],
        batch_size_per_request: int,
    ) -> int:
        try:
            return await self.store.write_documents(
                [Document.from_dict(doc) for doc in docs_with_embeddings_batch],
                batch_size=batch_size_per_request,
            )
        except DocumentStoreError as error:
            msg = (
                f"{self.__class__.__name__} error writing docs on batch_idx={batch_idx}. "
                f"batch_size={len(docs_with_embeddings_batch)}"
            )
            raise DocumentStoreError(msg) from error

    @torch.no_grad()
    async def index(
        self,
        documents: Iterable[dict | Document],
        batch_size: int = 512,
        batch_size_per_request: int = 64,
        max_parallel_requests: int = 1,
        device: str = None,
        flush_memory_every: int = 10,
        **props,
    ):
        docs_with_embeddings = self._runner(
            documents, batch_size, device, flush_memory_every
        )
        max_parallel_requests = max(1, int(max_parallel_requests))

        n_total_written_docs: int = 0
        pending: set[asio.Task] = set()

        for batch_idx, docs_with_embeddings_batch in tqdm_asyncio(
            enumerate(docs_with_embeddings)
        ):
            pending.add(
                asio.create_task(
                    self._write_batch(
                        batch_idx=batch_idx,
                        docs_with_embeddings_batch=docs_with_embeddings_batch,
                        batch_size_per_request=batch_size_per_request,
                    )
                )
            )

            if len(pending) >= max_parallel_requests:
                done, pending = await asio.wait(
                    pending,
                    return_when=asio.FIRST_COMPLETED,
                )
                for task in done:
                    n_total_written_docs += task.result()

        if pending:
            done, _ = await asio.wait(pending)
            for task in done:
                n_total_written_docs += task.result()

        return n_total_written_docs


class KWARGIndexer(IIndexerRunner):
    def __init__(self, store: INNDocStore):
        self.store = store

    async def index(
        self,
        documents: Iterable[str | dict],
        batch_size: int = 512,
        batch_size_per_request: int = 64,
        **props,
    ):
        n_total_written_docs: int = 0
        for batch_idx, docs_batch in enumerate(chunked(documents, n=batch_size)):
            try:
                cur_written_docs = await self.store.write_documents(
                    [
                        (
                            Document.from_dict(content=doc)
                            if isinstance(doc, str)
                            else Document.from_dict(doc)
                        )
                        for doc in docs_batch
                    ],
                    batch_size=batch_size_per_request,
                )
            except DocumentStoreError as error:
                raise Exception from error(
                    f"""
                    {self.__class__.__name__} Error writing docs on batch_idx {batch_idx}. Total written docs {n_total_written_docs}
                    """
                )
            else:
                n_total_written_docs += cur_written_docs
        return n_total_written_docs


class ByName:

    OPS = ["keywords", "embedding", "hybrid", "gamma-hybrid"]

    def named(self, name: str, **kwargs):

        if name == "keywords":
            klass = KWARGIndexer
        elif name in ["embedding", "hybrid", "gamma-hybrid"]:
            klass = NNIndexer
        else:
            msg = (
                f"Unknown name=[{name}] to init IIndexerRunner instance. "
                f"Use one of {', '.join(self.OPS)}"
            )
            logger.error(msg)
            raise ValueError(msg)

        return klass(**kwargs)


API = ByName()


__all__ = ["KWARGIndexer", "NNIndexer", "API"]
