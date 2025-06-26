import asyncio as asio
import copy

import torch
from loguru import logger
from more_itertools import chunked
from tqdm.autonotebook import tqdm

from justatom.etc.errors import DocumentStoreError
from justatom.etc.pattern import singleton
from justatom.etc.schema import Document
from justatom.processing import igniset
from justatom.processing.loader import NamedDataLoader
from justatom.processing.mask import IProcessor
from justatom.running.m1 import M1LMRunner
from justatom.running.m2 import M2LMRunner
from justatom.running.mask import IIndexerRunner
from justatom.storing.mask import INNDocStore


class NNIndexer(IIndexerRunner):
    def __init__(
        self,
        store: INNDocStore,
        runner: M1LMRunner | M2LMRunner,
        processor: IProcessor,
        device: str = "cpu",
    ):
        self.store = store
        self.runner = runner.eval()
        self.processor = processor
        self.device = device
        if runner.device != device:
            logger.info(f"Moving [{runner.__class__.__name__}] to the new device = {device}. Old device = {runner.device}")
        self.runner.to(device)

    def _pipeline(
        self,
        documents: list[dict | Document],
        batch_size: int = 512,
        device: str = None,
        flush_memory_every: int = 10,
        **props,
    ):
        device = device or self.device
        if device != self.device:
            logger.info(
                f"Moving [{self.runner.__class__.__name__}] to the new device = {device}. Old device = {self.runner.device}"
            )
            self.runner.to(device)
            self.runner = self.runner.eval()
        documents_as_dicts = [d.to_dict() if isinstance(d, Document) else d for d in documents]
        dataset, tensor_names = igniset(dicts=documents_as_dicts, processor=self.processor, batch_size=batch_size)
        loader = NamedDataLoader(dataset=dataset, tensor_names=tensor_names, batch_size=batch_size)

        for i, (docs, batch) in tqdm(enumerate(zip(chunked(documents_as_dicts, n=batch_size), loader, strict=False))):  # noqa: B007
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

    @torch.no_grad()
    async def index(
        self,
        documents: list[dict | Document],
        batch_size: int = 512,
        batch_size_per_request: int = 64,
        device: str = None,
        flush_memory_every: int = 10,
        **props,
    ):
        docs_with_embeddings = self._pipeline(documents, batch_size, device, flush_memory_every)
        n_total_written_docs: int = 0
        for batch_idx, docs_with_embeddings_batch in enumerate(docs_with_embeddings):
            try:
                cur_written_docs = await self.store.write_documents(
                    [Document.from_dict(doc) for doc in docs_with_embeddings_batch], batch_size=batch_size_per_request
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

        @torch.no_grad()
        def encode(
            self,
            documents: list[dict | Document],
            batch_size: int = 512,
            batch_size_per_request: int = 64,
            device: str = None,
            flush_memory_every: int = 10,
            **props,
        ):
            docs_with_embeddings = list(self._pipeline(documents, batch_size, device, flush_memory_every))
            return docs_with_embeddings


class KWARGIndexer(IIndexerRunner):
    def __init__(self, store: INNDocStore):
        self.store = store

    async def index(self, documents: list[str | dict], batch_size: int = 512, batch_size_per_request: int = 64, **props):
        n_total_written_docs: int = 0
        for batch_idx, docs_batch in enumerate(chunked(documents, n=batch_size)):
            try:
                cur_written_docs = await self.store.write_documents(
                    [Document.from_dict(content=doc) if isinstance(doc, str) else Document.from_dict(doc) for doc in docs_batch],
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
    def named(self, name: str, **kwargs):
        OPS = ["keywords", "emebedding", "atomicai"]

        if name == "keywords":
            klass = KWARGIndexer
        elif name in ["embedding", "atomicai"]:
            klass = NNIndexer
        else:
            msg = f"Unknown name=[{name}] to init IIndexerRunner instance. Use one of {', '.join(OPS)}"
            logger.error(msg)
            raise ValueError(msg)

        return klass(**kwargs)


API = ByName()


__all__ = ["KWARGIndexer", "NNIndexer", "API"]
