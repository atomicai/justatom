import copy

import torch
from loguru import logger
from more_itertools import chunked
from tqdm.autonotebook import tqdm

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

    @torch.no_grad()
    async def index(self, documents: list[dict | Document], batch_size: int = 1, device: str = None, **props):
        device = device or self.device
        if device != self.device:
            logger.info(
                f"Moving [{self.runner.__class__.__name__}] to the new device = {device}. Old device = {self.runner.device}"
            )
            self.runner.to(device)
        documents_as_dicts = [d.to_dict() if isinstance(d, Document) else d for d in documents]
        dataset, tensor_names = igniset(dicts=documents_as_dicts, processor=self.processor, batch_size=batch_size)
        loader = NamedDataLoader(dataset=dataset, tensor_names=tensor_names, batch_size=batch_size)
        for i, (docs, batch) in tqdm(enumerate(zip(chunked(documents_as_dicts, n=batch_size), loader, strict=False))):  # noqa: B007
            batches = {k: v.to(device) for k, v in batch.items()}
            vectors = self.runner(batch=batches)[0].cpu()
            _docs = copy.deepcopy(docs)
            for doc, vec in zip(_docs, vectors, strict=False):
                doc["embedding"] = vec
            self.store.write_documents([Document.from_dict(doc) for doc in _docs])


class KWARGIndexer(IIndexerRunner):
    def __init__(self, store: INNDocStore):
        self.store = store

    async def index(self, documents: list[str | dict], batch_size: int = 4, **props):
        for i, chunk in enumerate(
            tqdm(
                chunked(documents, n=batch_size),
            )
        ):
            docs = [(Document.from_dict(dict(content=ci)) if isinstance(ci, str) else Document.from_dict(ci)) for ci in chunk]
            self.store.write_documents(docs)
            logger.info(f"{self.__class__.__name__} - index - {i / len(documents)}")


@singleton
class ByName:
    def named(self, name: str, **kwargs):
        OPS = ["keywords", "emebedding", "hybrid", "fusion", "atomicai", "justatom"]

        if name == "keywords":
            klass = KWARGIndexer
        elif name in ["embedding", "hybrid", "fusion", "atomicai", "justatom"]:
            klass = NNIndexer
        else:
            msg = f"Unknown name=[{name}] to init IIndexerRunner instance. Use one of {','.join(OPS)}"
            logger.error(msg)
            raise ValueError(msg)

        return klass(**kwargs)


API = ByName()


__all__ = ["KWARGIndexer", "NNIndexer", "API"]
