import torch
from loguru import logger
from more_itertools import chunked

from justatom.etc.pattern import singleton
from justatom.processing.loader import NamedDataLoader
from justatom.processing.mask import IProcessor
from justatom.processing.silo import igniset
from justatom.running.atomic import ATOMICLMRunner
from justatom.running.m1 import M1LMRunner
from justatom.running.m2 import M2LMRunner
from justatom.running.mask import IRetrieverRunner
from justatom.storing.mask import INNDocStore


class ATOMICRetriever(IRetrieverRunner):
    def __init__(self, store: INNDocStore, model: ATOMICLMRunner):
        super().__init__()
        self.store = store
        self.model = model.eval()

    def retrieve_topk(self, queries: list[str], top_k: int = 5):
        pass


class HybridRetriever(IRetrieverRunner):
    def __init__(
        self,
        store: INNDocStore,
        runner: M1LMRunner | M2LMRunner,
        processor: IProcessor,
        device="cpu",
    ):
        super().__init__()
        self.store = store
        self.processor = processor
        self.runner = runner.eval()
        self.device = device
        if runner.device != device:
            logger.info(
                f"Callback from {self.__class__.__name__} is fired to move to new device {device}. Old device = {runner.device}"
            )
            self.runner.to(device)

    @torch.no_grad()
    def retrieve_topk(
        self,
        queries: str | list[str],
        top_k: int = 5,
        prefix: str = None,
        include_embedding: bool = False,
        alpha: float = 0.85,
        include_scores: bool = False,
        batch_size: int = 16,
    ):
        queries = [queries] if isinstance(queries, str) else queries
        queries = [({"content": q} if prefix is None else {"content": q, "meta": {"prefix": prefix}}) for q in queries]
        dataset, tensor_names = igniset(queries, processor=self.processor, batch_size=batch_size)
        loader = NamedDataLoader(dataset, tensor_names=tensor_names, batch_size=batch_size)
        answer = []

        for _queries, _batches in zip(chunked(queries, n=batch_size), loader, strict=False):
            batches = {k: v.to(self.device) for k, v in _batches.items()}
            vectors = self.runner(batch=batches)[0].cpu().numpy().tolist()  # batch_size x vector_dim
            for vector, query in zip(vectors, _queries, strict=False):  # noqa: B007
                res_topk = self.store.search(vector, alpha=alpha, top_k=top_k)
                answer.append(res_topk)
        return answer


class EmbeddingRetriever(IRetrieverRunner):
    def __init__(
        self,
        store: INNDocStore,
        runner: M1LMRunner | M2LMRunner | ATOMICLMRunner,
        processor: IProcessor,
        device: str = "cpu",
    ):
        super().__init__()
        self.store = store
        self.processor = processor
        self.runner = runner.eval()
        self.device = device
        if runner.device != device:
            logger.info(
                f"Callback from {self.__class__.__name__} is fired to move to new device {device}. Old device = {runner.device}"
            )
            self.runner.to(device)

        self.runner.to(device)

    @torch.no_grad()
    def retrieve_topk(
        self,
        queries: str | list[str],
        top_k: int = 5,
        prefix: str = None,
        include_embedding: bool = False,
        include_scores: bool = False,
        batch_size: int = 16,
    ):
        queries = [queries] if isinstance(queries, str) else queries
        queries = [({"content": q} if prefix is None else {"content": q, "meta": {"prefix": prefix}}) for q in queries]
        dataset, tensor_names = igniset(queries, processor=self.processor, batch_size=batch_size)
        loader = NamedDataLoader(dataset, tensor_names=tensor_names, batch_size=batch_size)
        answer = []

        for _queries, _batches in zip(chunked(queries, n=batch_size), loader, strict=False):
            batches = {k: v.to(self.device) for k, v in _batches.items()}
            vectors = self.runner(batch=batches)[0].cpu().numpy().tolist()  # batch_size x vector_dim
            for vector, query in zip(vectors, _queries, strict=False):  # noqa: B007
                res_topk = self.store.search_by_embedding(vector, top_k=top_k)
                answer.append(res_topk)
        return answer


class KWARGRetriever(IRetrieverRunner):
    def __init__(self, store: INNDocStore):
        super().__init__()
        self.store = store

    def retrieve_topk(
        self,
        queries: str | list[str],
        top_k: int = 5,
        include_scores: bool = False,
    ):
        queries = [queries] if isinstance(queries, str) else queries
        answer = []
        for query in queries:
            response = self.store.search_by_keywords(query=query, top_k=top_k)
            answer.append(response)
        return answer


@singleton
class ByName:
    def named(self, name: str, **kwargs):
        OPS = ["keywords", "emebedding", "hybrid", "justatom"]

        if name == "keywords":
            klass = KWARGRetriever
        elif name == "embedding":
            klass = EmbeddingRetriever
        elif name == "hybrid":
            klass = HybridRetriever
        elif name == "justatom":
            klass = ATOMICRetriever
        else:
            msg = f"Unknown name=[{name}] to init IRetrieverRunner instance. Use one of {','.join(OPS)}"
            logger.error(msg)
            raise ValueError(msg)

        return klass(**kwargs)


API = ByName()


__all__ = ["KWARGRetriever", "HybridRetriever", "EmbeddingRetriever", "API"]
