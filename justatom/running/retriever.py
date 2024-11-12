import copy
from functools import cmp_to_key

import torch
from jarowinkler import jarowinkler_similarity
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
    RANKER: dict = {"jarowinkler": jarowinkler_similarity}

    def __init__(
        self,
        store: INNDocStore,
        runner: M1LMRunner | M2LMRunner,
        processor: IProcessor,
        ranker: str | None = "jarowinkler",
        device: str = "cpu",
    ):
        super().__init__()
        self.store = store
        self.runner = runner.eval()
        self.device = device
        if runner.device != device:
            logger.info(
                f"Callback from {self.__class__.__name__} is fired to move to new device {device}. Old device = {runner.device}"
            )
            self.runner.to(device)
        if ranker not in self.RANKER:
            msg = f"Ranker=[{str(ranker)}] is NOT supported. Use one of the following options: {','.join(self.RANKER.keys())}"
            logger.error(msg)
            raise ValueError(msg)
        self.ranker = self.RANKER[ranker]

    @torch.no_grad()
    def retrieve_topk(
        self,
        queries: str | list[str],
        top_k: int = 5,
        top_p: int = 128,
        prefix: str = None,
        include_embedding: bool = False,
        alpha: float = 0.85,
        include_scores: bool = False,
        include_keywords: bool = True,
        include_explanation: bool = True,
        batch_size: int = 16,
        filters: dict | None = None,
        **props,
    ):
        if not include_keywords and not include_explanation:
            msg = f"""
            You've initialized `{self.__class__.__name__}` IR but not using any of the atomic keywords features.
            If you don't need or your dataset is free of keywords, please you one of the following: 
            {','.join(
                [
                    KWARGRetriever.__class__.__name__,
                    EmbeddingRetriever.__class__.__name__,
                    HybridRetriever.__class__.__name__
                ])
            }
            """
            logger.error(msg)
            raise ValueError(msg)
        queries = [queries] if isinstance(queries, str) else queries
        js_queries = [({"content": q} if prefix is None else {"content": q, "meta": {"prefix": prefix}}) for q in queries]
        dataset, tensor_names = igniset(js_queries, processor=self.processor, batch_size=batch_size)
        loader = NamedDataLoader(dataset, tensor_names=tensor_names, batch_size=batch_size)
        answer = []

        for _queries, _batches in zip(chunked(queries, n=batch_size), loader, strict=False):
            batches = {k: v.to(self.device) for k, v in _batches.items()}
            vectors = self.runner(batch=batches)[0].cpu().numpy().tolist()  # batch_size x vector_dim
            for vector, query in zip(vectors, _queries, strict=False):  # noqa: B007
                res_topk = []
                res_topp = self.store.search(query=query, query_embedding=vector, alpha=alpha, filters=filters, top_k=top_p)
                fusion = []
                for i, doc in enumerate(res_topp):
                    keywords_or_phrases = doc.meta.get("keywords_or_phrases", [])
                    keywords_content: str = None
                    if include_keywords and include_explanation:
                        keywords_content = "\n".join(
                            [kwp["keyword_or_phrase"].strip() + " - " + kwp["explanation"].strip() for kwp in keywords_or_phrases]
                        )
                    elif include_keywords:
                        keywords_content = "\n".join([kwp["keyword_or_phrase"].strip() for kwp in keywords_or_phrases])
                    else:
                        keywords_content = "\n".join([kwp["explanation"].strip() for kwp in keywords_or_phrases])

                    score: float = self.ranker(query, keywords_content)
                    fusion.append({"rank": i, "keywords_content": keywords_content, "kwp_score": score})
                    fusion = sorted(fusion, key=cmp_to_key(lambda obj1, obj2: obj1["kwp_score"] - obj2["kwp_score"]))

                res_topk = [res_topp[pos["rank"]] for pos in fusion[:top_k]]
                answer.append(copy.deepcopy(res_topk))
        return answer


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
        filters: dict | None = None,
    ):
        queries = [queries] if isinstance(queries, str) else queries
        js_queries = [({"content": q} if prefix is None else {"content": q, "meta": {"prefix": prefix}}) for q in queries]
        dataset, tensor_names = igniset(js_queries, processor=self.processor, batch_size=batch_size)
        loader = NamedDataLoader(dataset, tensor_names=tensor_names, batch_size=batch_size)
        answer = []

        for _queries, _batches in zip(chunked(queries, n=batch_size), loader, strict=False):
            batches = {k: v.to(self.device) for k, v in _batches.items()}
            vectors = self.runner(batch=batches)[0].cpu().numpy().tolist()  # batch_size x vector_dim
            for vector, query in zip(vectors, _queries, strict=False):  # noqa: B007
                res_topk = self.store.search(query=query, query_embedding=vector, alpha=alpha, top_k=top_k)
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
        filters: dict | None = None,
        keywords: list[str] | None = None,
    ):
        queries = [queries] if isinstance(queries, str) else queries
        js_queries = [({"content": q} if prefix is None else {"content": q, "meta": {"prefix": prefix}}) for q in queries]
        dataset, tensor_names = igniset(js_queries, processor=self.processor, batch_size=batch_size)
        loader = NamedDataLoader(dataset, tensor_names=tensor_names, batch_size=batch_size)
        answer = []

        for _queries, _batches in zip(chunked(queries, n=batch_size), loader, strict=False):
            batches = {k: v.to(self.device) for k, v in _batches.items()}
            vectors = self.runner(batch=batches)[0].cpu().numpy().tolist()  # batch_size x vector_dim
            for vector, query in zip(vectors, _queries, strict=False):  # noqa: B007
                res_topk = self.store.search_by_embedding(query_embedding=vector, top_k=top_k, filters=filters, keywords=keywords)
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
        filters: dict | None = None,
        keywords: list[str] | None = None,
    ):
        queries = [queries] if isinstance(queries, str) else queries
        answer = []
        for query in queries:
            response = self.store.search_by_keywords(query=query, top_k=top_k, filters=filters, keywords=keywords)
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
        elif name == "atomic":
            klass = ATOMICRetriever
        else:
            msg = f"Unknown name=[{name}] to init IRetrieverRunner instance. Use one of {','.join(OPS)}"
            logger.error(msg)
            raise ValueError(msg)

        return klass(**kwargs)


API = ByName()


__all__ = ["KWARGRetriever", "HybridRetriever", "EmbeddingRetriever", "ATOMICRetriever", "API"]
