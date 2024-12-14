import copy
import math
import string
from collections import Counter
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
from justatom.tooling import stl


class ATOMICRetriever(IRetrieverRunner):
    def _compute_recall(query: str, keywords_or_phrases: list[str], **props):
        k_words = Counter(stl.flatten_list([kwp.lower().split(" ") for kwp in keywords_or_phrases]))
        q_words = "".join(w for w in query if w not in string.punctuation).lower().strip().split()
        recall = sum([1.0 / math.log(1 + k_words.get(w, 1)) for w in q_words])

        return recall

    def _compute_inverse_recall(query: str, keywords_or_phrases: list[str], **props):
        k_words = Counter(stl.flatten_list([kwp.lower().split(" ") for kwp in keywords_or_phrases]))
        q_words = "".join(w for w in query if w not in string.punctuation).lower().strip().split()
        idf_recall = sum([1.0 / math.log(1 + k_words.get(w, 1)) for w in q_words]) / sum(
            [1.0 / math.log(1 + k_words.get(w, 1)) for w in q_words]
        )
        return idf_recall

    RANKER: dict = {"JaroWinkler": jarowinkler_similarity, "Recall": _compute_recall, "IDFRecall": _compute_inverse_recall}

    def __init__(
        self,
        store: INNDocStore,
        runner: M1LMRunner | M2LMRunner,
        processor: IProcessor,
        ranker: str | None = "IDFRecall",
        device: str = "cpu",
        alpha: float = 0.78,
        top_p: int = 128,
        prefix: str = None,
        cutoff_score: float = 0.8,
        gamma: float = 0.5,
        include_keywords: bool = True,
        include_explanation: bool = False,
    ):
        super().__init__()
        self.store = store
        self.processor = processor
        self.runner = runner.eval()
        self.device = device
        if runner.device != device:
            logger.info(f"Moving [{runner.__class__.__name__}] to the new device = {device}. Old device = {runner.device}")

        self.runner.to(device)
        if ranker not in self.RANKER:
            msg = f"Ranker=[{str(ranker)}] is NOT supported. Use one of the following options: {','.join(self.RANKER.keys())}"
            logger.error(msg)
            raise ValueError(msg)
        self.ranker = self.RANKER[ranker]

        self.alpha = alpha
        self.top_p = top_p
        self.prefix = prefix
        self.cutoff_score = cutoff_score
        self.gamma = gamma
        self.include_keywords = include_keywords
        self.include_explanation = include_explanation

    def compute_fusion_score(self, distance: float, keyword_score: float, gamma: float = 0.5) -> float:
        """
        distance: Semantic score from ANN search (e.g. Weaviate)
        keyword_score: Keyword intersection, Precision, Recall in terms of keywords intersection
        gamma: Weight parameter to include to
        """

        # Semantic score conversion
        semantic_relevance = distance  # значения ~ от (0,1], ближе = выше скор.

        # Final score
        combined_score = gamma * semantic_relevance + (1 - gamma) * keyword_score

        return combined_score

    @torch.no_grad()
    def retrieve_topk(
        self,
        queries: str | list[str],
        top_k: int = 5,
        top_p: int = None,
        prefix: str = None,
        include_embedding: bool = False,
        alpha: float = None,
        include_scores: bool = False,
        include_keywords: bool = None,
        include_explanation: bool = None,
        batch_size: int = 16,
        filters: dict | None = None,
        cutoff_score: float = None,
        gamma: float = None,
        return_keywords_or_phrases: bool = False,
        **props,
    ):
        alpha = alpha or self.alpha
        top_p = top_p or self.top_p
        prefix = prefix or self.prefix
        cutoff_score = cutoff_score or self.cutoff_score
        gamma = gamma or self.gamma
        include_keywords = include_keywords or self.include_keywords
        include_explanation = include_explanation or self.include_explanation
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
                        keywords_content = [
                            kwp["keyword_or_phrase"].strip() + " " + kwp["explanation"].strip() for kwp in keywords_or_phrases
                        ]
                    elif include_keywords:
                        keywords_content = [kwp["keyword_or_phrase"].strip() for kwp in keywords_or_phrases]
                    else:
                        keywords_content = [kwp["explanation"].strip() for kwp in keywords_or_phrases]

                    keyword_score: float = self.ranker(query, keywords_content, score_cutoff=cutoff_score)
                    semantic_score: float = doc.score
                    fusion_score: float = self.compute_fusion_score(
                        distance=semantic_score, keyword_score=keyword_score, gamma=gamma
                    )
                    fusion.append({"rank": i, "keywords_content": keywords_content, "fusion_score": fusion_score})
                fusion = sorted(
                    fusion, key=cmp_to_key(lambda obj1, obj2: obj1["fusion_score"] - obj2["fusion_score"]), reverse=True
                )
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
        alpha: float = 0.5,
        prefix: str = None,
    ):
        super().__init__()
        self.store = store
        self.processor = processor
        self.runner = runner.eval()
        self.device = device
        if runner.device != device:
            logger.info(f"Moving [{runner.__class__.__name__}] to the new device = {device}. Old device = {runner.device}")

        self.runner.to(device)
        self.alpha = alpha
        self.prefix = prefix

    @torch.no_grad()
    def retrieve_topk(
        self,
        queries: str | list[str],
        top_k: int = 5,
        prefix: str = None,
        include_embedding: bool = False,
        alpha: float = None,
        include_scores: bool = False,
        batch_size: int = 16,
        filters: dict | None = None,
    ):
        alpha = alpha or self.alpha
        prefix = prefix or self.prefix
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
        prefix: str = None,
    ):
        super().__init__()
        self.store = store
        self.processor = processor
        self.runner = runner.eval()
        self.device = device
        if runner.device != device:
            logger.info(f"Moving [{runner.__class__.__name__}] to the new device = {device}. Old device = {runner.device}")

        self.runner.to(device)
        self.prefix = prefix

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
        prefix = prefix or self.prefix
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
        OPS = ["keywords", "emebedding", "hybrid", "atomicai"]

        if name == "keywords":
            klass = KWARGRetriever
        elif name == "embedding":
            klass = EmbeddingRetriever
        elif name == "hybrid":
            klass = HybridRetriever
        elif name == "atomicai":
            klass = ATOMICRetriever
        else:
            msg = f"Unknown name=[{name}] to init IRetrieverRunner instance. Use one of {','.join(OPS)}"
            logger.error(msg)
            raise ValueError(msg)

        return klass(**kwargs)


API = ByName()


__all__ = ["KWARGRetriever", "HybridRetriever", "EmbeddingRetriever", "ATOMICRetriever", "API"]
