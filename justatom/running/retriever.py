from justatom.running.mask import IRetrieverRunner
from justatom.running.atomic import ATOMICLMRunner
from justatom.running.m1 import M1LMRunner
from justatom.running.m2 import M2LMRunner
from justatom.processing.mask import IProcessor
from typing import List, Union
from loguru import logger
from justatom.etc.pattern import singleton
from justatom.storing.mask import INNDocStore


class ATOMICRetriever(IRetrieverRunner):

    def __init__(self, store: INNDocStore, model: ATOMICLMRunner):
        super().__init__()
        self.store = store
        self.model = model.eval()

    def retrieve_topk(self, queries: List[str], top_k: int = 5):
        pass


class EmbeddingRetriever(IRetrieverRunner):

    def __init__(self, store: INNDocStore, model: Union[M1LMRunner, M2LMRunner], processor: IProcessor):
        super().__init__()
        self.store = store
        self.processor = processor
        self.model = model.eval()

    def retrieve_topk(
        self,
        queries: Union[str, List[str]],
        top_k: int = 5,
        include_embedding: bool = False,
        include_scores: bool = False,
    ):
        pass


class KWARGRetriever(IRetrieverRunner):

    def __init__(self, store: INNDocStore):
        super().__init__()
        self.store = store

    def retrieve_topk(self, queries: Union[str, List[str]], top_k: int = 5, include_scores: bool = False):
        queries = [queries] if isinstance(queries, str) else queries
        answers = []
        for query in queries:
            response = self.store.search_by_keywords(query=query, top_k=top_k)
            answers.extend(response)
        return answers


@singleton
class ByName:

    def named(self, name: str, **kwargs):
        OPS = ["keywords", "emebedding", "justatom"]

        if name == "keywords":
            klass = KWARGRetriever
        elif name == "embedding":
            klass = EmbeddingRetriever
        elif name == "justatom":
            klass = ATOMICRetriever
        else:
            msg = f"Unknown name=[{name}] to init IRetrieverRunner instance. Use one of {','.join(OPS)}"
            logger.error(msg)
            raise ValueError(msg)

        return klass(**kwargs)


API = ByName()


__all__ = ["KWARGRetriever", "EmbeddingRetriever", "API"]
