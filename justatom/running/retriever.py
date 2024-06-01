from justatom.running.mask import IRetrieverRunner
from justatom.running.atomic import ATOMICLMRunner
from justatom.running.m1 import M1LMRunner
from justatom.running.m2 import M2LMRunner
from typing import List, Union
from loguru import logger
from justatom.etc.pattern import singleton
from justatom.storing.mask import INNDocStore


class AtomicRetriever(IRetrieverRunner):

    def __init__(self, store: INNDocStore, model: ATOMICLMRunner):
        super().__init__()
        self.store = store
        self.model = model.eval()

    def retrieve_topk(self, queries: List[str], top_k: int = 5):
        pass


class EmbeddingRetriever(IRetrieverRunner):

    def __init__(self, store: INNDocStore, model: Union[M1LMRunner, M2LMRunner]):
        super().__init__()
        self.store = store
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
            answers.append([res.content for res in response])
        return answers


@singleton
class IFinder:

    def find(self, name: str, **kwargs) -> IRetrieverRunner:
        if name == "keywords":
            return KWARGRetriever(**kwargs)
        elif name == "embedding":
            return EmbeddingRetriever(**kwargs)
        elif name == "justatom":
            return AtomicRetriever(**kwargs)
        else:
            ops = ["keywords", "embedding", "justatom"]
            msg = f"Unexpected name [{name}] passed down to {self.__name__} call. Please use one of [{','.join(ops)}]"
            logger.error(msg)
            raise ValueError(msg)


Finder = IFinder()


__all__ = ["KWARGRetriever", "EmbeddingRetriever", "Finder"]
