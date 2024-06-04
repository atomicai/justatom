from loguru import logger
from justatom.processing.mask import IProcessor
from justatom.storing.mask import INNDocStore
from justatom.configuring import Config
from justatom.etc.schema import Document
from more_itertools import chunked
from justatom.processing import igniset, loader
from justatom.running.mask import IIndexerRunner
from justatom.running.m1 import M1LMRunner
from justatom.running.m2 import M2LMRunner
from justatom.running.atomic import ATOMICLMRunner
from tqdm.autonotebook import tqdm
import torch
from justatom.etc.pattern import singleton
from typing import List, Dict, Union, Optional, Generator


class EmbeddingIndexer(IIndexerRunner):

    def __init__(self, store: INNDocStore, model: Union[M1LMRunner, M2LMRunner], processor: IProcessor):
        self.store = store
        self.similarity = store.similarity or ""
        self.model = model.eval()
        self.processor = processor

    async def index(self, documents: List[str], batch_size: int = None, log_every_step: int = 10):
        batch_size = batch_size or Config.index.batch_size
        iter_docs_generator = self.index_docs_generator(documents=documents, batch_size=batch_size, with_docs=True)

        for i, (embeddings, docs) in tqdm(enumerate(iter_docs_generator, desc=f" {self.__class__.__name__} progress")):
            docs = [
                Document.from_dict({"content": doc["content"], "embedding": emb}) for doc, emb in zip(docs, embeddings)
            ]
            try:
                self.store.write_documents(docs)
            except:
                logger.info(f"Error indexing @ {str(i)}-th position. Check the `weaviate` service logs for more info.")
                return False
            else:
                if i % log_every_step == 0:
                    logger.info(f"INDEX ok {str(i)} docs")

    def index_docs_generator(self, documents: List[str], batch_size: int = 1, with_docs: bool = False) -> Generator:
        batch_size = batch_size or Config.index.batch_size
        dataset, tensor_names = igniset(documents, processor=self.processor, batch_size=batch_size)
        _loader = loader.NamedDataLoader(dataset, tensor_names=tensor_names, batch_size=batch_size)
        # desc=f" {self.__class__.__name__} progress")
        for i, (batch, docs) in enumerate(zip(_loader, chunked(documents, n=batch_size))):
            with torch.no_grad():
                embeddings = self.model(**batch, average=True).cpu().numpy()
            if with_docs:
                yield (embeddings, docs)
            else:
                yield embeddings

    def ignite_docs(
        self, documents: List[str], batch_size: int = 1, with_docs: bool = True, schema: Optional[Dict] = None
    ):
        batch_size = batch_size or Config.index.batch_size
        schema = schema or dict(text="content")
        documents = list(check_schema(documents))  # List of dicts with `text` field
        dataset, tensor_names = igniset(documents, processor=self.processor, batch_size=batch_size)

        _loader = loader.NamedDataLoader(dataset, tensor_names=tensor_names, batch_size=batch_size)

        ignited_result = []

        for i, batch in enumerate(tqdm(_loader, desc=f" {self.__class__.__name__} progress")):
            with torch.no_grad():
                embeddings = self.model(**batch, average=True).cpu().numpy()
            igni = [
                Document.from_dict({"text": doc["text"], "embedding": emb}, field_map=schema)
                for doc, emb in zip(documents[i * batch_size : (i + 1) * batch_size], embeddings)
            ]
            ignited_result.extend(igni)
        return ignited_result

    def evaluate(
        self,
        documents: List[str],
        batch_size: int = None,
        filters: Optional[Dict] = None,
        eval_save_dir: Optional[str] = None,
    ):
        """
        Perform `search_by_embedding` using components provided in `__init__` document store
        using provided `documents` as query_key against which we would like to evaluate.
        """
        iterator = self.retrieve_embeddings(documents=documents, batch_size=batch_size, with_docs=True)
        response = []
        for i, (embeddings, docs) in enumerate(iterator):
            logger.info(f"Done {str(i)} processing batch")
            chunked_response = []
            for emb, doc in zip(embeddings, docs):
                topk_docs = self.store.search_by_embedding(emb, top_k=1_000)
                if len(topk_docs) > 0:
                    if filters is not None:
                        _topk_docs = [
                            d for d in topk_docs if filters.get(d.content, "") == filters.get(doc.get("text", ""))
                        ]
                        if len(_topk_docs) < 4:
                            _topk_docs = _topk_docs + topk_docs
                    else:
                        _topk_docs = topk_docs
                    current_eval_sample = dict(
                        query=doc.get("text"), r1=None, s1=None, r2=None, s2=None, r3=None, s3=None, r4=None, s4=None
                    )
                    for pos, top_doc in enumerate(_topk_docs[:4]):
                        rk = f"r{str(pos + 1)}"
                        sk = f"s{str(pos + 1)}"
                        current_eval_sample[rk] = top_doc.content
                        current_eval_sample[sk] = top_doc.score
                    chunked_response.append(current_eval_sample)
            response.extend(chunked_response)
        evalshot(response, name=f"evalreport_{self.similarity}")


class KWARGIndexer(IIndexerRunner):
    def __init__(self, store: INNDocStore):
        self.store = store

    async def index(self, documents: List[Union[str, Dict]], batch_size: int = 4):
        for i, chunk in enumerate(
            tqdm(
                chunked(documents, n=batch_size),
            )
        ):
            docs = [
                Document.from_dict(dict(content=ci)) if isinstance(ci, str) else Document.from_dict(ci) for ci in chunk
            ]
            self.store.write_documents(docs)
            logger.info(f"{self.__class__.__name__} - index - {i / len(documents)}")


class ATOMICIndexer(IIndexerRunner):
    def __init__(self, store: INNDocStore, model: ATOMICLMRunner):
        self.store = store


@singleton
class ByName:

    def named(self, name: str, **kwargs):
        OPS = ["keywords", "emebedding", "justatom"]

        if name == "keywords":
            klass = KWARGIndexer
        elif name == "embedding":
            klass = EmbeddingIndexer
        elif name == "justatom":
            klass = ATOMICIndexer
        else:
            msg = f"Unknown name=[{name}] to init IIndexerRunner instance. Use one of {','.join(OPS)}"
            logger.error(msg)
            raise ValueError(msg)

        return klass(**kwargs)


API = ByName()


__all__ = ["KWARGIndexer", "EmbeddingIndexer", "API"]
