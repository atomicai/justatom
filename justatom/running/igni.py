from pathlib import Path

from loguru import logger

from justatom.configuring.prime import Config
from justatom.etc.pattern import singleton
from justatom.modeling.prime import LMFinder
from justatom.processing.prime import INFERProcessor, ITokenizer
from justatom.running.indexer import API as IndexerApi
from justatom.running.indexer import IIndexerRunner, KWARGIndexer, NNIndexer
from justatom.running.m1 import M1LMRunner
from justatom.running.mask import IPatcherRunner
from justatom.running.patcher import PatcherRunner
from justatom.running.retriever import API as RetrieverApi
from justatom.running.retriever import ATOMICRetriever, EmbeddingRetriever, HybridRetriever, IRetrieverRunner, KWARGRetriever


@singleton
class IIGNIRunner:
    """
    This class is meant to be used in production while igniting different apis.
    """

    RETRIEVERS = {}  # TODO:  Integrate this to allow multiple retrievers. Might increase memory consumption
    INDEXERS = {}  # TODO: Allow different kind of models to be used in parallel without initializing delay

    def __init__(self, name: str = None):
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self._ix_runner = None
        self._ir_runner = None

    async def PATCHER(self, collection_name: str, new_collection_name: str) -> IPatcherRunner:
        """
        Asynchronous function to ignite `PATCHER` runner.
        `PATCHER` is responsible to re-write docs from one collection to the other.
        """

        patcher = PatcherRunner(collection_name=collection_name, new_collection_name=new_collection_name)
        return patcher

    async def INDEXER(
        self, store, index_by: str, model_name_or_path: str | None = None, prefix_to_use: str = None, device: str = "cpu", **props
    ) -> IIndexerRunner:
        """
        Asynchronous function to ignite INDEXER runner by caching the underlying model if one uses it.
        """
        MAPPING = dict(keywords=KWARGIndexer, embedding=NNIndexer)
        assert index_by in MAPPING, logger.error(
            f"""
            {self.__class__.__name__} |
            VAR search_by={index_by} is not supported. Please use one of the following: {', '.join(MAPPING.keys())}"
            """
        )
        if type(self._ix_runner) is not type(MAPPING.get(index_by)):
            if index_by == "keywords":
                self._ix_runner = IndexerApi.named(index_by, store=store)
            else:
                model_name_or_path = (
                    str(Path(Config.api["model_name_or_path"]).expanduser())
                    if model_name_or_path is None
                    else str(Path(model_name_or_path).expanduser())
                )
                prefix_to_use = "passage: " if prefix_to_use is None else prefix_to_use
                logger.info(
                    f"Creating new `IX` Runner instance. Might take a while. Loading model on {device} device and using prefix=[{prefix_to_use}]"  # noqa
                )
                tokenizer = ITokenizer.from_pretrained(model_name_or_path)
                processor = INFERProcessor(tokenizer=tokenizer, prefix=prefix_to_use)

                lm_model = LMFinder.find(model_name_or_path, device=device, **props)
                runner = M1LMRunner(model=lm_model, prediction_heads=[], device=device, processor=processor)

                self._ix_runner = IndexerApi.named(index_by, store=store, runner=runner, processor=processor, device=device)
        self._ix_runner.store = store
        return self._ix_runner

    async def RETRIEVER(
        self, store, search_by: str, model_name_or_path: str | None = None, prefix_to_use: str = None, device: str = "cpu", **props
    ) -> IRetrieverRunner:
        """
        Asynchronous function to ignite IR (aka Information Retrieval) runner by caching the underlying model if one uses it.
        """
        MAPPING = dict(keywords=KWARGRetriever, embedding=EmbeddingRetriever, hybrid=HybridRetriever, fusion=ATOMICRetriever)
        assert search_by in MAPPING, logger.error(
            f"""
            {self.__class__.__name__} |
            VAR search_by={search_by} is not supported. Please use one of the following: {', '.join(MAPPING.keys())}"
            """
        )
        if type(self._ir_runner) is not MAPPING[search_by]:
            if search_by == "keywords":
                self._ir_runner = RetrieverApi.named(search_by, store=store)
            else:
                model_name_or_path = (
                    str(Path(Config.api["model_name_or_path"]).expanduser())
                    if model_name_or_path is None
                    else str(Path(model_name_or_path).expanduser())
                )

                prefix_to_use = "query: " if prefix_to_use is None else prefix_to_use
                logger.info(
                    f"Creating new `IR` Runner instance. Might take a while. Loading model on {device} device and using prefix=[{prefix_to_use}]"  # noqa
                )

                tokenizer = ITokenizer.from_pretrained(model_name_or_path)
                processor = INFERProcessor(tokenizer=tokenizer, prefix=prefix_to_use)

                lm_model = LMFinder.find(model_name_or_path, device=device, **props)

                runner = M1LMRunner(model=lm_model, prediction_heads=[], device=device, processor=processor)

                self._ir_runner = RetrieverApi.named(search_by, store=store, runner=runner, processor=processor, device=device)
        self._ir_runner.store = store
        return self._ir_runner

    async def SERVER(self, **props):
        def callback(message, metadata):
            logger.info(message)
            logger.info(metadata)

        return callback


IGNIRunner = IIGNIRunner()


__all__ = ["IGNIRunner"]
