from pathlib import Path

from loguru import logger

from justatom.configuring.prime import Config
from justatom.etc.pattern import singleton
from justatom.modeling.prime import LMFinder
from justatom.processing.prime import INFERProcessor, ITokenizer
from justatom.running.indexer import API as IndexerApi
from justatom.running.indexer import IIndexerRunner
from justatom.running.m1 import M1LMRunner
from justatom.running.retriever import API as RetrieverApi
from justatom.running.retriever import IRetrieverRunner
from justatom.tooling.hardware import initialize_device_settings


@singleton
class IIGNIRunner:
    """
    This class is meant to be used in production while igniting different apis.
    """

    def __init__(self, name: str = None):
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self._ix_runner = None
        self._ir_runner = None

    async def INDEXER(
        self, store, index_by: str, model_name_or_path: str | None = None, prefix_to_use: str = None, device: str = "cpu", **props
    ) -> IIndexerRunner:
        """
        Asynchronous function to ignite IX (aka IndeXer) runner by caching the underlying model if one uses it.
        """
        if self._ix_runner is None:
            if index_by == "keywords":
                self._ix_runner = IndexerApi.named(index_by, store=store)
            else:
                model_name_or_path = (
                    str(Path(Config.api.model_name_or_path).expanduser())
                    if model_name_or_path is None
                    else str(Path(model_name_or_path).expanduser())
                )
                prefix_to_use = str(Config.api.model_prefix_content_default) if prefix_to_use is None else str(prefix_to_use)
                logger.info(
                    f"Creating new `IX` Runner instance. Might take a while. Loading model on {device} device and using prefix \
                    {prefix_to_use}"
                )
                tokenizer = ITokenizer.from_pretrained(model_name_or_path)
                processor = INFERProcessor(tokenizer=tokenizer, prefix=prefix_to_use)

                lm_model = LMFinder.find(model_name_or_path, device=device, **props)
                runner = M1LMRunner(model=lm_model, prediction_heads=[], device=device, processor=processor)

                self._ix_runner = IndexerApi.named(index_by, store=store, runner=runner, processor=processor, device=device)
        return self._ix_runner

    async def RETRIEVER(
        self, store, search_by: str, model_name_or_path: str | None = None, prefix_to_use: str = None, device: str = "cpu", **props
    ) -> IRetrieverRunner:
        """
        Asynchronous function to ignite IR (aka Information Retrieval) runner by caching the underlying model if one uses it.
        """
        if self._ir_runner is None:
            if search_by == "keywords":
                self._ir_runner = RetrieverApi.named(search_by, store=store)
            else:
                model_name_or_path = (
                    str(Path(Config.api.model_name_or_path).expanduser())
                    if model_name_or_path is None
                    else str(Path(model_name_or_path).expanduser())
                )

                prefix_to_use = str(Config.api.model_prefix_content_default) if prefix_to_use is None else str(prefix_to_use)
                logger.info(
                    f"Creating new `IR` Runner instance. Might take a while. Loading model on {device} device and using prefix \
                    {prefix_to_use}"
                )

                tokenizer = ITokenizer.from_pretrained(model_name_or_path)
                processor = INFERProcessor(tokenizer=tokenizer, prefix=prefix_to_use)

                lm_model = LMFinder.find(model_name_or_path, device=device, **props)

                runner = M1LMRunner(model=lm_model, prediction_heads=[], device=device, processor=processor)

                self._ir_runner = RetrieverApi.named(search_by, store=store, runner=runner, processor=processor, device=device)
        return self._ir_runner

    async def SERVER(self, **props):
        def callback(message, metadata):
            logger.info(message)
            logger.info(metadata)

        return callback


IGNIRunner = IIGNIRunner()


__all__ = ["IGNIRunner"]