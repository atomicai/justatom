from collections.abc import Iterable

import torch
from loguru import logger

from justatom.etc.errors import DocumentStoreError
from justatom.modeling.mask import ILanguageModel
from justatom.processing import ITokenizer, RuntimeProcessor
from justatom.running.encoders import EncoderRunner
from justatom.running.indexer import API as IndexerAPI
from justatom.running.mask import IRetrieverRunner
from justatom.running.retriever import API as RetrieverApi
from justatom.storing.weaviate import Finder as WeaviateApi
from justatom.storing.weaviate import WeaviateDocStore


class RunningService:
    @staticmethod
    def maybe_cuda_or_mps(devices: list[str] = None):  # type: ignore
        devices = {"cuda", "mps", "cpu"} if devices is None else set(devices)  # type: ignore
        if torch.cuda.is_available() and "cuda" in devices:
            return "cuda:0"
        if torch.mps.is_available() and "mps" in devices:
            return "mps"
        return "cpu"

    @staticmethod
    async def check_store_and_message(
        store: WeaviateDocStore, delete_if_not_empty: bool
    ):
        n_docs_count: int = await store.count_documents()
        collection_name = store.collection_name
        if delete_if_not_empty:
            status_message: bool = await store.delete_all_documents()
            if not status_message:
                raise DocumentStoreError(
                    f"Documents per collection {collection_name} are not deleted. See logs for more details"
                )
        elif n_docs_count > 0:
            logger.warning(
                f"You're not deleting any documents. Using pre-built {n_docs_count} documents per collection {collection_name}"
            )
        return store, n_docs_count

    @staticmethod
    def igni_runners(
        store,
        search_pipeline: str,
        model_name_or_path,
        query_prefix: str = "",
        content_prefix: str = "",
        device: str = "cpu",
        **props,
    ):
        if search_pipeline == "keywords":
            return IndexerAPI.named(search_pipeline, store=store), RetrieverApi.named(
                search_pipeline, store=store
            )

        if model_name_or_path is None:
            msg = f"You have specified `runner_name`=[{search_pipeline}] but `model_name_or_path` is None."
            logger.error(msg)
            raise ValueError(msg)

        lm_model = ILanguageModel.load(model_name_or_path)
        ix_processor = RuntimeProcessor(
            ITokenizer.from_pretrained(model_name_or_path), prefix=content_prefix
        )
        ir_processor = RuntimeProcessor(
            ITokenizer.from_pretrained(model_name_or_path), prefix=query_prefix
        )
        runner = EncoderRunner(model=lm_model, prediction_heads=[], device=device)

        ix_runner = IndexerAPI.named(
            search_pipeline,
            store=store,
            runner=runner,
            processor=ix_processor,
            device=device,
        )
        ir_runner = RetrieverApi.named(
            search_pipeline,
            store=store,
            runner=runner,
            processor=ir_processor,
            device=device,
            **props,
        )
        return ix_runner, ir_runner

    @staticmethod
    async def do_index_and_prepare_for_search(
        collection_name: str,
        documents: Iterable[dict],
        model_name_or_path: str | None = None,
        index_and_eval_by: str = "embedding",
        query_prefix: str = None,
        content_prefix: str = None,
        batch_size: int = 4,
        flush_collection: bool = False,
        devices: list[str] = None,
        weaviate_host: str = "localhost",
        weaviate_port: int = 2211,
        **props,
    ) -> IRetrieverRunner:
        store: WeaviateDocStore = await WeaviateApi.find(
            collection_name, WEAVIATE_HOST=weaviate_host, WEAVIATE_PORT=weaviate_port
        )
        store, n_total_docs = await RunningService.check_store_and_message(
            store, delete_if_not_empty=flush_collection
        )
        device = RunningService.maybe_cuda_or_mps(devices=devices)

        ix_runner, ir_runner = RunningService.igni_runners(
            store=store,
            search_pipeline=index_and_eval_by,
            model_name_or_path=model_name_or_path,
            device=device,
            query_prefix=query_prefix,
            content_prefix=content_prefix,
            **props,
        )
        if not flush_collection and n_total_docs > 0:
            return ir_runner

        logger.info("Indexing in progress")
        n_total_docs = await ix_runner.index(
            documents=documents, batch_size=batch_size, device=device
        )
        logger.info(f"Total docs in index {n_total_docs}")
        return ir_runner


__all__ = ["RunningService"]
