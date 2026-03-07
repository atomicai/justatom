import os
from collections import OrderedDict
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

import torch
from loguru import logger

from justatom.etc.errors import DocumentStoreError
from justatom.configuring.builtins import load_builtin_yaml
from justatom.configuring.builtins import load_repo_yaml
from justatom.configuring.builtins import get_master_ref
from justatom.modeling.mask import ILanguageModel
from justatom.processing import ITokenizer, RuntimeProcessor
from justatom.running.encoders import EncoderRunner
from justatom.running.embeddings import EmbeddingClientFactory
from justatom.running.embeddings import IEmbeddingClient
from justatom.running.indexer import API as IndexerAPI
from justatom.running.mask import IRetrieverRunner
from justatom.running.retriever import API as RetrieverApi
from justatom.storing.weaviate import Finder as WeaviateApi
from justatom.storing.weaviate import WeaviateDocStore


LM_CACHE_MAXSIZE = int(os.getenv("RUNNING_MODEL_CACHE_SIZE", "4"))
TOKENIZER_CACHE_MAXSIZE = int(os.getenv("RUNNING_TOKENIZER_CACHE_SIZE", "4"))
PROCESSOR_CACHE_MAXSIZE = int(os.getenv("RUNNING_PROCESSOR_CACHE_SIZE", "16"))
ENCODER_CACHE_MAXSIZE = int(os.getenv("RUNNING_ENCODER_CACHE_SIZE", "8"))


@lru_cache(maxsize=LM_CACHE_MAXSIZE)
def _cached_lm_model(model_name_or_path: str) -> ILanguageModel:
    return ILanguageModel.load(model_name_or_path)


@lru_cache(maxsize=TOKENIZER_CACHE_MAXSIZE)
def _cached_tokenizer(model_name_or_path: str):
    return ITokenizer.from_pretrained(model_name_or_path)


@lru_cache(maxsize=PROCESSOR_CACHE_MAXSIZE)
def _cached_processor(
    model_name_or_path: str,
    prefix: str = "",
    max_seq_len: int = 512,
) -> RuntimeProcessor:
    return RuntimeProcessor(
        tokenizer=_cached_tokenizer(model_name_or_path),
        max_seq_len=max_seq_len,
        prefix=prefix,
    )


@lru_cache(maxsize=ENCODER_CACHE_MAXSIZE)
def _cached_encoder_runner(
    model_name_or_path: str,
    device: str = "cpu",
) -> EncoderRunner:
    return EncoderRunner(
        model=_cached_lm_model(model_name_or_path),
        prediction_heads=[],
        device=device,
    ).eval()


class RunningService:
    _embedding_clients: OrderedDict[str, IEmbeddingClient] = OrderedDict()

    _EMBED_CACHE_MAXSIZE: int = int(os.getenv("RUNNING_EMBED_CACHE_SIZE", "8"))

    @staticmethod
    def _to_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        norm = str(value).strip().lower()
        if norm in {"1", "true", "yes", "on"}:
            return True
        if norm in {"0", "false", "no", "off"}:
            return False
        if norm.startswith("${") and norm.endswith("}"):
            return default
        return default

    @staticmethod
    def _to_optional_str(value: Any, default: str | None = None) -> str | None:
        if value is None:
            return default
        out = str(value).strip()
        if out == "":
            return default
        if out.startswith("${") and out.endswith("}"):
            return default
        return out

    @staticmethod
    def _to_optional_int(value: Any, default: int | None = None) -> int | None:
        if value is None:
            return default
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("${") and stripped.endswith("}"):
                return default
        try:
            out = int(value)
        except (TypeError, ValueError):
            return default
        return out if out > 0 else default

    @staticmethod
    @lru_cache(maxsize=1)
    def _embedding_openai_defaults() -> dict[str, Any]:
        cfg_ref = get_master_ref(
            "refs",
            "builtins",
            "embeddings_config",
            default="justatom/builtins/configs/embeddings.yaml",
        )
        if cfg_ref and cfg_ref.startswith("justatom/builtins/"):
            cfg = load_builtin_yaml(cfg_ref.replace("justatom/builtins/", "", 1))
        elif cfg_ref:
            cfg = load_repo_yaml(cfg_ref)
        else:
            cfg = load_builtin_yaml("configs/embeddings.yaml")
        openai_cfg = cfg.get("openai_compatible") or {}
        prefixes = openai_cfg.get("prefixes") or {}
        req = openai_cfg.get("request") or {}
        return {
            "query_prefix": RunningService._to_optional_str(
                prefixes.get("query", ""),
                "",
            )
            or "",
            "passage_prefix": RunningService._to_optional_str(
                prefixes.get("passage", ""),
                "",
            )
            or "",
            "default_input_type": RunningService._to_optional_str(
                prefixes.get("default_input_type", "raw"),
                "raw",
            )
            or "raw",
            "prefix_enabled": RunningService._to_bool(prefixes.get("enabled"), True),
            "prefix_skip_if_present": RunningService._to_bool(
                prefixes.get("skip_if_present"),
                True,
            ),
            "default_pooling": RunningService._to_optional_str(
                req.get("pooling", ""),
                None,
            ),
            "default_encoding_format": RunningService._to_optional_str(
                req.get("encoding_format", ""),
                None,
            ),
            "default_max_seq_len": RunningService._to_optional_int(
                req.get("max_seq_len"),
                None,
            ),
        }

    @staticmethod
    def maybe_cuda_or_mps(devices: list[str] = None):  # type: ignore
        devices = {"cuda", "mps", "cpu"} if devices is None else set(devices)  # type: ignore
        if torch.cuda.is_available() and "cuda" in devices:
            return "cuda:0"
        if torch.mps.is_available() and "mps" in devices:
            return "mps"
        return "cpu"

    @staticmethod
    def _cache_get(cache: OrderedDict[str, Any], key: str) -> Any | None:
        if key not in cache:
            return None
        value = cache.pop(key)
        cache[key] = value
        return value

    @staticmethod
    def _cache_put(
        cache: OrderedDict[str, Any],
        key: str,
        value: Any,
        maxsize: int,
    ) -> None:
        if key in cache:
            cache.pop(key)
        cache[key] = value
        if len(cache) > maxsize:
            cache.popitem(last=False)

    @staticmethod
    def _embedding_cache_key(
        backend: str,
        model: str,
        base_url: str | None,
        device: str,
        max_seq_len: int | None,
    ) -> str:
        return "|".join(
            [
                backend.strip().lower(),
                model,
                "" if base_url is None else base_url.strip().lower(),
                device,
                "" if max_seq_len is None else str(int(max_seq_len)),
            ]
        )

    @staticmethod
    async def get_embedding_client(
        backend: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        device: str | None = None,
        prefix: str = "",
        max_seq_len: int | None = None,
        batch_size: int = 64,
        timeout: float = 30.0,
    ) -> IEmbeddingClient:
        backend = (backend or os.getenv("EMBEDDING_BACKEND") or "local").strip()
        model = model or os.getenv("EMBEDDING_MODEL_NAME_OR_PATH")
        if model is None:
            msg = "Embedding model is not set. Use `EMBEDDING_MODEL_NAME_OR_PATH` or pass `model`."
            raise ValueError(msg)

        device = device or RunningService.maybe_cuda_or_mps()
        cache_key = RunningService._embedding_cache_key(
            backend=backend,
            model=model,
            base_url=base_url,
            device=device,
            max_seq_len=max_seq_len,
        )
        cached_client = RunningService._cache_get(
            RunningService._embedding_clients, cache_key
        )
        if cached_client is not None:
            return cached_client

        normalized_backend = backend.strip().lower()
        if normalized_backend in {"openai", "openai-compatible", "openai_compatible"}:
            defaults = RunningService._embedding_openai_defaults()
            resolved_base_url = (
                base_url or os.getenv("EMBEDDING_BASE_URL") or ""
            ).strip()
            resolved_api_key = (api_key or os.getenv("EMBEDDING_API_KEY") or "").strip()
            if resolved_base_url == "":
                raise ValueError(
                    "`base_url` or `EMBEDDING_BASE_URL` is required for openai backend"
                )
            if resolved_api_key == "":
                raise ValueError(
                    "`api_key` or `EMBEDDING_API_KEY` is required for openai backend"
                )

            client = EmbeddingClientFactory.from_backend(
                normalized_backend,
                base_url=resolved_base_url,
                api_key=resolved_api_key,
                model=model,
                timeout=timeout,
                query_prefix=defaults["query_prefix"],
                passage_prefix=defaults["passage_prefix"],
                default_input_type=defaults["default_input_type"],
                prefix_enabled=defaults["prefix_enabled"],
                prefix_skip_if_present=defaults["prefix_skip_if_present"],
                default_pooling=defaults["default_pooling"],
                default_encoding_format=defaults["default_encoding_format"],
                default_max_seq_len=(
                    max_seq_len
                    if max_seq_len is not None
                    else defaults["default_max_seq_len"]
                ),
            )
        else:
            client = EmbeddingClientFactory.from_backend(
                "local",
                model_name_or_path=model,
                device=device,
                prefix=prefix,
                max_seq_len=(512 if max_seq_len is None else max_seq_len),
                batch_size=batch_size,
            )

        RunningService._cache_put(
            RunningService._embedding_clients,
            cache_key,
            client,
            RunningService._EMBED_CACHE_MAXSIZE,
        )
        return client

    @staticmethod
    async def embed_texts(
        texts: list[str],
        *,
        backend: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        device: str | None = None,
        prefix: str = "",
        max_seq_len: int | None = None,
        batch_size: int = 64,
        timeout: float = 30.0,
        **props: Any,
    ) -> list[list[float]]:
        client = await RunningService.get_embedding_client(
            backend=backend,
            model=model,
            base_url=base_url,
            api_key=api_key,
            device=device,
            prefix=prefix,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            timeout=timeout,
        )
        return await client.embed(texts=texts, model=model, **props)

    @staticmethod
    async def embed_queries(
        queries: list[str],
        *,
        backend: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        device: str | None = None,
        prefix: str = "",
        max_seq_len: int | None = None,
        batch_size: int = 64,
        timeout: float = 30.0,
        **props: Any,
    ) -> list[list[float]]:
        # Client decides how to apply query prefixes for the selected backend.
        return await RunningService.embed_texts(
            texts=queries,
            backend=backend,
            model=model,
            base_url=base_url,
            api_key=api_key,
            device=device,
            prefix=prefix,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            timeout=timeout,
            input_type="query",
            **props,
        )

    @staticmethod
    async def embed_passages(
        passages: list[str],
        *,
        backend: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        device: str | None = None,
        prefix: str = "",
        max_seq_len: int | None = None,
        batch_size: int = 64,
        timeout: float = 30.0,
        **props: Any,
    ) -> list[list[float]]:
        # Client decides how to apply passage/document prefixes for selected backend.
        return await RunningService.embed_texts(
            texts=passages,
            backend=backend,
            model=model,
            base_url=base_url,
            api_key=api_key,
            device=device,
            prefix=prefix,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            timeout=timeout,
            input_type="passage",
            **props,
        )

    @staticmethod
    async def close_embedding_clients() -> None:
        for client in RunningService._embedding_clients.values():
            await client.close()
        RunningService._embedding_clients.clear()

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
    def get_or_create_lm_model(model_name_or_path: str) -> ILanguageModel:
        return _cached_lm_model(model_name_or_path)

    @staticmethod
    def get_or_create_tokenizer(model_name_or_path: str):
        return _cached_tokenizer(model_name_or_path)

    @staticmethod
    def get_or_create_processor(
        model_name_or_path: str,
        prefix: str = "",
        max_seq_len: int = 512,
    ) -> RuntimeProcessor:
        return _cached_processor(model_name_or_path, prefix or "", max_seq_len)

    @staticmethod
    def get_or_create_encoder_runner(
        model_name_or_path: str,
        device: str = "cpu",
    ) -> EncoderRunner:
        return _cached_encoder_runner(model_name_or_path, device)

    @staticmethod
    def clear_runner_caches() -> None:
        _cached_processor.cache_clear()
        _cached_tokenizer.cache_clear()
        _cached_encoder_runner.cache_clear()
        _cached_lm_model.cache_clear()

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

        ix_processor = RunningService.get_or_create_processor(
            model_name_or_path=model_name_or_path,
            prefix=content_prefix or "",
        )
        ir_processor = RunningService.get_or_create_processor(
            model_name_or_path=model_name_or_path,
            prefix=query_prefix or "",
        )
        runner = RunningService.get_or_create_encoder_runner(
            model_name_or_path=model_name_or_path,
            device=device,
        )

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
