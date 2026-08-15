from __future__ import annotations

import asyncio
import math
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any

from justatom.etc.schema import Document
from justatom.retrieval.contracts import DocumentStore, Embedder, EmbeddingProfile, SearchMode
from justatom.retrieval.errors import ConfigurationError
from justatom.retrieval.indexer import Indexer
from justatom.retrieval.retriever import HybridRetriever, KeywordRetriever, VectorRetriever

HuggingFaceEmbedder: Any | None = None
OpenAICompatibleEmbedder: Any | None = None
WeaviateDocumentStore: Any | None = None

_ROOT_KEYS = {"mode", "alpha", "embedding", "store"}
_STORE_KEYS = {"collection", "url", "grpc_port", "grpc_secure"}
_LOCAL_KEYS = {
    "backend",
    "model",
    "device",
    "batch_size",
    "max_length",
    "query_prefix",
    "document_prefix",
    "skip_prefix_if_present",
}
_REMOTE_KEYS = {
    "backend",
    "base_url",
    "model",
    "api_key",
    "timeout",
    "batch_size",
    "max_length",
    "query_prefix",
    "document_prefix",
    "skip_prefix_if_present",
    "encoding_format",
    "extra_body",
}


def _reject_unknown(values: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(key if isinstance(key, str) else repr(key) for key in values if key not in allowed)
    if unknown:
        raise ConfigurationError(f"unknown {section} keys: {', '.join(unknown)}")


def _require_mapping(value: object, section: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{section} must be a mapping")
    return value


def _require_string(values: Mapping[str, Any], key: str) -> str:
    if key not in values:
        raise ConfigurationError(f"{key} is required")
    value = values[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{key} must be a non-empty string")
    return value


def _optional_string(values: Mapping[str, Any], key: str) -> str | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{key} must be a non-empty string when provided")
    return value


def _optional_prefix(values: Mapping[str, Any], key: str) -> str | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigurationError(f"{key} must be a string when provided")
    return value


def _validate_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def _validate_positive_finite_number(value: object, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a positive finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError as error:
        raise ConfigurationError(f"{name} must be a positive finite number") from error
    if not finite or value <= 0:
        raise ConfigurationError(f"{name} must be a positive finite number")
    return value


def _validate_alpha(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError("alpha must be a finite numeric value in [0, 1]")
    try:
        finite = math.isfinite(value)
    except OverflowError as error:
        raise ConfigurationError("alpha must be a finite numeric value in [0, 1]") from error
    if not finite or not 0 <= value <= 1:
        raise ConfigurationError("alpha must be a finite numeric value in [0, 1]")
    return value


def _validate_store(values: Mapping[str, Any]) -> dict[str, Any]:
    _reject_unknown(values, _STORE_KEYS, "store")
    _require_string(values, "collection")
    _optional_string(values, "url")

    grpc_port = values.get("grpc_port", 50051)
    if isinstance(grpc_port, bool) or not isinstance(grpc_port, int):
        raise ConfigurationError("grpc_port must be an integer")
    if not 1 <= grpc_port <= 65535:
        raise ConfigurationError("grpc_port must be in [1, 65535]")

    grpc_secure = values.get("grpc_secure", False)
    if not isinstance(grpc_secure, bool):
        raise ConfigurationError("grpc_secure must be a boolean")

    return dict(values)


def _validate_embedding(values: Mapping[str, Any]) -> dict[str, Any]:
    backend = _require_string(values, "backend")
    if backend == "local":
        _reject_unknown(values, _LOCAL_KEYS, "local embedding")
    elif backend == "openai-compatible":
        _reject_unknown(values, _REMOTE_KEYS, "openai-compatible embedding")
    else:
        raise ConfigurationError("backend must be 'local' or 'openai-compatible'")

    _require_string(values, "model")
    if backend == "local":
        _optional_string(values, "device")
    else:
        _require_string(values, "base_url")
        _optional_string(values, "api_key")
        _optional_string(values, "encoding_format")
        _validate_positive_finite_number(values.get("timeout", 30.0), "timeout")
        if "extra_body" in values and not isinstance(values["extra_body"], Mapping):
            raise ConfigurationError("extra_body must be a mapping")

    _validate_positive_integer(values.get("batch_size", 64), "batch_size")
    _validate_positive_integer(values.get("max_length", 512), "max_length")
    _optional_prefix(values, "query_prefix")
    _optional_prefix(values, "document_prefix")

    skip_prefix_if_present = values.get("skip_prefix_if_present", True)
    if not isinstance(skip_prefix_if_present, bool):
        raise ConfigurationError("skip_prefix_if_present must be a boolean")

    normalized = dict(values)
    if "extra_body" in normalized:
        try:
            normalized["extra_body"] = deepcopy(normalized["extra_body"])
        except Exception as error:
            raise ConfigurationError("extra_body must be safely copyable") from error
    return normalized


def _validate_config(config: Mapping[str, Any]) -> tuple[SearchMode, int | float, dict[str, Any], dict[str, Any] | None]:
    _reject_unknown(config, _ROOT_KEYS, "retrieval")
    mode_value = _require_string(config, "mode")
    try:
        mode = SearchMode(mode_value)
    except ValueError as error:
        raise ConfigurationError(f"Unsupported retrieval mode: {mode_value!r}") from error

    alpha = _validate_alpha(config.get("alpha", 0.5))
    if "store" not in config:
        raise ConfigurationError("store is required")
    store = _validate_store(_require_mapping(config["store"], "store"))

    embedding: dict[str, Any] | None = None
    if "embedding" in config:
        embedding = _validate_embedding(_require_mapping(config["embedding"], "embedding"))
    elif mode is not SearchMode.KEYWORD:
        raise ConfigurationError("embedding is required")
    return mode, alpha, store, embedding


def _build_embedder(embedding: Mapping[str, Any]) -> Embedder:
    profile = EmbeddingProfile(
        query_prefix=embedding.get("query_prefix") or "",
        document_prefix=embedding.get("document_prefix") or "",
        batch_size=embedding.get("batch_size", 64),
        max_length=embedding.get("max_length", 512),
        skip_prefix_if_present=embedding.get("skip_prefix_if_present", True),
    )
    if embedding["backend"] == "local":
        embedder_class = HuggingFaceEmbedder
        if embedder_class is None:
            from justatom.retrieval.embedders.huggingface import HuggingFaceEmbedder as embedder_class

        return embedder_class(
            model=embedding["model"],
            device=embedding.get("device", "auto"),
            profile=profile,
        )
    embedder_class = OpenAICompatibleEmbedder
    if embedder_class is None:
        from justatom.retrieval.embedders.openai_compatible import OpenAICompatibleEmbedder as embedder_class

    return embedder_class(
        base_url=embedding["base_url"],
        model=embedding["model"],
        api_key=embedding.get("api_key"),
        timeout=embedding.get("timeout", 30.0),
        profile=profile,
        encoding_format=embedding.get("encoding_format"),
        extra_body=embedding.get("extra_body"),
    )


async def _close_resources_after_failure(store: DocumentStore | None, embedder: Embedder | None) -> BaseException | None:
    cleanup_error: BaseException | None = None
    resources = (embedder,) if store is embedder else (embedder, store)
    for resource in resources:
        if resource is None:
            continue
        try:
            await resource.close()
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
    return cleanup_error


async def _cleanup_after_failure(store: DocumentStore | None, embedder: Embedder | None) -> BaseException | None:
    cleanup_task = asyncio.create_task(_close_resources_after_failure(store, embedder))
    try:
        return await asyncio.shield(cleanup_task)
    except asyncio.CancelledError as cancellation:
        cleanup_error = await asyncio.shield(cleanup_task)
        if cleanup_error is not None:
            raise cleanup_error from cancellation
        raise


async def build_runtime(config: Mapping[str, Any]) -> RetrievalRuntime:
    config = _require_mapping(config, "retrieval")
    mode, alpha, store_config, embedding_config = _validate_config(config)
    embedder: Embedder | None = None
    store: DocumentStore | None = None
    try:
        if mode is not SearchMode.KEYWORD:
            assert embedding_config is not None
            embedder = _build_embedder(embedding_config)
        store_class = WeaviateDocumentStore
        if store_class is None:
            from justatom.storing.weaviate import WeaviateDocumentStore as store_class

        store = await store_class.connect(
            store_config["collection"],
            url=store_config.get("url"),
            grpc_port=store_config.get("grpc_port", 50051),
            grpc_secure=store_config.get("grpc_secure", False),
        )
        return RetrievalRuntime(store=store, embedder=embedder, mode=mode, alpha=alpha)
    except BaseException as primary_error:
        cleanup_error = await _cleanup_after_failure(store, embedder)
        if cleanup_error is not None:
            raise primary_error from cleanup_error
        raise


class RetrievalRuntime:
    def __init__(
        self,
        store: DocumentStore,
        embedder: Embedder | None,
        mode: SearchMode | str,
        alpha: float = 0.5,
    ) -> None:
        try:
            self.mode = mode if isinstance(mode, SearchMode) else SearchMode(mode)
        except ValueError as error:
            raise ConfigurationError(f"Unsupported retrieval mode: {mode!r}") from error
        if self.mode is not SearchMode.KEYWORD and embedder is None:
            raise ConfigurationError(f"{self.mode.value} retrieval requires an embedder")

        self.store = store
        self.embedder = embedder
        self.indexer = Indexer(store, embedder)
        match self.mode:
            case SearchMode.KEYWORD:
                self.retriever = KeywordRetriever(store)
            case SearchMode.VECTOR:
                self.retriever = VectorRetriever(store, embedder)
            case SearchMode.HYBRID:
                self.retriever = HybridRetriever(store, embedder, alpha=alpha)

        self._lifecycle_lock = asyncio.Lock()
        self._idle = asyncio.Event()
        self._idle.set()
        self._active_operations = 0
        self._closing = False
        self._closed = False
        self._finalization_task: asyncio.Task[None] | None = None

    async def index(
        self,
        documents: Iterable[Document | dict[str, Any]],
        batch_size: int = 64,
        max_parallel_writes: int = 1,
    ) -> int:
        return await self._delegate(
            self.indexer.index(
                documents,
                batch_size=batch_size,
                max_parallel_writes=max_parallel_writes,
            )
        )

    async def retrieve(self, query: str, *, top_k: int = 5, **kwargs: Any) -> list[Document]:
        return await self._delegate(self.retriever.retrieve(query, top_k=top_k, **kwargs))

    async def retrieve_many(
        self,
        queries: Sequence[str],
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[list[Document]]:
        return await self._delegate(self.retriever.retrieve_many(queries, top_k=top_k, **kwargs))

    async def __aenter__(self) -> RetrievalRuntime:
        async with self._lifecycle_lock:
            if self._closing or self._closed:
                raise RuntimeError("RetrievalRuntime is closed")
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            if self._finalization_task is None:
                self._closing = True
                self._finalization_task = asyncio.create_task(self._finalize_close())
            finalization_task = self._finalization_task

        await asyncio.shield(finalization_task)

    async def _delegate(self, operation: Any) -> Any:
        async with self._lifecycle_lock:
            if self._closing or self._closed:
                operation.close()
                raise RuntimeError("RetrievalRuntime is closed")
            self._active_operations += 1
            self._idle.clear()

        try:
            return await operation
        finally:
            async with self._lifecycle_lock:
                self._active_operations -= 1
                if self._active_operations == 0:
                    self._idle.set()

    async def _finalize_close(self) -> None:
        await self._idle.wait()
        first_error: BaseException | None = None
        later_error: BaseException | None = None

        try:
            if self.embedder is not None:
                try:
                    await self.embedder.close()
                except BaseException as error:
                    first_error = error
            if self.store is not self.embedder:
                try:
                    await self.store.close()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
                    else:
                        later_error = error
        finally:
            async with self._lifecycle_lock:
                self._closed = True

        if first_error is not None:
            if later_error is not None:
                raise first_error from later_error
            raise first_error


__all__ = ["RetrievalRuntime", "build_runtime"]
