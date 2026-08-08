from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger
from quart import Quart, request

from justatom.retrieval.contracts import Embedder, EmbeddingProfile, validate_embeddings
from justatom.retrieval.embedders.huggingface import HuggingFaceEmbedder
from justatom.retrieval.errors import ConfigurationError


def _positive_integer(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{name} must be a positive integer") from error
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return parsed


@dataclass(frozen=True)
class EmbeddingServerSettings:
    model: str
    device: str
    batch_size: int
    max_length: int

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "EmbeddingServerSettings":
        values = os.environ if env is None else env
        model = values.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B").strip()
        if not model:
            raise ConfigurationError("EMBEDDING_MODEL must be non-empty")
        device = values.get("EMBEDDING_DEVICE", "cpu").strip()
        if not device:
            raise ConfigurationError("EMBEDDING_DEVICE must be non-empty")
        return cls(
            model=model,
            device=device,
            batch_size=_positive_integer(values.get("EMBEDDING_BATCH_SIZE", "8"), "EMBEDDING_BATCH_SIZE"),
            max_length=_positive_integer(values.get("EMBEDDING_MAX_LENGTH", "512"), "EMBEDDING_MAX_LENGTH"),
        )


def build_local_embedder(settings: EmbeddingServerSettings) -> Embedder:
    return HuggingFaceEmbedder(
        model=settings.model,
        device=settings.device,
        profile=EmbeddingProfile(batch_size=settings.batch_size, max_length=settings.max_length),
    )


EmbedderFactory = Callable[[EmbeddingServerSettings], Embedder]


async def _finish_close(app: Quart) -> None:
    close_task = app.extensions.get("embedding_close_task")
    if close_task is None:
        embedder = app.extensions.pop("embedder", None)
        if embedder is None:
            return
        close_task = asyncio.create_task(embedder.close())
        app.extensions["embedding_close_task"] = close_task
    try:
        await asyncio.shield(close_task)
    except asyncio.CancelledError:
        await asyncio.shield(close_task)
        raise


def create_embedding_app(
    settings: EmbeddingServerSettings | None = None,
    embedder: Embedder | None = None,
    embedder_factory: EmbedderFactory = build_local_embedder,
) -> Quart:
    resolved = settings or EmbeddingServerSettings.from_env()
    app = Quart(__name__, static_folder=None)
    app.json.ensure_ascii = False
    app.config.setdefault("PROVIDE_AUTOMATIC_OPTIONS", True)
    app.extensions["embedding_settings"] = resolved
    if embedder is not None:
        app.extensions["embedder"] = embedder

    @app.errorhandler(500)
    async def internal_error(error):
        logger.error("unhandled embedding server error [{}]", type(error).__name__)
        return _error("embedding backend failed", 500, "server_error")

    @app.before_serving
    async def start() -> None:
        if "embedder" not in app.extensions:
            app.extensions["embedder"] = embedder_factory(resolved)

    @app.after_serving
    async def stop() -> None:
        await _finish_close(app)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": resolved.model}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": resolved.model, "object": "model", "owned_by": "justatom"}]}

    @app.post("/v1/embeddings")
    async def embeddings():
        payload: Any = await request.get_json(silent=True)
        error, texts = _validate_embedding_request(payload, resolved)
        if error is not None:
            return error
        try:
            vectors = validate_embeddings(
                await app.extensions["embedder"].embed_documents(texts),
                expected_count=len(texts),
            )
        except Exception as error:
            logger.error("embedding backend failed [{}]", type(error).__name__)
            return _error("embedding backend failed", 500, "server_error")
        return {
            "object": "list",
            "model": resolved.model,
            "data": [{"object": "embedding", "index": index, "embedding": vector} for index, vector in enumerate(vectors)],
        }

    return app


def _error(message: str, status: int, error_type: str):
    return {"error": {"message": message, "type": error_type}}, status


def _validate_embedding_request(payload: Any, settings: EmbeddingServerSettings):
    if not isinstance(payload, dict):
        return (_error("request body must be a JSON object", 400, "invalid_request_error"), None)
    unknown = sorted(set(payload) - {"model", "input", "encoding_format"})
    if unknown:
        return (_error(f"unsupported fields: {', '.join(unknown)}", 400, "invalid_request_error"), None)
    if payload.get("model") != settings.model:
        return (_error("requested model is not available", 404, "model_not_found"), None)
    encoding = payload.get("encoding_format", "float")
    if encoding != "float":
        return (_error("encoding_format must be 'float'", 400, "invalid_request_error"), None)

    source = payload.get("input")
    if isinstance(source, str):
        texts = [source]
    elif isinstance(source, list):
        texts = source
    else:
        return (_error("input must be a string or list of strings", 400, "invalid_request_error"), None)
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        return (_error("input strings must be non-empty", 400, "invalid_request_error"), None)
    if len(texts) > settings.batch_size:
        return (_error("input exceeds configured batch size", 413, "request_too_large"), None)
    return None, texts
