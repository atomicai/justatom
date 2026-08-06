from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

from quart import Quart, request

from justatom.api.dataset_input import documents_from_input
from justatom.configuring.scenarios import load_scenario_config
from justatom.retrieval.runtime import RetrievalRuntime, build_runtime


def _reject_unknown_fields(payload: dict[str, Any], allowed: set[str]):
    unknown = sorted(set(payload) - allowed)
    if unknown:
        return {"error": f"unsupported fields: {', '.join(unknown)}"}, 400
    return None


async def _payload() -> tuple[dict[str, Any] | None, tuple[dict[str, str], int] | None]:
    payload = await request.get_json(silent=True)
    if not isinstance(payload, dict):
        return None, ({"error": "request body must be a JSON object"}, 400)
    return payload, None


def _positive_integer(value: object, name: str, default: int) -> tuple[int | None, tuple[dict[str, str], int] | None]:
    if value is None:
        return default, None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None, ({"error": f"{name} must be a positive integer"}, 400)
    return value, None


def _documents_are_valid(source: object) -> bool:
    if isinstance(source, str):
        return bool(source.strip())
    if not isinstance(source, list):
        return False
    return all(
        isinstance(document, dict)
        and isinstance(document.get("content"), str)
        and isinstance(document.get("meta", {}), dict)
        and isinstance(document.get("keywords_or_phrases", []), list)
        for document in source
    )


async def _finish_cleanup(awaitable: Awaitable[object]) -> None:
    cleanup_task = asyncio.ensure_future(awaitable)
    try:
        await asyncio.shield(cleanup_task)
    except asyncio.CancelledError:
        await asyncio.shield(cleanup_task)
        raise


def create_app(
    config: dict[str, Any] | None = None,
    runtime: RetrievalRuntime | None = None,
    start_mq: bool = True,
) -> Quart:
    scenario_config = load_scenario_config("serve", config=config)
    retrieval_config = scenario_config["retrieval"]
    app = Quart(__name__, static_folder=None)
    app.config.setdefault("PROVIDE_AUTOMATIC_OPTIONS", True)
    app.extensions["retrieval_config"] = retrieval_config
    if runtime is not None:
        app.extensions["retrieval_runtime"] = runtime

    def _runtime() -> RetrievalRuntime:
        return app.extensions["retrieval_runtime"]

    @app.before_serving
    async def start() -> None:
        if "retrieval_runtime" not in app.extensions:
            app.extensions["retrieval_runtime"] = await build_runtime(app.extensions["retrieval_config"])

        if not start_mq:
            return

        from justatom.mq.clients.rabbitmq import RabbitMQClient
        from justatom.mq.settings.rabbitmq import SettingsRabbitMQ

        client_name = "consumer"

        def receive(message: str, metadata: dict[str, Any]) -> None:
            del message, metadata

        client = RabbitMQClient(SettingsRabbitMQ(), client_name=client_name)
        app.extensions["retrieval_mq_task"] = asyncio.create_task(
            client.consume_with_callback(callback=receive, routing_key=client_name)
        )

    @app.after_serving
    async def stop() -> None:
        mq_task = app.extensions.pop("retrieval_mq_task", None)
        if mq_task is not None:
            mq_task.cancel()
            try:
                await _finish_cleanup(mq_task)
            except asyncio.CancelledError:
                pass

        app_runtime = app.extensions.pop("retrieval_runtime", None)
        if app_runtime is not None:
            await _finish_cleanup(app_runtime.close())

    @app.get("/")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/searching")
    async def search():
        payload, error = await _payload()
        if error is not None:
            return error
        assert payload is not None
        rejected = _reject_unknown_fields(payload, {"text", "top_k", "filter_by"})
        if rejected is not None:
            return rejected

        query = payload.get("text")
        if not isinstance(query, str) or not query.strip():
            return {"error": "text must be a non-empty string"}, 400
        top_k, error = _positive_integer(payload.get("top_k"), "top_k", default=5)
        if error is not None:
            return error
        filter_by = payload.get("filter_by")
        if filter_by is not None and not isinstance(filter_by, dict):
            return {"error": "filter_by must be an object"}, 400

        documents = await _runtime().retrieve(query.strip(), top_k=top_k, filters=filter_by)
        return {"docs": [document.to_dict(uuid_to_str=True) for document in documents]}

    @app.post("/indexing")
    async def index():
        payload, error = await _payload()
        if error is not None:
            return error
        assert payload is not None
        rejected = _reject_unknown_fields(payload, {"dataset_name_or_docs", "batch_size"})
        if rejected is not None:
            return rejected

        source = payload.get("dataset_name_or_docs")
        if not _documents_are_valid(source):
            return {"error": "dataset_name_or_docs must be a non-empty dataset name or a list of documents"}, 400
        batch_size, error = _positive_integer(payload.get("batch_size"), "batch_size", default=64)
        if error is not None:
            return error

        await _runtime().index(documents_from_input(source), batch_size=batch_size)
        return {"total_docs": await _runtime().store.count_documents()}

    @app.post("/delete")
    async def delete():
        payload, error = await _payload()
        if error is not None:
            return error
        assert payload is not None
        rejected = _reject_unknown_fields(payload, set())
        if rejected is not None:
            return rejected

        app_runtime = _runtime()
        total_docs = await app_runtime.store.count_documents()
        await app_runtime.store.clear()
        return {"deleted_docs": total_docs}

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5555)
