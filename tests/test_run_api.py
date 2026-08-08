import asyncio
import sys
from types import ModuleType

import pytest

from justatom.api.run import create_app
from justatom.etc.schema import Document
from justatom.retrieval.errors import ConfigurationError, EmbeddingBackendError


class FakeStore:
    def __init__(self):
        self.cleared = 0

    async def count_documents(self):
        return 7

    async def clear(self):
        self.cleared += 1


class FakeRuntime:
    def __init__(self, *, close_error=None, close_started=None, close_release=None):
        self.store = FakeStore()
        self.closed = 0
        self.indexed = []
        self.close_error = close_error
        self.close_started = close_started
        self.close_release = close_release

    async def retrieve(self, query, **kwargs):
        return [Document(content=f"result:{query}")]

    async def index(self, documents, **kwargs):
        self.indexed.extend(list(documents))
        return len(self.indexed)

    async def close(self):
        self.closed += 1
        if self.close_started is not None:
            self.close_started.set()
        if self.close_release is not None:
            await self.close_release.wait()
        if self.close_error is not None:
            raise self.close_error


def test_app_reuses_one_runtime_for_search_index_and_shutdown():
    async def scenario():
        runtime = FakeRuntime()
        app = create_app(runtime=runtime, start_mq=False)
        async with app.test_app() as test_app:
            client = test_app.test_client()
            search = await client.post("/searching", json={"text": "cats", "top_k": 3})
            assert (await search.get_json())["docs"][0]["content"] == "result:cats"
            indexed = await client.post("/indexing", json={"dataset_name_or_docs": [{"content": "doc"}]})
            assert (await indexed.get_json())["total_docs"] == 7
        assert runtime.closed == 1

    asyncio.run(scenario())


def test_search_response_serializes_unicode_as_utf8():
    async def scenario():
        app = create_app(runtime=FakeRuntime(), start_mq=False)
        async with app.test_app() as test_app:
            response = await test_app.test_client().post("/searching", json={"text": "привет"})
            body = await response.get_data()

        assert "result:привет".encode() in body
        assert b"\\u043f" not in body

    asyncio.run(scenario())


def test_app_builds_runtime_during_lifecycle_only(monkeypatch):
    async def scenario():
        runtime = FakeRuntime()
        calls = []

        async def build(config):
            calls.append(config)
            return runtime

        monkeypatch.setattr("justatom.api.run.build_runtime", build)
        app = create_app(start_mq=False)
        assert calls == []
        async with app.test_app():
            assert app.extensions["retrieval_runtime"] is runtime
        assert len(calls) == 1
        assert runtime.closed == 1

    asyncio.run(scenario())


def test_mq_construction_failure_closes_runtime_and_preserves_startup_error(monkeypatch):
    async def scenario():
        runtime = FakeRuntime(close_error=RuntimeError("runtime close failed"))

        def fail_to_construct(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("mq unavailable")

        rabbitmq = ModuleType("justatom.mq.clients.rabbitmq")
        rabbitmq.RabbitMQClient = fail_to_construct
        settings = ModuleType("justatom.mq.settings.rabbitmq")
        settings.SettingsRabbitMQ = object
        monkeypatch.setitem(sys.modules, "justatom.mq.clients.rabbitmq", rabbitmq)
        monkeypatch.setitem(sys.modules, "justatom.mq.settings.rabbitmq", settings)
        app = create_app(runtime=runtime)
        start = app.before_serving_funcs[0]

        with pytest.raises(RuntimeError, match="mq unavailable") as raised:
            await start()

        assert str(raised.value.__cause__) == "runtime close failed"
        assert runtime.closed == 1
        assert "retrieval_runtime" not in app.extensions
        assert "retrieval_mq_task" not in app.extensions

    asyncio.run(scenario())


def test_startup_failure_finishes_close_when_caller_is_cancelled(monkeypatch):
    async def scenario():
        close_started = asyncio.Event()
        release_close = asyncio.Event()
        runtime = FakeRuntime(close_started=close_started, close_release=release_close)

        def fail_to_construct(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("mq unavailable")

        rabbitmq = ModuleType("justatom.mq.clients.rabbitmq")
        rabbitmq.RabbitMQClient = fail_to_construct
        settings = ModuleType("justatom.mq.settings.rabbitmq")
        settings.SettingsRabbitMQ = object
        monkeypatch.setitem(sys.modules, "justatom.mq.clients.rabbitmq", rabbitmq)
        monkeypatch.setitem(sys.modules, "justatom.mq.settings.rabbitmq", settings)
        app = create_app(runtime=runtime)
        start_task = asyncio.create_task(app.before_serving_funcs[0]())
        await close_started.wait()
        start_task.cancel()
        release_close.set()

        with pytest.raises(RuntimeError, match="mq unavailable"):
            await start_task
        assert runtime.closed == 1
        assert "retrieval_runtime" not in app.extensions

    asyncio.run(scenario())


def test_failed_mq_task_does_not_skip_runtime_close_or_double_close():
    async def scenario():
        mq_failed = asyncio.Event()

        async def failed_consumer():
            mq_failed.set()
            raise RuntimeError("mq task failed")

        runtime = FakeRuntime(close_error=RuntimeError("runtime close failed"))
        app = create_app(runtime=runtime, start_mq=False)
        mq_task = asyncio.create_task(failed_consumer())
        await mq_failed.wait()
        await asyncio.wait({mq_task})
        app.extensions["retrieval_mq_task"] = mq_task
        stop = app.after_serving_funcs[0]

        with pytest.raises(RuntimeError, match="mq task failed") as raised:
            await stop()
        assert str(raised.value.__cause__) == "runtime close failed"
        assert runtime.closed == 1

        with pytest.raises(RuntimeError, match="mq task failed"):
            await stop()
        assert runtime.closed == 1

    asyncio.run(scenario())


def test_shutdown_reraises_caller_cancellation_after_cleanup():
    async def scenario():
        consumer_ready = asyncio.Event()
        consumer_cancelled = asyncio.Event()
        release_consumer = asyncio.Event()
        close_started = asyncio.Event()
        release_close = asyncio.Event()

        async def consumer():
            consumer_ready.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                consumer_cancelled.set()
                await release_consumer.wait()
                raise

        runtime = FakeRuntime(close_started=close_started, close_release=release_close)
        app = create_app(runtime=runtime, start_mq=False)
        mq_task = asyncio.create_task(consumer())
        await consumer_ready.wait()
        app.extensions["retrieval_mq_task"] = mq_task
        stop = app.after_serving_funcs[0]

        stop_task = asyncio.create_task(stop())
        await consumer_cancelled.wait()
        stop_task.cancel()
        release_consumer.set()
        await close_started.wait()
        release_close.set()

        with pytest.raises(asyncio.CancelledError):
            await stop_task
        assert runtime.closed == 1

    asyncio.run(scenario())


def test_search_rejects_runtime_composition_fields():
    async def scenario():
        app = create_app(runtime=FakeRuntime(), start_mq=False)
        async with app.test_app() as test_app:
            response = await test_app.test_client().post(
                "/searching",
                json={"text": "q", "collection_name": "other", "search_by": "hybrid", "top_p": 0.9},
            )
            assert response.status_code == 400
            assert "collection_name" in (await response.get_json())["error"]

    asyncio.run(scenario())


def test_requests_reject_invalid_payloads_without_delegating():
    async def scenario():
        runtime = FakeRuntime()
        app = create_app(runtime=runtime, start_mq=False)
        async with app.test_app() as test_app:
            client = test_app.test_client()
            invalid_searches = (
                {"text": "", "top_k": 1},
                {"text": "q", "top_k": 0},
                {"text": "q", "top_k": True},
                {"text": "q", "filter_by": "not-a-mapping"},
            )
            for payload in invalid_searches:
                assert (await client.post("/searching", json=payload)).status_code == 400
            invalid_indexes = (
                {"dataset_name_or_docs": 7},
                {"dataset_name_or_docs": [], "batch_size": 0},
                {"dataset_name_or_docs": [], "batch_size": True},
                {"dataset_name_or_docs": [{"content": 7}]},
            )
            for payload in invalid_indexes:
                assert (await client.post("/indexing", json=payload)).status_code == 400
        assert runtime.indexed == []

    asyncio.run(scenario())


def test_delete_delegates_to_runtime_store():
    async def scenario():
        runtime = FakeRuntime()
        app = create_app(runtime=runtime, start_mq=False)
        async with app.test_app() as test_app:
            response = await test_app.test_client().post("/delete", json={})
            assert await response.get_json() == {"deleted_docs": 7}
        assert runtime.store.cleared == 1

    asyncio.run(scenario())


def test_create_app_loads_explicit_config_path(monkeypatch, tmp_path):
    path = tmp_path / "serve.yaml"
    path.write_text(
        "retrieval:\n" "  mode: keyword\n" "  store:\n" "    collection: ExplicitConfig\n",
        encoding="utf-8",
    )
    app = create_app(config_path=path, runtime=FakeRuntime(), start_mq=False)
    assert app.extensions["retrieval_config"]["store"]["collection"] == "ExplicitConfig"


def test_create_app_rejects_unresolved_environment_placeholders(tmp_path):
    path = tmp_path / "serve.yaml"
    path.write_text(
        "retrieval:\n"
        "  mode: vector\n"
        "  embedding:\n"
        "    backend: openai-compatible\n"
        "    base_url: ${EMBEDDING_BASE_URL}\n"
        "    model: model\n"
        "  store:\n"
        "    collection: Docs\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="EMBEDDING_BASE_URL"):
        create_app(config_path=path, runtime=FakeRuntime(), start_mq=False)


def test_create_app_sanitizes_embedding_backend_failures():
    class FailingRuntime(FakeRuntime):
        async def retrieve(self, query, **kwargs):
            raise EmbeddingBackendError(f"upstream secret for {query}")

    async def scenario():
        app = create_app(runtime=FailingRuntime(), start_mq=False)
        async with app.test_app() as test_app:
            response = await test_app.test_client().post("/searching", json={"text": "private query"})
            body = await response.get_data()
            assert response.status_code == 502
            assert await response.get_json() == {"error": "embedding backend unavailable"}
            assert b"upstream secret" not in body
            assert b"private query" not in body

    asyncio.run(scenario())
