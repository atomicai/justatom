import asyncio

from justatom.api.run import create_app
from justatom.etc.schema import Document


class FakeStore:
    def __init__(self):
        self.cleared = 0

    async def count_documents(self):
        return 7

    async def clear(self):
        self.cleared += 1


class FakeRuntime:
    def __init__(self):
        self.store = FakeStore()
        self.closed = 0
        self.indexed = []

    async def retrieve(self, query, **kwargs):
        return [Document(content=f"result:{query}")]

    async def index(self, documents, **kwargs):
        self.indexed.extend(list(documents))
        return len(self.indexed)

    async def close(self):
        self.closed += 1


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
