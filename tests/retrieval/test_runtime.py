import asyncio

import pytest

from justatom.etc.schema import Document
from justatom.retrieval.contracts import SearchMode
from justatom.retrieval.errors import ConfigurationError
from justatom.retrieval.indexer import Indexer
from justatom.retrieval.retriever import HybridRetriever, KeywordRetriever, VectorRetriever
from justatom.retrieval.runtime import RetrievalRuntime


class CloseableStore:
    def __init__(self):
        self.closed = 0
        self.calls = []
        self.written = []

    async def close(self):
        self.closed += 1

    async def write_documents(self, documents):
        batch = list(documents)
        self.written.append(batch)
        return len(batch)

    async def search_keywords(self, queries, **kwargs):
        self.calls.append(("keyword", list(queries), kwargs))
        return [[Document(content=f"keyword:{query}")] for query in queries]

    async def search_vector(self, vectors, **kwargs):
        self.calls.append(("vector", list(vectors), kwargs))
        return [[Document(content=f"vector:{index}")] for index, _ in enumerate(vectors)]

    async def search_hybrid(self, queries, vectors, **kwargs):
        self.calls.append(("hybrid", list(queries), list(vectors), kwargs))
        return [[Document(content=f"hybrid:{query}")] for query in queries]


class CloseableEmbedder:
    def __init__(self):
        self.closed = 0
        self.query_calls = []
        self.document_calls = []

    async def close(self):
        self.closed += 1

    async def embed_queries(self, texts):
        self.query_calls.append(list(texts))
        return [[float(index), 1.0] for index, _ in enumerate(texts)]

    async def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class DualProtocolCloseable(CloseableStore):
    async def embed_queries(self, texts):
        return [[float(index), 1.0] for index, _ in enumerate(texts)]

    async def embed_documents(self, texts):
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


def test_runtime_selects_mode_and_closes_resources_once():
    store = CloseableStore()
    embedder = CloseableEmbedder()
    runtime = RetrievalRuntime(store=store, embedder=embedder, mode=SearchMode.VECTOR)

    assert isinstance(runtime.indexer, Indexer)
    assert isinstance(runtime.retriever, VectorRetriever)
    asyncio.run(runtime.close())
    asyncio.run(runtime.close())

    assert store.closed == 1
    assert embedder.closed == 1


def test_keyword_runtime_requires_no_embedder():
    runtime = RetrievalRuntime(store=CloseableStore(), embedder=None, mode=SearchMode.KEYWORD)

    assert isinstance(runtime.retriever, KeywordRetriever)
    assert [document.content for document in asyncio.run(runtime.retrieve("q"))] == ["keyword:q"]
    asyncio.run(runtime.close())


def test_vector_runtime_requires_embedder():
    with pytest.raises(ConfigurationError, match="embedder"):
        RetrievalRuntime(store=CloseableStore(), embedder=None, mode=SearchMode.VECTOR)


def test_hybrid_runtime_uses_retriever_alpha_validation():
    with pytest.raises(ValueError, match="alpha"):
        RetrievalRuntime(store=CloseableStore(), embedder=CloseableEmbedder(), mode="hybrid", alpha=True)


def test_runtime_rejects_unsupported_mode_through_configuration_contract():
    with pytest.raises(ConfigurationError, match="Unsupported retrieval mode"):
        RetrievalRuntime(store=CloseableStore(), embedder=None, mode="semantic")


def test_runtime_delegates_index_and_retrieval_without_changing_shapes_or_kwargs():
    store = CloseableStore()
    runtime = RetrievalRuntime(store=store, embedder=None, mode="keyword")
    filters = {"language": "en"}

    written = asyncio.run(runtime.index([{"content": "document"}], batch_size=1, max_parallel_writes=1))
    result = asyncio.run(runtime.retrieve("query", top_k=2, filters=filters))
    results = asyncio.run(runtime.retrieve_many(["one", "two"], top_k=3, filters=filters))

    assert written == 1
    assert [document.content for document in store.written[0]] == ["document"]
    assert [document.content for document in result] == ["keyword:query"]
    assert [[document.content for document in row] for row in results] == [["keyword:one"], ["keyword:two"]]
    assert store.calls == [
        ("keyword", ["query"], {"top_k": 2, "filters": filters}),
        ("keyword", ["one", "two"], {"top_k": 3, "filters": filters}),
    ]
    assert store.calls[0][2]["filters"] is filters


def test_hybrid_runtime_selects_hybrid_retriever_and_forwards_alpha():
    store = CloseableStore()
    runtime = RetrievalRuntime(store=store, embedder=CloseableEmbedder(), mode="hybrid", alpha=0.3)

    assert isinstance(runtime.retriever, HybridRetriever)
    assert [document.content for document in asyncio.run(runtime.retrieve("q", top_k=4))] == ["hybrid:q"]

    assert store.calls == [
        ("hybrid", ["q"], [[0.0, 1.0]], {"alpha": 0.3, "top_k": 4, "filters": None, "include_vectors": False})
    ]


class FailingCloseEmbedder(CloseableEmbedder):
    async def close(self):
        self.closed += 1
        raise RuntimeError("embedder close failed")


class FailingCloseStore(CloseableStore):
    async def close(self):
        self.closed += 1
        raise RuntimeError("store close failed")


def test_runtime_closes_store_when_embedder_close_fails_and_retains_later_failure():
    store = FailingCloseStore()
    embedder = FailingCloseEmbedder()
    runtime = RetrievalRuntime(store=store, embedder=embedder, mode=SearchMode.VECTOR)

    with pytest.raises(RuntimeError, match="embedder close failed") as exc_info:
        asyncio.run(runtime.close())

    assert embedder.closed == 1
    assert store.closed == 1
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "store close failed"


def test_runtime_closes_shared_store_and_embedder_once():
    dependency = DualProtocolCloseable()
    runtime = RetrievalRuntime(store=dependency, embedder=dependency, mode="vector")

    asyncio.run(runtime.close())

    assert dependency.closed == 1


def test_runtime_reraises_shared_dependency_close_failure_without_retrying_close():
    class FailingDualProtocolCloseable(DualProtocolCloseable):
        async def close(self):
            self.closed += 1
            raise RuntimeError("shared close failed")

    dependency = FailingDualProtocolCloseable()
    runtime = RetrievalRuntime(store=dependency, embedder=dependency, mode="vector")

    with pytest.raises(RuntimeError, match="shared close failed"):
        asyncio.run(runtime.close())

    assert dependency.closed == 1


def test_runtime_rejects_operations_after_close():
    runtime = RetrievalRuntime(store=CloseableStore(), embedder=None, mode="keyword")
    asyncio.run(runtime.close())

    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(runtime.retrieve("q"))


def test_runtime_waits_for_active_operation_before_closing_resources():
    class BlockingStore(CloseableStore):
        def __init__(self):
            super().__init__()
            self.search_started = asyncio.Event()
            self.release_search = asyncio.Event()
            self.search_finished = asyncio.Event()

        async def search_keywords(self, queries, **kwargs):
            self.search_started.set()
            await self.release_search.wait()
            self.search_finished.set()
            return await super().search_keywords(queries, **kwargs)

        async def close(self):
            assert self.search_finished.is_set()
            await super().close()

    async def exercise():
        store = BlockingStore()
        runtime = RetrievalRuntime(store=store, embedder=None, mode="keyword")
        retrieve_task = asyncio.create_task(runtime.retrieve("q"))
        await asyncio.wait_for(store.search_started.wait(), timeout=1)
        close_task = asyncio.create_task(runtime.close())
        await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="closed"):
            await runtime.retrieve("later")
        assert store.closed == 0

        store.release_search.set()
        assert [document.content for document in await retrieve_task] == ["keyword:q"]
        await close_task
        assert store.closed == 1

    asyncio.run(exercise())


def test_runtime_close_is_concurrent_and_cancellation_safe():
    class BlockingCloseEmbedder(CloseableEmbedder):
        def __init__(self):
            super().__init__()
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()

        async def close(self):
            self.closed += 1
            self.close_started.set()
            await self.release_close.wait()

    async def exercise():
        store = CloseableStore()
        embedder = BlockingCloseEmbedder()
        runtime = RetrievalRuntime(store=store, embedder=embedder, mode="vector")
        first_close = asyncio.create_task(runtime.close())
        await asyncio.wait_for(embedder.close_started.wait(), timeout=1)
        second_close = asyncio.create_task(runtime.close())
        await asyncio.sleep(0)
        assert not second_close.done()

        first_close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_close

        embedder.release_close.set()
        await second_close
        await runtime.close()
        assert embedder.closed == 1
        assert store.closed == 1

    asyncio.run(exercise())


def test_runtime_context_manager_preserves_body_exception_when_close_succeeds():
    async def exercise():
        store = CloseableStore()
        embedder = CloseableEmbedder()
        runtime = RetrievalRuntime(store=store, embedder=embedder, mode="vector")

        with pytest.raises(ValueError, match="body failed"):
            async with runtime:
                raise ValueError("body failed")

        assert store.closed == 1
        assert embedder.closed == 1

    asyncio.run(exercise())


def test_runtime_context_manager_raises_close_failure_over_body_exception():
    async def exercise():
        runtime = RetrievalRuntime(store=CloseableStore(), embedder=FailingCloseEmbedder(), mode="vector")

        with pytest.raises(RuntimeError, match="embedder close failed") as exc_info:
            async with runtime:
                raise ValueError("body failed")

        assert isinstance(exc_info.value.__context__, ValueError)

    asyncio.run(exercise())
