import asyncio
import math
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from justatom.etc.schema import Document
from justatom.retrieval import runtime as runtime_module
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


def _patch_store_connect(monkeypatch, connect):
    store_class = getattr(runtime_module, "WeaviateDocumentStore", None)
    if store_class is None:
        store_class = type("WeaviateDocumentStore", (), {"connect": staticmethod(connect)})
        monkeypatch.setattr(runtime_module, "WeaviateDocumentStore", store_class, raising=False)
    else:
        monkeypatch.setattr(store_class, "connect", connect)


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

    assert store.calls == [("hybrid", ["q"], [[0.0, 1.0]], {"alpha": 0.3, "top_k": 4, "filters": None, "include_vectors": False})]


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


def test_builder_rejects_unknown_keys_before_opening_resources(monkeypatch):
    opened = []

    async def fake_connect(*args, **kwargs):
        opened.append((args, kwargs))

    _patch_store_connect(monkeypatch, fake_connect)

    with pytest.raises(ConfigurationError, match="unknown retrieval keys"):
        asyncio.run(runtime_module.build_runtime({"mode": "keyword", "unknown": True, "store": {"collection": "Docs"}}))

    assert opened == []


def test_keyword_builder_never_constructs_an_embedder(monkeypatch):
    store = CloseableStore()

    async def fake_connect(*args, **kwargs):
        return store

    _patch_store_connect(monkeypatch, fake_connect)
    monkeypatch.setattr(
        runtime_module,
        "HuggingFaceEmbedder",
        lambda **kwargs: pytest.fail("local embedder constructed"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_module,
        "OpenAICompatibleEmbedder",
        lambda **kwargs: pytest.fail("remote embedder constructed"),
        raising=False,
    )

    runtime = asyncio.run(runtime_module.build_runtime({"mode": "keyword", "store": {"collection": "Docs"}}))

    assert runtime.embedder is None
    asyncio.run(runtime.close())


def test_keyword_builder_with_fake_weaviate_connect_does_not_import_torch():
    root = Path(__file__).resolve().parents[2]
    script = """
import asyncio
import sys

from justatom.retrieval import runtime as runtime_module
from justatom.storing.weaviate import WeaviateDocumentStore

class Store:
    async def close(self):
        pass

class FakeWeaviateDocumentStore:
    @classmethod
    async def connect(cls, *args, **kwargs):
        return Store()

runtime_module.WeaviateDocumentStore = FakeWeaviateDocumentStore
runtime = asyncio.run(runtime_module.build_runtime({"mode": "keyword", "store": {"collection": "Docs"}}))
asyncio.run(runtime.close())
assert "torch" not in sys.modules
"""

    completed = subprocess.run([sys.executable, "-c", script], cwd=root, check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr


def test_builder_closes_embedder_when_store_connection_fails(monkeypatch):
    embedder = CloseableEmbedder()
    monkeypatch.setattr(runtime_module, "HuggingFaceEmbedder", lambda **kwargs: embedder, raising=False)

    async def failed_connect(*args, **kwargs):
        raise RuntimeError("weaviate unavailable")

    _patch_store_connect(monkeypatch, failed_connect)
    config = {
        "mode": "vector",
        "embedding": {"backend": "local", "model": "local-model"},
        "store": {"collection": "Docs"},
    }

    with pytest.raises(RuntimeError, match="unavailable"):
        asyncio.run(runtime_module.build_runtime(config))

    assert embedder.closed == 1


def test_builder_maps_local_and_remote_embedding_config(monkeypatch):
    captures = []

    def fake_local(**kwargs):
        captures.append(("local", kwargs))
        return CloseableEmbedder()

    def fake_remote(**kwargs):
        captures.append(("remote", kwargs))
        return CloseableEmbedder()

    async def fake_connect(*args, **kwargs):
        return CloseableStore()

    monkeypatch.setattr(runtime_module, "HuggingFaceEmbedder", fake_local, raising=False)
    monkeypatch.setattr(runtime_module, "OpenAICompatibleEmbedder", fake_remote, raising=False)
    _patch_store_connect(monkeypatch, fake_connect)

    local = asyncio.run(
        runtime_module.build_runtime(
            {
                "mode": "vector",
                "embedding": {
                    "backend": "local",
                    "model": "local-model",
                    "device": "mps",
                    "query_prefix": "q: ",
                    "document_prefix": "d: ",
                },
                "store": {"collection": "LocalDocs"},
            }
        )
    )
    remote = asyncio.run(
        runtime_module.build_runtime(
            {
                "mode": "hybrid",
                "alpha": 0.3,
                "embedding": {
                    "backend": "openai-compatible",
                    "base_url": "http://encoder/v1",
                    "model": "remote-model",
                    "api_key": "key",
                    "timeout": 12,
                    "encoding_format": "float",
                    "extra_body": {"pooling": "mean"},
                },
                "store": {"collection": "RemoteDocs"},
            }
        )
    )

    assert captures[0][0] == "local"
    assert captures[0][1]["model"] == "local-model"
    assert captures[0][1]["device"] == "mps"
    assert captures[0][1]["profile"].query_prefix == "q: "
    assert captures[1][0] == "remote"
    assert captures[1][1]["base_url"] == "http://encoder/v1"
    assert captures[1][1]["extra_body"] == {"pooling": "mean"}
    asyncio.run(local.close())
    asyncio.run(remote.close())


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ([], "retrieval must be a mapping"),
        ({"mode": "keyword", "store": {"collection": "Docs", "unknown": True}}, "unknown store keys"),
        (
            {
                "mode": "vector",
                "embedding": {"backend": "local", "model": "model", "base_url": "http://encoder"},
                "store": {"collection": "Docs"},
            },
            "unknown local embedding keys",
        ),
        (
            {
                "mode": "vector",
                "embedding": {"backend": "openai-compatible", "base_url": "http://encoder", "model": "model", "device": "cpu"},
                "store": {"collection": "Docs"},
            },
            "unknown openai-compatible embedding keys",
        ),
        ({"mode": "keyword", "store": []}, "store must be a mapping"),
        ({"mode": "vector", "embedding": [], "store": {"collection": "Docs"}}, "embedding must be a mapping"),
        ({"mode": "keyword", "store": {"collection": ""}}, "collection must be a non-empty string"),
        ({"mode": "vector", "store": {"collection": "Docs"}}, "embedding is required"),
        (
            {"mode": "vector", "embedding": {"backend": "local", "model": ""}, "store": {"collection": "Docs"}},
            "model must be a non-empty string",
        ),
        (
            {
                "mode": "vector",
                "embedding": {"backend": "openai-compatible", "model": "model"},
                "store": {"collection": "Docs"},
            },
            "base_url is required",
        ),
        (
            {"mode": "vector", "embedding": {"backend": "remote", "model": "model"}, "store": {"collection": "Docs"}},
            "backend must be 'local' or 'openai-compatible'",
        ),
    ],
)
def test_builder_validates_exact_config_shape_before_opening_resources(monkeypatch, config, message):
    async def unexpected_connect(*args, **kwargs):
        pytest.fail("store connected before configuration validation")

    _patch_store_connect(monkeypatch, unexpected_connect)

    with pytest.raises(ConfigurationError, match=message):
        asyncio.run(runtime_module.build_runtime(config))


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"mode": True, "store": {"collection": "Docs"}}, "mode must be a non-empty string"),
        ({"mode": "invalid", "store": {"collection": "Docs"}}, "Unsupported retrieval mode"),
        ({"mode": "keyword", "alpha": True, "store": {"collection": "Docs"}}, "alpha must be a finite numeric value"),
        ({"mode": "keyword", "alpha": math.inf, "store": {"collection": "Docs"}}, "alpha must be a finite numeric value"),
        ({"mode": "keyword", "alpha": -0.1, "store": {"collection": "Docs"}}, "alpha must be a finite numeric value"),
        ({"mode": "keyword", "store": {"collection": "Docs", "grpc_port": True}}, "grpc_port must be an integer"),
        ({"mode": "keyword", "store": {"collection": "Docs", "grpc_port": 0}}, "grpc_port must be in"),
        ({"mode": "keyword", "store": {"collection": "Docs", "grpc_secure": 1}}, "grpc_secure must be a boolean"),
        (
            {
                "mode": "vector",
                "embedding": {"backend": "local", "model": "model", "batch_size": True},
                "store": {"collection": "Docs"},
            },
            "batch_size must be a positive integer",
        ),
        (
            {
                "mode": "vector",
                "embedding": {"backend": "local", "model": "model", "max_length": 0},
                "store": {"collection": "Docs"},
            },
            "max_length must be a positive integer",
        ),
        (
            {
                "mode": "vector",
                "embedding": {
                    "backend": "openai-compatible",
                    "base_url": "http://encoder",
                    "model": "model",
                    "timeout": True,
                },
                "store": {"collection": "Docs"},
            },
            "timeout must be a positive finite number",
        ),
        (
            {
                "mode": "vector",
                "embedding": {
                    "backend": "openai-compatible",
                    "base_url": "http://encoder",
                    "model": "model",
                    "extra_body": [],
                },
                "store": {"collection": "Docs"},
            },
            "extra_body must be a mapping",
        ),
    ],
)
def test_builder_rejects_invalid_scalar_config_before_opening_resources(monkeypatch, config, message):
    async def unexpected_connect(*args, **kwargs):
        pytest.fail("store connected before scalar validation")

    _patch_store_connect(monkeypatch, unexpected_connect)

    with pytest.raises(ConfigurationError, match=message):
        asyncio.run(runtime_module.build_runtime(config))


def test_keyword_builder_validates_supplied_embedding_without_constructing_it(monkeypatch):
    async def unexpected_connect(*args, **kwargs):
        pytest.fail("store connected before embedding validation")

    _patch_store_connect(monkeypatch, unexpected_connect)
    monkeypatch.setattr(
        runtime_module,
        "HuggingFaceEmbedder",
        lambda **kwargs: pytest.fail("embedder constructed"),
        raising=False,
    )

    with pytest.raises(ConfigurationError, match="backend must be 'local' or 'openai-compatible'"):
        asyncio.run(
            runtime_module.build_runtime(
                {
                    "mode": "keyword",
                    "embedding": {"backend": "invalid", "model": "model"},
                    "store": {"collection": "Docs"},
                }
            )
        )


def test_builder_passes_typed_store_values_and_copies_extra_body(monkeypatch):
    store = CloseableStore()
    captured = {}
    extra_body = {"options": {"pooling": "mean"}}

    def fake_remote(**kwargs):
        captured["embedder"] = kwargs
        return CloseableEmbedder()

    async def fake_connect(*args, **kwargs):
        captured["store"] = (args, kwargs)
        return store

    monkeypatch.setattr(runtime_module, "OpenAICompatibleEmbedder", fake_remote, raising=False)
    _patch_store_connect(monkeypatch, fake_connect)
    runtime = asyncio.run(
        runtime_module.build_runtime(
            {
                "mode": "vector",
                "embedding": {
                    "backend": "openai-compatible",
                    "base_url": "http://encoder/v1",
                    "model": "model",
                    "extra_body": extra_body,
                },
                "store": {"collection": "Docs", "grpc_port": 7443, "grpc_secure": True},
            }
        )
    )

    assert captured["store"] == (("Docs",), {"url": None, "grpc_port": 7443, "grpc_secure": True})
    assert captured["embedder"]["extra_body"] == extra_body
    assert captured["embedder"]["extra_body"] is not extra_body
    extra_body["options"]["pooling"] = "max"
    assert captured["embedder"]["extra_body"] == {"options": {"pooling": "mean"}}
    assert runtime.store is store
    asyncio.run(runtime.close())


def test_builder_closes_store_and_embedder_when_runtime_construction_fails(monkeypatch):
    store = CloseableStore()
    embedder = CloseableEmbedder()

    def failed_runtime(*args, **kwargs):
        raise RuntimeError("runtime construction failed")

    async def fake_connect(*args, **kwargs):
        return store

    monkeypatch.setattr(runtime_module, "HuggingFaceEmbedder", lambda **kwargs: embedder, raising=False)
    _patch_store_connect(monkeypatch, fake_connect)
    monkeypatch.setattr(runtime_module, "RetrievalRuntime", failed_runtime)

    with pytest.raises(RuntimeError, match="runtime construction failed"):
        asyncio.run(
            runtime_module.build_runtime(
                {
                    "mode": "vector",
                    "embedding": {"backend": "local", "model": "model"},
                    "store": {"collection": "Docs"},
                }
            )
        )

    assert embedder.closed == 1
    assert store.closed == 1


def test_builder_closes_shared_store_and_embedder_once_when_runtime_construction_fails(monkeypatch):
    dependency = DualProtocolCloseable()

    def failed_runtime(*args, **kwargs):
        raise RuntimeError("runtime construction failed")

    async def fake_connect(*args, **kwargs):
        return dependency

    monkeypatch.setattr(runtime_module, "HuggingFaceEmbedder", lambda **kwargs: dependency, raising=False)
    _patch_store_connect(monkeypatch, fake_connect)
    monkeypatch.setattr(runtime_module, "RetrievalRuntime", failed_runtime)

    with pytest.raises(RuntimeError, match="runtime construction failed"):
        asyncio.run(
            runtime_module.build_runtime(
                {
                    "mode": "vector",
                    "embedding": {"backend": "local", "model": "model"},
                    "store": {"collection": "Docs"},
                }
            )
        )

    assert dependency.closed == 1


def test_builder_preserves_connection_failure_when_embedder_cleanup_fails(monkeypatch):
    class FailingCloseEmbedder(CloseableEmbedder):
        async def close(self):
            await super().close()
            raise RuntimeError("embedder cleanup failed")

    embedder = FailingCloseEmbedder()

    async def failed_connect(*args, **kwargs):
        raise RuntimeError("weaviate unavailable")

    monkeypatch.setattr(runtime_module, "HuggingFaceEmbedder", lambda **kwargs: embedder, raising=False)
    _patch_store_connect(monkeypatch, failed_connect)

    with pytest.raises(RuntimeError, match="weaviate unavailable") as exc_info:
        asyncio.run(
            runtime_module.build_runtime(
                {
                    "mode": "vector",
                    "embedding": {"backend": "local", "model": "model"},
                    "store": {"collection": "Docs"},
                }
            )
        )

    assert embedder.closed == 1
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "embedder cleanup failed"


def test_builder_finishes_failure_cleanup_when_cancelled(monkeypatch):
    class BlockingCloseEmbedder(CloseableEmbedder):
        def __init__(self):
            super().__init__()
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()

        async def close(self):
            self.closed += 1
            self.close_started.set()
            await self.release_close.wait()
            raise RuntimeError("embedder cleanup failed")

    async def exercise():
        embedder = BlockingCloseEmbedder()

        async def failed_connect(*args, **kwargs):
            raise RuntimeError("weaviate unavailable")

        monkeypatch.setattr(runtime_module, "HuggingFaceEmbedder", lambda **kwargs: embedder, raising=False)
        _patch_store_connect(monkeypatch, failed_connect)
        task = asyncio.create_task(
            runtime_module.build_runtime(
                {
                    "mode": "vector",
                    "embedding": {"backend": "local", "model": "model"},
                    "store": {"collection": "Docs"},
                }
            )
        )
        await asyncio.wait_for(embedder.close_started.wait(), timeout=1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()

        embedder.release_close.set()
        with pytest.raises(RuntimeError, match="embedder cleanup failed") as exc_info:
            await task
        assert embedder.closed == 1
        assert isinstance(exc_info.value.__cause__, asyncio.CancelledError)

    asyncio.run(exercise())
