import asyncio
import base64
import copy
import inspect
from types import SimpleNamespace
from uuid import uuid4

import pytest
from weaviate.collections.classes.batch import DeleteManyReturn
from weaviate.collections.classes.internal import MetadataReturn, Object

from justatom.etc.errors import DocumentStoreError
from justatom.etc.schema import Document
from justatom.retrieval.contracts import DocumentStore
from justatom.storing import weaviate as weaviate_module
from justatom.storing.weaviate import WeaviateDocumentStore


class FakeQuery:
    def __init__(self):
        self.calls = []

    async def bm25(self, **kwargs):
        self.calls.append(("bm25", kwargs))
        return SimpleNamespace(objects=[kwargs["query"]])

    async def hybrid(self, **kwargs):
        self.calls.append(("hybrid", kwargs))
        return SimpleNamespace(objects=[kwargs["query"]])

    async def near_vector(self, **kwargs):
        self.calls.append(("near_vector", kwargs))
        return SimpleNamespace(objects=[kwargs["near_vector"]])


class FakeCollection:
    def __init__(self, name):
        self.name = name
        self.query = FakeQuery()


class FakeCollections:
    def __init__(self):
        self.collection = None
        self.created = []

    async def exists(self, name):
        self.collection = FakeCollection(name)
        return True

    async def create_from_dict(self, settings):
        self.created.append(settings)

    def get(self, name):
        assert self.collection is not None
        assert self.collection.name == name
        return self.collection


class FakeAsyncClient:
    instances = []
    connect_error = None

    def __init__(self, **kwargs):
        self.constructor_options = kwargs
        self.collections = FakeCollections()
        self.connected = False
        self.close_calls = 0
        type(self).instances.append(self)

    async def connect(self):
        if self.connect_error is not None:
            raise self.connect_error
        self.connected = True

    def is_connected(self):
        return self.connected

    async def close(self):
        self.close_calls += 1
        self.connected = False


@pytest.fixture
def fake_client(monkeypatch):
    FakeAsyncClient.instances = []
    FakeAsyncClient.connect_error = None
    monkeypatch.setattr(weaviate_module.weaviate, "WeaviateAsyncClient", FakeAsyncClient)
    return FakeAsyncClient


def test_weaviate_store_exposes_protocol_vocabulary():
    assert weaviate_module.__all__ == ["WeaviateDocumentStore"]
    assert not hasattr(weaviate_module, "WeaviateDocStore")
    assert not hasattr(weaviate_module, "IFinder")
    assert not hasattr(weaviate_module, "Finder")
    assert inspect.iscoroutinefunction(WeaviateDocumentStore.search_vector)
    assert inspect.iscoroutinefunction(WeaviateDocumentStore.search_keywords)
    assert inspect.iscoroutinefunction(WeaviateDocumentStore.search_hybrid)
    assert inspect.iscoroutinefunction(WeaviateDocumentStore.clear)
    assert not hasattr(WeaviateDocumentStore, "search_by_keywords_sync")


def test_weaviate_store_satisfies_document_store_protocol_structurally():
    store = object.__new__(WeaviateDocumentStore)
    assert isinstance(store, DocumentStore)


def test_weaviate_store_uses_protocol_signatures():
    assert list(inspect.signature(WeaviateDocumentStore.connect).parameters) == [
        "collection",
        "url",
        "grpc_port",
        "grpc_secure",
        "client_options",
    ]
    assert list(inspect.signature(WeaviateDocumentStore.search_vector).parameters) == [
        "self",
        "vectors",
        "top_k",
        "filters",
        "include_vectors",
    ]
    assert list(inspect.signature(WeaviateDocumentStore.search_keywords).parameters) == [
        "self",
        "queries",
        "top_k",
        "filters",
    ]
    assert list(inspect.signature(WeaviateDocumentStore.search_hybrid).parameters) == [
        "self",
        "queries",
        "vectors",
        "alpha",
        "top_k",
        "filters",
        "include_vectors",
    ]
    assert list(inspect.signature(WeaviateDocumentStore.write_documents).parameters) == [
        "self",
        "documents",
    ]


def test_connect_parses_explicit_secure_url_and_forwards_only_client_options(fake_client, monkeypatch):
    monkeypatch.setenv("WEAVIATE_HOST", "ignored.example")
    monkeypatch.setenv("WEAVIATE_PORT", "not-a-port")

    store = asyncio.run(
        WeaviateDocumentStore.connect(
            "documents",
            url="https://weaviate.example:8443",
            grpc_port=7443,
            grpc_secure=False,
            additional_headers={"X-API-Key": "secret"},
            skip_init_checks=True,
        )
    )

    assert len(fake_client.instances) == 1
    client = fake_client.instances[0]
    params = client.constructor_options.pop("connection_params")
    assert (params.http.host, params.http.port, params.http.secure) == ("weaviate.example", 8443, True)
    assert (params.grpc.host, params.grpc.port, params.grpc.secure) == ("weaviate.example", 7443, True)
    assert client.constructor_options == {
        "additional_headers": {"X-API-Key": "secret"},
        "skip_init_checks": True,
    }
    asyncio.run(store.close())


def test_connect_uses_environment_host_and_port_only_without_url(fake_client, monkeypatch):
    monkeypatch.setenv("WEAVIATE_HOST", "weaviate.internal")
    monkeypatch.setenv("WEAVIATE_PORT", "8088")

    store = asyncio.run(WeaviateDocumentStore.connect("documents", grpc_port=50052, grpc_secure=True))

    params = fake_client.instances[0].constructor_options["connection_params"]
    assert (params.http.host, params.http.port, params.http.secure) == ("weaviate.internal", 8088, False)
    assert (params.grpc.host, params.grpc.port, params.grpc.secure) == ("weaviate.internal", 50052, True)
    asyncio.run(store.close())


def test_connect_forwards_native_weaviate_auth_option_unchanged(fake_client):
    auth = object()

    store = asyncio.run(
        WeaviateDocumentStore.connect(
            "documents",
            url="http://localhost:2211",
            auth_client_secret=auth,
        )
    )

    assert fake_client.instances[0].constructor_options["auth_client_secret"] is auth
    asyncio.run(store.close())


def test_connect_rejects_conflicting_connection_params_before_constructing_client(fake_client):
    with pytest.raises(DocumentStoreError, match="connection_params"):
        asyncio.run(
            WeaviateDocumentStore.connect(
                "documents",
                url="http://localhost:2211",
                connection_params=object(),
            )
        )
    assert fake_client.instances == []


def test_connect_closes_async_client_and_retains_cause_on_connect_failure(fake_client):
    failure = RuntimeError("offline")
    fake_client.connect_error = failure

    with pytest.raises(DocumentStoreError, match="Failed to connect") as exc_info:
        asyncio.run(WeaviateDocumentStore.connect("documents", url="http://localhost:2211"))

    assert exc_info.value.__cause__ is failure
    assert len(fake_client.instances) == 1
    assert fake_client.instances[0].close_calls == 1


@pytest.mark.parametrize(
    "url",
    [
        "https://user:secret@weaviate.example:8443",
        "https://weaviate.example:8443/v1",
        "https://weaviate.example:8443?debug=true",
        "https://weaviate.example:8443#fragment",
    ],
)
def test_connect_rejects_unsafe_url_components_without_logging_them(fake_client, monkeypatch, url):
    logs = []
    monkeypatch.setattr(weaviate_module.logger, "info", lambda message, *args: logs.append((message, args)))

    with pytest.raises(DocumentStoreError, match="URL") as exc_info:
        asyncio.run(WeaviateDocumentStore.connect("documents", url=url))

    assert "secret" not in str(exc_info.value)
    assert logs == []
    assert fake_client.instances == []


def test_connect_normalizes_root_path_and_does_not_log_the_endpoint(fake_client, monkeypatch):
    logs = []
    monkeypatch.setattr(weaviate_module.logger, "info", lambda message, *args: logs.append((message, args)))

    store = asyncio.run(
        WeaviateDocumentStore.connect(
            "documents",
            url="https://weaviate.example:8443/",
        )
    )

    params = fake_client.instances[0].constructor_options["connection_params"]
    assert (params.http.host, params.http.port, params.http.secure) == ("weaviate.example", 8443, True)
    assert logs == [("WEAVIATE | connecting collection=[{}]", ("documents",))]
    asyncio.run(store.close())


def _sdk_object(*, score=None, certainty=None):
    return Object(
        uuid=uuid4(),
        metadata=MetadataReturn(score=score, certainty=certainty),
        properties={
            "_original_id": "doc-1",
            "content": "document text",
            "blob_data": base64.b64encode(b"payload").decode(),
            "blob_mime_type": "text/plain",
            "meta": {"labels": ["one"], "nested": {"value": 1}},
        },
        references=None,
        vector={"default": [0.1, 0.2]},
        collection="Documents",
    )


def test_document_conversion_is_repeatable_and_does_not_mutate_sdk_object():
    store = object.__new__(WeaviateDocumentStore)
    source = _sdk_object(score=0.75, certainty=0.5)
    original_properties = copy.deepcopy(source.properties)

    first = store._to_document(source)
    second = store._to_document(source)

    assert first == second
    assert first.id == "doc-1"
    assert first.embedding == [0.1, 0.2]
    assert source.properties == original_properties


def test_write_conversion_does_not_mutate_document_blob():
    store = object.__new__(WeaviateDocumentStore)
    document = Document(content="document text", id="doc-1", embedding=[0.1, 0.2])
    document.blob = {"data": bytearray(b"payload"), "mime_type": "text/plain"}
    original_blob = copy.deepcopy(document.blob)

    converted = store._to_data_object(document)

    assert converted["blob_data"] == base64.b64encode(b"payload").decode()
    assert converted["blob_mime_type"] == "text/plain"
    assert document.blob == original_blob


@pytest.mark.parametrize(
    ("score", "certainty", "expected"),
    [
        (0.0, None, 0.0),
        (None, 0.0, 0.0),
        (0.4, 0.8, 0.4),
    ],
    ids=["bm25-zero-score", "vector-zero-certainty", "hybrid-score-precedence"],
)
def test_document_conversion_uses_score_then_certainty(score, certainty, expected):
    store = object.__new__(WeaviateDocumentStore)

    document = store._to_document(_sdk_object(score=score, certainty=certainty))

    assert document.score == expected


class AsyncObjectIterator:
    def __init__(self, objects=(), error=None):
        self._objects = iter(objects)
        self._error = error

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        try:
            return next(self._objects)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def test_filter_documents_consumes_async_iterator_without_awaiting_it():
    store = object.__new__(WeaviateDocumentStore)
    source = _sdk_object()
    calls = []

    def iterator(**kwargs):
        calls.append(kwargs)
        return AsyncObjectIterator([source])

    store._WeaviateDocumentStore__collection = SimpleNamespace(iterator=iterator)

    documents = asyncio.run(store.filter_documents())

    assert [document.id for document in documents] == ["doc-1"]
    assert calls == [{"include_vector": True, "return_properties": None}]


def test_filter_documents_wraps_weaviate_query_failure_with_cause():
    store = object.__new__(WeaviateDocumentStore)
    failure = weaviate_module.weaviate.exceptions.WeaviateQueryError("offline", "GRPC search")
    store._WeaviateDocumentStore__collection = SimpleNamespace(
        iterator=lambda **kwargs: AsyncObjectIterator(error=failure)
    )

    with pytest.raises(DocumentStoreError, match="Failed to query") as exc_info:
        asyncio.run(store.filter_documents())
    assert exc_info.value.__cause__ is failure


def test_search_methods_preserve_query_groups_and_include_vectors(fake_client):
    store = asyncio.run(WeaviateDocumentStore.connect("documents", url="http://localhost:2211"))
    store._to_document = lambda value: value
    collection = fake_client.instances[0].collections.collection

    vector_results = asyncio.run(store.search_vector([[1.0], [2.0]], top_k=3, include_vectors=True))
    keyword_results = asyncio.run(store.search_keywords(["first", "second"], top_k=4))
    hybrid_results = asyncio.run(
        store.search_hybrid(
            ["first", "second"],
            [[1.0], [2.0]],
            alpha=0.25,
            top_k=5,
            include_vectors=True,
        )
    )

    assert vector_results == [[[1.0]], [[2.0]]]
    assert keyword_results == [["first"], ["second"]]
    assert hybrid_results == [["first"], ["second"]]
    vector_calls = [options for method, options in collection.query.calls if method == "near_vector"]
    keyword_calls = [options for method, options in collection.query.calls if method == "bm25"]
    hybrid_calls = [options for method, options in collection.query.calls if method == "hybrid"]
    assert [options["include_vector"] for options in vector_calls] == [True, True]
    assert [options["include_vector"] for options in keyword_calls] == [False, False]
    assert [options["include_vector"] for options in hybrid_calls] == [True, True]
    asyncio.run(store.close())


def test_search_methods_return_no_groups_for_empty_batches(fake_client):
    store = asyncio.run(WeaviateDocumentStore.connect("documents", url="http://localhost:2211"))
    collection = fake_client.instances[0].collections.collection

    assert asyncio.run(store.search_vector([], top_k=3)) == []
    assert asyncio.run(store.search_keywords([], top_k=3)) == []
    assert asyncio.run(store.search_hybrid([], [], alpha=0.5, top_k=3)) == []
    assert collection.query.calls == []
    asyncio.run(store.close())


def test_clear_raises_document_store_error_with_deletion_cause():
    failure = RuntimeError("delete failed")
    store = object.__new__(WeaviateDocumentStore)
    store.collection_settings = {"class": "Documents"}
    store._WeaviateDocumentStore__collection = SimpleNamespace(name="Documents")

    async def get_all_documents():
        yield SimpleNamespace(properties={"_original_id": "doc-1"})

    async def delete_documents(document_ids):
        assert document_ids == ["doc-1"]
        raise failure

    async def ensure_connection():
        return None

    store.get_all_documents = get_all_documents
    store.delete_documents = delete_documents
    store._ensure_async_connection = ensure_connection

    with pytest.raises(DocumentStoreError, match="deleting documents") as exc_info:
        asyncio.run(store.clear())
    assert exc_info.value.__cause__ is failure


def test_delete_documents_raises_when_weaviate_reports_failed_objects():
    store = object.__new__(WeaviateDocumentStore)

    async def ensure_connection():
        return None

    async def delete_many(**kwargs):
        return DeleteManyReturn(failed=1, matches=2, objects=None, successful=1)

    store._ensure_async_connection = ensure_connection
    store._WeaviateDocumentStore__collection = SimpleNamespace(data=SimpleNamespace(delete_many=delete_many))

    with pytest.raises(DocumentStoreError, match="1 of 2"):
        asyncio.run(store.delete_documents(["doc-1", "doc-2"]))


def test_clear_wraps_enumeration_failure_with_cause():
    failure = RuntimeError("query failed")
    store = object.__new__(WeaviateDocumentStore)
    store.collection_settings = {"class": "Documents"}

    async def ensure_connection():
        return None

    async def get_all_documents():
        raise failure
        yield

    store._ensure_async_connection = ensure_connection
    store.get_all_documents = get_all_documents

    with pytest.raises(DocumentStoreError, match="deleting documents") as exc_info:
        asyncio.run(store.clear())
    assert exc_info.value.__cause__ is failure


def test_close_failure_propagates_retains_client_and_allows_retry(fake_client):
    store = asyncio.run(WeaviateDocumentStore.connect("documents", url="http://localhost:2211"))
    client = fake_client.instances[0]
    failure = RuntimeError("close failed")

    async def flaky_close():
        client.close_calls += 1
        if client.close_calls == 1:
            raise failure
        client.connected = False

    client.close = flaky_close

    async def close_concurrently():
        return await asyncio.gather(store.close(), store.close(), return_exceptions=True)

    results = asyncio.run(close_concurrently())
    assert results == [failure, failure]
    assert store._client is client

    asyncio.run(store.close())
    assert client.close_calls == 2
    assert store._client is None


def test_close_is_idempotent_and_waits_for_cleanup_when_cancelled(fake_client):
    store = asyncio.run(WeaviateDocumentStore.connect("documents", url="http://localhost:2211"))
    client = fake_client.instances[0]
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def delayed_close():
        client.close_calls += 1
        started.set()
        await release.wait()
        client.connected = False
        finished.set()

    client.close = delayed_close

    async def cancel_during_close():
        close_task = asyncio.create_task(store.close())
        await started.wait()
        close_task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await close_task
        assert finished.is_set()
        await store.close()

    asyncio.run(cancel_during_close())
    assert client.close_calls == 1
