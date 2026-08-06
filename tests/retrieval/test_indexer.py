import asyncio

import pytest

from justatom.etc.errors import DocumentStoreError
from justatom.etc.schema import Document
from justatom.retrieval.errors import ConfigurationError
from justatom.retrieval.indexer import Indexer


class FakeStore:
    def __init__(self):
        self.batches = []

    async def write_documents(self, documents):
        self.batches.append(list(documents))
        return len(documents)


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    async def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class FailingStore(FakeStore):
    async def write_documents(self, documents):
        raise DocumentStoreError("write failed")


def test_indexer_streams_batches_and_skips_embedding_in_keyword_mode():
    seen = []

    def documents():
        for index in range(5):
            seen.append(index)
            yield {"content": f"doc-{index}"}

    store = FakeStore()
    written = asyncio.run(Indexer(store).index(documents(), batch_size=2))

    assert written == 5
    assert seen == [0, 1, 2, 3, 4]
    assert [len(batch) for batch in store.batches] == [2, 2, 1]
    assert all(doc.embedding is None for batch in store.batches for doc in batch)


def test_indexer_attaches_one_embedding_per_document():
    store = FakeStore()
    embedder = FakeEmbedder()

    written = asyncio.run(Indexer(store, embedder).index([Document(content="a"), Document(content="b")]))

    assert written == 2
    assert embedder.calls == [["a", "b"]]
    assert [doc.embedding for doc in store.batches[0]] == [[0.0, 1.0], [1.0, 1.0]]


def test_indexer_copies_documents_before_attaching_embeddings():
    source = Document(content="a")
    store = FakeStore()

    asyncio.run(Indexer(store, FakeEmbedder()).index([source]))

    assert source.embedding is None
    assert store.batches[0][0] is not source
    assert store.batches[0][0].embedding == [0.0, 1.0]


def test_indexer_preserves_write_failure_as_cause():
    with pytest.raises(DocumentStoreError, match="batch 0") as exc_info:
        asyncio.run(Indexer(FailingStore()).index([{"content": "a"}]))

    assert isinstance(exc_info.value.__cause__, DocumentStoreError)
    assert str(exc_info.value.__cause__) == "write failed"


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("batch_size", 0),
        ("batch_size", -1),
        ("batch_size", True),
        ("batch_size", 1.5),
        ("batch_size", "1"),
        ("batch_size", None),
        ("max_parallel_writes", 0),
        ("max_parallel_writes", -1),
        ("max_parallel_writes", True),
        ("max_parallel_writes", 1.5),
        ("max_parallel_writes", "1"),
        ("max_parallel_writes", None),
    ],
)
def test_indexer_rejects_non_positive_boolean_or_non_integer_batch_settings(parameter, value):
    kwargs = {parameter: value}

    with pytest.raises(ConfigurationError, match=parameter):
        asyncio.run(Indexer(FakeStore()).index([], **kwargs))


def test_indexer_reports_later_failed_batch_index_and_preserves_its_cause():
    class LaterFailingStore:
        def __init__(self):
            self.batch_zero_started = asyncio.Event()
            self.batch_zero_may_complete = asyncio.Event()
            self.batch_zero_completed = asyncio.Event()
            self.batch_one_started = asyncio.Event()
            self.batch_one_may_fail = asyncio.Event()

        async def write_documents(self, documents):
            if documents[0].content == "doc-0":
                self.batch_zero_started.set()
                await self.batch_zero_may_complete.wait()
                self.batch_zero_completed.set()
                return 1
            if documents[0].content == "doc-1":
                self.batch_one_started.set()
                await self.batch_one_may_fail.wait()
                raise DocumentStoreError("batch one failed")
            raise AssertionError("Unexpected batch")

    async def exercise():
        store = LaterFailingStore()
        index_task = asyncio.create_task(
            Indexer(store).index(
                [{"content": "doc-0"}, {"content": "doc-1"}],
                batch_size=1,
                max_parallel_writes=2,
            )
        )
        await asyncio.wait_for(asyncio.gather(store.batch_zero_started.wait(), store.batch_one_started.wait()), timeout=1)
        store.batch_zero_may_complete.set()
        assert await asyncio.wait_for(store.batch_zero_completed.wait(), timeout=1)
        store.batch_one_may_fail.set()

        with pytest.raises(DocumentStoreError, match="batch 1") as exc_info:
            await index_task
        assert isinstance(exc_info.value.__cause__, DocumentStoreError)
        assert str(exc_info.value.__cause__) == "batch one failed"

    asyncio.run(exercise())


def test_indexer_cancels_pending_writes_and_stops_stream_consumption_after_failure():
    seen = []

    class ConcurrentFailingStore:
        def __init__(self):
            self.active_writes = 0
            self.maximum_active_writes = 0
            self.cancelled_write_cleaned_up = asyncio.Event()

        async def write_documents(self, documents):
            document = documents[0]
            self.active_writes += 1
            self.maximum_active_writes = max(self.maximum_active_writes, self.active_writes)
            try:
                if document.content == "doc-0":
                    raise DocumentStoreError("write failed")
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled_write_cleaned_up.set()
                raise
            finally:
                self.active_writes -= 1

    def documents():
        for index in range(4):
            seen.append(index)
            yield {"content": f"doc-{index}"}

    async def exercise():
        store = ConcurrentFailingStore()
        with pytest.raises(DocumentStoreError, match="batch 0") as exc_info:
            await Indexer(store).index(documents(), batch_size=1, max_parallel_writes=2)
        assert isinstance(exc_info.value.__cause__, DocumentStoreError)
        assert await asyncio.wait_for(store.cancelled_write_cleaned_up.wait(), timeout=1)
        assert store.maximum_active_writes <= 2

    asyncio.run(exercise())
    assert seen == [0, 1]
