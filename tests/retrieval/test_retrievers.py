import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from justatom.etc.schema import Document
from justatom.retrieval.errors import RetrievalError
from justatom.retrieval.retriever import HybridRetriever, KeywordRetriever, VectorRetriever


class FakeStore:
    def __init__(self):
        self.calls = []

    async def search_keywords(self, queries, **kwargs):
        self.calls.append(("keyword", list(queries), kwargs))
        return [[Document(content=f"kw:{query}")] for query in queries]

    async def search_vector(self, vectors, **kwargs):
        self.calls.append(("vector", list(vectors), kwargs))
        return [[Document(content=f"vec:{index}")] for index, _ in enumerate(vectors)]

    async def search_hybrid(self, queries, vectors, **kwargs):
        self.calls.append(("hybrid", list(queries), list(vectors), kwargs))
        return [[Document(content=f"hy:{query}")] for query in queries]


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    async def embed_queries(self, texts):
        self.calls.append(list(texts))
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class ShortResultStore(FakeStore):
    async def search_keywords(self, queries, **kwargs):
        del queries, kwargs
        return [[]]


def test_keyword_retriever_has_explicit_single_and_many_shapes():
    retriever = KeywordRetriever(FakeStore())

    assert [doc.content for doc in asyncio.run(retriever.retrieve("q", top_k=3))] == ["kw:q"]

    many = asyncio.run(retriever.retrieve_many(["q1", "q2"], top_k=3))
    assert [[doc.content for doc in row] for row in many] == [["kw:q1"], ["kw:q2"]]


def test_vector_and_hybrid_forward_vectors_filters_include_vectors_and_alpha():
    store = FakeStore()
    embedder = FakeEmbedder()
    filters = {"lang": "ru"}

    asyncio.run(VectorRetriever(store, embedder).retrieve_many(["q"], top_k=4, filters=filters, include_vectors=True))
    asyncio.run(HybridRetriever(store, embedder, alpha=0.3).retrieve_many(["q"], top_k=5, include_vectors=True))

    assert store.calls[0] == ("vector", [[0.0, 1.0]], {"top_k": 4, "filters": filters, "include_vectors": True})
    assert store.calls[0][2]["filters"] is filters
    assert store.calls[1] == (
        "hybrid",
        ["q"],
        [[0.0, 1.0]],
        {"alpha": 0.3, "top_k": 5, "filters": None, "include_vectors": True},
    )


@pytest.mark.parametrize("retriever", [KeywordRetriever(FakeStore()), VectorRetriever(FakeStore(), FakeEmbedder())])
def test_retrievers_reject_non_positive_or_boolean_top_k(retriever):
    for top_k in (0, -1, True, False):
        with pytest.raises(ValueError, match="top_k"):
            asyncio.run(retriever.retrieve("q", top_k=top_k))


@pytest.mark.parametrize("alpha", [True, False, -0.1, 1.1, float("nan"), float("inf"), "0.5"])
def test_hybrid_retriever_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        HybridRetriever(FakeStore(), FakeEmbedder(), alpha=alpha)


def test_empty_batches_skip_store_and_embedder_calls():
    store = FakeStore()
    embedder = FakeEmbedder()

    assert asyncio.run(KeywordRetriever(store).retrieve_many([])) == []
    assert asyncio.run(VectorRetriever(store, embedder).retrieve_many([])) == []
    assert asyncio.run(HybridRetriever(store, embedder).retrieve_many([])) == []
    assert store.calls == []
    assert embedder.calls == []


def test_vector_and_hybrid_embed_each_batch_once():
    store = FakeStore()
    embedder = FakeEmbedder()

    asyncio.run(VectorRetriever(store, embedder).retrieve_many(["q1", "q2"]))
    asyncio.run(HybridRetriever(store, embedder).retrieve_many(["q3", "q4"]))

    assert embedder.calls == [["q1", "q2"], ["q3", "q4"]]


def test_retriever_rejects_store_results_with_wrong_query_cardinality():
    with pytest.raises(RetrievalError, match="Expected 2 result groups"):
        asyncio.run(KeywordRetriever(ShortResultStore()).retrieve_many(["q1", "q2"]))


def test_keyword_retriever_import_and_use_do_not_load_torch_or_local_embedder():
    root = Path(__file__).resolve().parents[2]
    script = """
import sys

assert \"torch\" not in sys.modules
assert \"justatom.retrieval.embedders.huggingface\" not in sys.modules
from justatom.retrieval.retriever import KeywordRetriever

class Store:
    async def search_keywords(self, queries, **kwargs):
        return [[] for _ in queries]

KeywordRetriever(Store())
assert \"torch\" not in sys.modules
assert \"justatom.retrieval.embedders.huggingface\" not in sys.modules
"""

    completed = subprocess.run([sys.executable, "-c", script], cwd=root, check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
