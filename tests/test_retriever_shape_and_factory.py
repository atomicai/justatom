import asyncio
import sys
import types
import unittest

# Provide lightweight stub for optional dependency required by justatom.running.mask
if "bertopic" not in sys.modules:
    bertopic_mod = types.ModuleType("bertopic")
    backend_mod = types.ModuleType("bertopic.backend")

    class _BaseEmbedder:  # pragma: no cover
        pass

    backend_mod.BaseEmbedder = _BaseEmbedder
    bertopic_mod.backend = backend_mod
    sys.modules["bertopic"] = bertopic_mod
    sys.modules["bertopic.backend"] = backend_mod

from justatom.etc.schema import Document
from justatom.running.retriever import API as RetrieverAPI
from justatom.running.retriever import KeywordsRetriever
from justatom.running.indexer import API as IndexerAPI


def _mk_doc(
    content: str, labels: list[str] | None = None, score: float = 1.0
) -> Document:
    payload = {
        "content": content,
        "meta": {"labels": labels or []},
        "score": score,
    }
    return Document.from_dict(payload)


class _DummyKeywordsStore:
    async def search_by_keywords(
        self,
        queries: list[str],
        top_k: int = 5,
        filters: dict | None = None,
        keywords: list[str] | None = None,
    ):
        del filters, keywords
        return [[_mk_doc(f"doc:{q}", labels=[q], score=1.0)] for q in queries]

    def search_by_keywords_sync(
        self,
        queries: list[str],
        top_k: int = 5,
        filters: dict | None = None,
        keywords: list[str] | None = None,
    ):
        del top_k, filters, keywords
        return [[_mk_doc(f"doc:{q}", labels=[q], score=1.0)] for q in queries]


class RetrieverFactoryAndShapeTest(unittest.TestCase):
    def test_retriever_factory_names(self):
        store = _DummyKeywordsStore()
        retriever = RetrieverAPI.named("keywords", store=store)
        self.assertIsInstance(retriever, KeywordsRetriever)

        with self.assertRaises(ValueError):
            RetrieverAPI.named("emebedding", store=store)

    def test_indexer_factory_names(self):
        _ = IndexerAPI.named("keywords", store=_DummyKeywordsStore())
        with self.assertRaises(ValueError):
            IndexerAPI.named("emebedding", store=_DummyKeywordsStore())

    def test_keywords_retriever_return_shape_single_and_multi(self):
        store = _DummyKeywordsStore()
        retriever = KeywordsRetriever(store=store)

        single = asyncio.run(retriever.retrieve_topk("q1", top_k=1))
        self.assertIsInstance(single, list)
        self.assertEqual(len(single), 1)
        self.assertIsInstance(single[0], Document)

        multi = asyncio.run(retriever.retrieve_topk(["q1", "q2"], top_k=1))
        self.assertIsInstance(multi, list)
        self.assertEqual(len(multi), 2)
        self.assertIsInstance(multi[0], list)
        self.assertIsInstance(multi[0][0], Document)


if __name__ == "__main__":
    unittest.main()
