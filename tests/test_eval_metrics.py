import asyncio
import math
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

from justatom.running.evaluator import EvaluatorRunner
from justatom.running.retriever import KeywordsRetriever
from justatom.etc.schema import Document
from justatom.tooling.dataset import DatasetRecordAdapter


def _doc(content: str, queries: list[str], score: float) -> Document:
    doc = Document.from_dict(
        {
            "content": content,
            "queries": queries,
            "score": score,
        }
    )
    doc.meta["labels"] = list(doc.meta.get("queries", []))
    return doc


class _DummyKeywordsStore:
    def __init__(self):
        self._rankings = {
            "cat diet": [
                _doc("cat-0", ["cat diet"], 1.0),
                _doc("cat-1", ["other-cat"], 0.9),
                _doc("cat-2", ["other-cat-2"], 0.8),
                _doc("cat-3", ["other-cat-3"], 0.7),
                _doc("cat-4", ["other-cat-4"], 0.6),
            ],
            "dog training": [
                _doc("dog-0", ["other-dog"], 1.0),
                _doc("dog-1", ["dog training"], 0.9),
                _doc("dog-2", ["other-dog-2"], 0.8),
                _doc("dog-3", ["other-dog-3"], 0.7),
                _doc("dog-4", ["other-dog-4"], 0.6),
            ],
            "planet facts": [
                _doc("space-0", ["other-space"], 1.0),
                _doc("space-1", ["other-space-2"], 0.9),
                _doc("space-2", ["planet facts"], 0.8),
                _doc("space-3", ["other-space-3"], 0.7),
                _doc("space-4", ["other-space-4"], 0.6),
            ],
        }

    async def search_by_keywords(
        self,
        queries: list[str],
        top_k: int = 5,
        filters: dict | None = None,
        keywords: list[str] | None = None,
    ):
        del filters, keywords
        return [self._rankings[q][:top_k] for q in queries]


class EvalMetricsTest(unittest.TestCase):
    def test_keywords_retriever_evaluator_metrics_k_1_2_5(self):
        rows = [
            {"content": "Paragraph about Python.", "queries": None},
            {"content": "Paragraph about cats.", "queries": "cat diet"},
            {
                "content": "Paragraph about dogs.",
                "queries": ["dog training", None, ""],
            },
            {"content": "Paragraph about planets.", "queries": ["planet facts"]},
        ]

        adapter = DatasetRecordAdapter(
            records=rows,
            content_col="content",
            queries_col="queries",
        )

        docs = list(adapter.iterator())
        self.assertEqual(docs[0]["meta"]["labels"], [])
        self.assertEqual(docs[1]["meta"]["labels"], ["cat diet"])
        self.assertEqual(docs[2]["meta"]["labels"], ["dog training"])
        self.assertEqual(docs[3]["meta"]["labels"], ["planet facts"])

        queries = DatasetRecordAdapter.extract_labels(docs)
        self.assertEqual(queries, ["cat diet", "dog training", "planet facts"])

        retriever = KeywordsRetriever(store=_DummyKeywordsStore())
        evaluator = EvaluatorRunner(ir=retriever)
        metrics = asyncio.run(
            evaluator.evaluate_topk(
                queries=queries,
                metrics=None,
                metrics_top_k=["HitRate@", "mrr@", "map@", "ndcg@"],
                eval_top_k=[1, 2, 5],
                top_k=5,
                batch_size=2,
            )
        )

        expected_means = {
            "HitRate@1": 1.0 / 3.0,
            "HitRate@2": 2.0 / 3.0,
            "HitRate@5": 1.0,
            "mrr@1": 1.0 / 3.0,
            "mrr@2": 0.5,
            "mrr@5": 11.0 / 18.0,
            "map@1": 1.0 / 3.0,
            "map@2": 0.5,
            "map@5": 11.0 / 18.0,
            "ndcg@1": 1.0 / 3.0,
            "ndcg@2": 0.5436432511904858,
            "ndcg@5": 0.7103099178571526,
        }
        expected_stds = {
            "HitRate@1": math.sqrt(1.0 / 3.0),
            "HitRate@2": math.sqrt(1.0 / 3.0),
            "HitRate@5": 0.0,
            "mrr@1": math.sqrt(1.0 / 3.0),
            "mrr@2": 0.5,
            "mrr@5": 0.3469443332443555,
            "map@1": math.sqrt(1.0 / 3.0),
            "map@2": 0.5,
            "map@5": 0.3469443332443555,
            "ndcg@1": math.sqrt(1.0 / 3.0),
            "ndcg@2": 0.5056819159545134,
            "ndcg@5": 0.2592795939936615,
        }

        for metric_key, expected_mean in expected_means.items():
            mean, std = metrics[metric_key].compute()
            self.assertAlmostEqual(mean, expected_mean, places=7)
            self.assertAlmostEqual(std, expected_stds[metric_key], places=7)

        print("")


if __name__ == "__main__":
    unittest.main()
