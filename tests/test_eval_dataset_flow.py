import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from justatom.api import eval as eval_api


class _Metric:
    def compute(self):
        return 1.0, 0.0


class _Store:
    async def count_documents(self):
        return 2

    async def close(self):
        return None


def test_eval_opens_dataset_once_and_captures_labels_during_indexing(tmp_path):
    source_calls = []
    indexed_documents = []
    evaluated_queries = []
    documents = [
        {"content": "one", "meta": {"labels": ["q1"]}},
        {"content": "two", "meta": {"labels": ["q2", "q3"]}},
    ]

    class _Adapter:
        def iterator(self):
            return iter(documents)

    def fake_from_source(*args, **kwargs):
        source_calls.append((args, kwargs))
        return _Adapter()

    runner = SimpleNamespace(store=_Store())

    async def fake_index(**kwargs):
        indexed_documents.extend(list(kwargs["documents"]))
        return runner

    class _Evaluator:
        async def evaluate_topk(self, **kwargs):
            evaluated_queries.extend(kwargs["queries"])
            return {"HitRate@1": _Metric()}

    with (
        patch.object(eval_api.DatasetRecordAdapter, "from_source", side_effect=fake_from_source),
        patch.object(eval_api.RunningService, "do_index_and_prepare_for_search", side_effect=fake_index),
        patch.object(eval_api.RunningService, "close_embedding_clients", new=AsyncMock()),
        patch.object(eval_api, "EvaluatorRunner", return_value=_Evaluator()),
    ):
        asyncio.run(
            eval_api.main(
                model_name_or_path=None,
                search_pipeline="keywords",
                collection_name="DatasetFlow",
                flush_collection=True,
                dataset_name_or_path="owner/data",
                dataset_lazy=True,
                dataset_config="russian",
                save_results_to_dir=tmp_path,
                labels_field="labels",
                content_field="content",
                split="train",
                metrics=["HitRate"],
                metrics_top_k=[1],
            )
        )

    assert len(source_calls) == 1
    assert source_calls[0][1]["lazy"] is True
    assert source_calls[0][1]["config"] == "russian"
    assert indexed_documents == documents
    assert evaluated_queries == ["q1", "q2", "q3"]


def test_eval_captures_labels_when_existing_index_skips_documents(tmp_path):
    source_calls = []
    evaluated_queries = []
    documents = [
        {"content": "one", "meta": {"labels": ["q1"]}},
        {"content": "two", "meta": {"labels": ["q2", "q3"]}},
    ]

    class _Adapter:
        def iterator(self):
            return iter(documents)

    def fake_from_source(*args, **kwargs):
        source_calls.append((args, kwargs))
        return _Adapter()

    runner = SimpleNamespace(store=_Store())

    async def fake_index(**kwargs):
        assert kwargs["flush_collection"] is False
        return runner

    class _Evaluator:
        async def evaluate_topk(self, **kwargs):
            evaluated_queries.extend(kwargs["queries"])
            return {"HitRate@1": _Metric()}

    with (
        patch.object(eval_api.DatasetRecordAdapter, "from_source", side_effect=fake_from_source),
        patch.object(eval_api.RunningService, "do_index_and_prepare_for_search", side_effect=fake_index),
        patch.object(eval_api.RunningService, "close_embedding_clients", new=AsyncMock()),
        patch.object(eval_api, "EvaluatorRunner", return_value=_Evaluator()),
    ):
        asyncio.run(
            eval_api.main(
                model_name_or_path=None,
                search_pipeline="keywords",
                collection_name="DatasetFlow",
                flush_collection=False,
                dataset_name_or_path="owner/data",
                dataset_lazy=True,
                dataset_config="russian",
                save_results_to_dir=tmp_path,
                labels_field="labels",
                content_field="content",
                split="train",
                metrics=["HitRate"],
                metrics_top_k=[1],
            )
        )

    assert len(source_calls) == 1
    assert evaluated_queries == ["q1", "q2", "q3"]
