import asyncio
from copy import deepcopy
from unittest.mock import patch

import pytest

from justatom.api import eval as eval_api


class _Metric:
    def compute(self):
        return 1.0, 0.0


class _Store:
    def __init__(self, events, existing):
        self.events = events
        self.existing = existing

    async def count_documents(self):
        self.events.append("count")
        return self.existing

    async def clear(self):
        self.events.append("clear")
        self.existing = 0


class _Runtime:
    def __init__(self, *, existing=2):
        self.events = []
        self.indexed_documents = []
        self.store = _Store(self.events, existing)
        self.retriever = object()

    async def index(self, documents, *, batch_size):
        assert not isinstance(documents, list)
        self.events.append(("index", batch_size))
        self.indexed_documents.extend(documents)
        self.store.existing = len(self.indexed_documents)
        return len(self.indexed_documents)

    async def close(self):
        self.events.append("close")


def _dataset_patch(documents, source_calls):
    class _Adapter:
        def iterator(self):
            return iter(documents)

    def fake_from_source(*args, **kwargs):
        source_calls.append((args, kwargs))
        return _Adapter()

    return patch.object(eval_api.DatasetRecordAdapter, "from_source", side_effect=fake_from_source)


def test_eval_opens_dataset_once_and_streams_documents_during_flush_index(tmp_path):
    source_calls = []
    evaluated = []
    runtime = _Runtime(existing=7)
    documents = [
        {"content": "one", "meta": {"labels": ["q1"]}},
        {"content": "two", "meta": {"labels": ["q2", "q3"]}},
    ]

    async def fake_build_runtime(config):
        runtime.events.append(("build", config))
        return runtime

    class _Evaluator:
        def __init__(self, ir):
            assert ir is runtime.retriever

        async def evaluate_topk(self, **kwargs):
            runtime.events.append("evaluate")
            evaluated.append(kwargs)
            return {"HitRate@1": _Metric()}

    retrieval_config = {
        "mode": "keyword",
        "alpha": 0.5,
        "store": {"collection": "DatasetFlow", "grpc_port": 50051},
    }
    with (
        _dataset_patch(documents, source_calls),
        patch.object(eval_api, "build_runtime", side_effect=fake_build_runtime),
        patch.object(eval_api, "EvaluatorRunner", _Evaluator),
    ):
        asyncio.run(
            eval_api.main(
                retrieval_config=retrieval_config,
                flush_collection=True,
                dataset_name_or_path="owner/data",
                dataset_lazy=True,
                dataset_config="russian",
                save_results_to_dir=tmp_path,
                labels_field="labels",
                content_field="content",
                split="train",
                index_batch_size=4,
                top_k=9,
                search_batch_size=3,
                metrics=["HitRate"],
                metrics_top_k=["HitRate"],
                eval_top_k=[1],
                filters={"fields": ["tenant"]},
            )
        )

    assert len(source_calls) == 1
    assert source_calls[0][1]["lazy"] is True
    assert source_calls[0][1]["config"] == "russian"
    assert source_calls[0][1]["filter_fields"] == ["tenant"]
    assert runtime.indexed_documents == documents
    assert runtime.events == [
        ("build", retrieval_config),
        "count",
        "clear",
        ("index", 4),
        "evaluate",
        "close",
    ]
    assert evaluated == [
        {
            "queries": ["q1", "q2", "q3"],
            "top_k": 9,
            "metrics": ["HitRate"],
            "metrics_top_k": ["HitRate"],
            "eval_top_k": [1],
            "batch_size": 3,
        }
    ]
    result_files = list(tmp_path.glob("*.csv"))
    assert len(result_files) == 1
    assert result_files[0].parent == tmp_path
    assert "|" not in result_files[0].name


def test_eval_exhausts_labels_when_existing_index_skips_writes(tmp_path):
    source_calls = []
    evaluated_queries = []
    runtime = _Runtime(existing=2)
    documents = [
        {"content": "one", "meta": {"labels": ["q1"]}},
        {"content": "two", "meta": {"labels": ["q2", "q3"]}},
    ]

    async def fake_build_runtime(config):
        runtime.events.append(("build", config))
        return runtime

    class _Evaluator:
        def __init__(self, ir):
            assert ir is runtime.retriever

        async def evaluate_topk(self, **kwargs):
            runtime.events.append("evaluate")
            evaluated_queries.extend(kwargs["queries"])
            return {"HitRate@1": _Metric()}

    retrieval_config = {
        "mode": "keyword",
        "store": {"collection": "DatasetFlow"},
    }
    with (
        _dataset_patch(documents, source_calls),
        patch.object(eval_api, "build_runtime", side_effect=fake_build_runtime),
        patch.object(eval_api, "EvaluatorRunner", _Evaluator),
    ):
        asyncio.run(
            eval_api.main(
                retrieval_config=retrieval_config,
                flush_collection=False,
                dataset_name_or_path="owner/data",
                dataset_lazy=True,
                dataset_config="russian",
                save_results_to_dir=tmp_path,
                labels_field="labels",
                metrics=["HitRate"],
                metrics_top_k=["HitRate"],
                eval_top_k=[1],
            )
        )

    assert len(source_calls) == 1
    assert runtime.indexed_documents == []
    assert evaluated_queries == ["q1", "q2", "q3"]
    assert runtime.events == [
        ("build", retrieval_config),
        "count",
        "evaluate",
        "close",
    ]


def test_eval_injects_e5_prefixes_into_builder_copy_only():
    runtime = _Runtime(existing=1)
    received_configs = []
    retrieval_config = {
        "mode": "vector",
        "embedding": {
            "backend": "local",
            "model": "intfloat/multilingual-e5-small",
            "query_prefix": None,
            "document_prefix": None,
        },
        "store": {"collection": "DatasetFlow"},
    }
    original_config = deepcopy(retrieval_config)

    async def fake_build_runtime(config):
        received_configs.append(config)
        return runtime

    with patch.object(eval_api, "build_runtime", side_effect=fake_build_runtime):
        asyncio.run(eval_api.main(retrieval_config=retrieval_config))

    assert retrieval_config == original_config
    assert received_configs[0] is not retrieval_config
    assert received_configs[0]["embedding"]["query_prefix"] == "query: "
    assert received_configs[0]["embedding"]["document_prefix"] == "passage: "
    assert runtime.events == ["count", "close"]


def test_eval_closes_runtime_when_evaluator_fails(tmp_path):
    source_calls = []
    runtime = _Runtime(existing=1)
    documents = [{"content": "one", "meta": {"labels": ["q1"]}}]

    async def fake_build_runtime(config):
        return runtime

    class _FailingEvaluator:
        def __init__(self, ir):
            assert ir is runtime.retriever

        async def evaluate_topk(self, **kwargs):
            raise RuntimeError("evaluation failed")

    with (
        _dataset_patch(documents, source_calls),
        patch.object(eval_api, "build_runtime", side_effect=fake_build_runtime),
        patch.object(eval_api, "EvaluatorRunner", _FailingEvaluator),
        pytest.raises(RuntimeError, match="evaluation failed"),
    ):
        asyncio.run(
            eval_api.main(
                retrieval_config={"mode": "keyword", "store": {"collection": "DatasetFlow"}},
                dataset_name_or_path="owner/data",
                save_results_to_dir=tmp_path,
                labels_field="labels",
                metrics_top_k=["HitRate"],
            )
        )

    assert runtime.events == ["count", "close"]


def test_eval_closes_runtime_when_cancelled():
    runtime = _Runtime(existing=1)

    async def cancel_during_count():
        runtime.events.append("count")
        raise asyncio.CancelledError

    runtime.store.count_documents = cancel_during_count

    async def fake_build_runtime(config):
        return runtime

    with (
        patch.object(eval_api, "build_runtime", side_effect=fake_build_runtime),
        pytest.raises(asyncio.CancelledError),
    ):
        asyncio.run(
            eval_api.main(
                retrieval_config={"mode": "keyword", "store": {"collection": "DatasetFlow"}},
            )
        )

    assert runtime.events == ["count", "close"]
