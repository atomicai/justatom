import asyncio
from pathlib import Path

import numpy as np
import pytest

from justatom.api.eval_qrels import evaluate_qrels, load_benchmark_records, load_eval_qrels_config


class _FakeEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors

    async def embed_queries(self, texts):
        return [self.vectors[text] for text in texts]

    async def embed_documents(self, texts):
        return [self.vectors[text] for text in texts]

    async def close(self):
        raise AssertionError("injected embedder must not be closed by evaluator")


def _config(tmp_path: Path) -> dict:
    return {
        "dataset": {
            "name_or_path": "owner/retrieval",
            "config": "default",
            "revision": "snapshot123",
            "eval": {
                "split": "dev",
                "query_id_col": "query_id",
                "query_field": "query",
                "relevant_id_col": "positive_doc_id",
                "group_id_col": "article_id",
            },
            "corpus": {
                "split": "corpus",
                "document_id_col": "doc_id",
                "content_field": "content",
            },
        },
        "embedding": {
            "model": "fake/model",
            "device": "cpu",
            "batch_size": 2,
            "outer_batch_size": 1,
            "max_length": 16,
            "query_prefix": "",
            "document_prefix": "",
        },
        "ranking": {
            "device": "cpu",
            "query_batch_size": 1,
            "corpus_block_size": 1,
        },
        "output": {"dir": str(tmp_path), "reuse_embeddings": True},
    }


def _loader(*args, **kwargs):
    assert args == ("owner/retrieval",)
    assert kwargs["config"] == "default"
    assert kwargs["revision"] == "snapshot123"
    if kwargs["split"] == "dev":
        return iter(
            [
                {"query_id": "q1", "query": "query one", "positive_doc_id": "d1", "article_id": "a1"},
                {"query_id": "q2", "query": "query two", "positive_doc_id": "d2", "article_id": "a2"},
            ]
        )
    assert kwargs["split"] == "corpus"
    return iter(
        [
            {"doc_id": "d1", "content": "document one"},
            {"doc_id": "d2", "content": "document two"},
            {"doc_id": "d3", "content": "document three"},
        ]
    )


def test_load_benchmark_records_uses_separate_query_and_corpus_splits(tmp_path):
    records = load_benchmark_records(_config(tmp_path), loader=_loader)

    assert records.query_ids == ("q1", "q2")
    assert records.positive_document_ids == ("d1", "d2")
    assert records.group_ids == ("a1", "a2")
    assert records.document_ids == ("d1", "d2", "d3")


def test_exact_qrels_api_writes_metrics_ranks_and_reuses_embedding_cache(tmp_path):
    vectors = {
        "query one": [1.0, 0.0],
        "query two": [1.0, 0.0],
        "document one": [1.0, 0.0],
        "document two": [0.0, 1.0],
        "document three": [-1.0, 0.0],
    }
    config = _config(tmp_path)

    first = asyncio.run(evaluate_qrels(config=config, embedder=_FakeEmbedder(vectors), loader=_loader))
    second = asyncio.run(evaluate_qrels(config=config, embedder=_FakeEmbedder({}), loader=_loader))

    assert first["dataset"]["queries"] == 2
    assert first["dataset"]["corpus"] == 3
    assert first["dataset"]["revision"] == "snapshot123"
    assert len(first["dataset"]["qrels_fingerprint"]) == 64
    assert len(first["dataset"]["group_fingerprint"]) == 64
    assert first["embedding"]["resolved_revision"] is None
    assert first["embedding"]["local_fingerprint"] is None
    assert first["metrics"]["hit_at_1"] == pytest.approx(0.5)
    assert first["metrics"]["recall_at_5"] == 1.0
    assert second["metrics"] == first["metrics"]
    ranks = np.load(tmp_path / "dev.ranks.npz")
    assert ranks["ranks"].tolist() == [1, 2]
    assert ranks["query_ids"].tolist() == ["q1", "q2"]
    assert (tmp_path / "dev.results.json").is_file()


def test_exact_qrels_api_forwards_pinned_model_revision(monkeypatch, tmp_path):
    vectors = {
        "query one": [1.0, 0.0],
        "query two": [0.0, 1.0],
        "document one": [1.0, 0.0],
        "document two": [0.0, 1.0],
        "document three": [-1.0, 0.0],
    }
    captured = {}

    class CapturingEmbedder(_FakeEmbedder):
        def __init__(self, model, *, device, profile, revision):
            super().__init__(vectors)
            captured.update(model=model, device=device, profile=profile, revision=revision)

        async def close(self):
            captured["closed"] = True

    monkeypatch.setattr("justatom.api.eval_qrels.HuggingFaceEmbedder", CapturingEmbedder)
    config = _config(tmp_path)
    config["embedding"]["revision"] = "model-snapshot-123"

    result = asyncio.run(evaluate_qrels(config=config, loader=_loader))

    assert captured["model"] == "fake/model"
    assert captured["device"] == "cpu"
    assert captured["revision"] == "model-snapshot-123"
    assert captured["closed"] is True
    assert result["embedding"]["requested_revision"] == "model-snapshot-123"


def test_benchmark_records_reject_relevant_document_missing_from_corpus(tmp_path):
    def missing_loader(*args, **kwargs):
        rows = list(_loader(*args, **kwargs))
        if kwargs["split"] == "corpus":
            rows = [row for row in rows if row["doc_id"] != "d2"]
        return iter(rows)

    with pytest.raises(ValueError, match="missing from corpus"):
        load_benchmark_records(_config(tmp_path), loader=missing_loader)


def test_habr_preset_pins_snapshot_and_separates_eval_from_corpus():
    config = load_eval_qrels_config(overrides={"dataset": {"id": "habr-ir"}})

    assert config["dataset"]["revision"] == "304cd8e1d1df49f9641c931f8d2bba5daebb330a"
    assert config["dataset"]["eval"]["split"] == "dev"
    assert config["dataset"]["eval"]["relevant_id_col"] == "positive_doc_id"
    assert config["dataset"]["corpus"]["split"] == "corpus"
    assert config["dataset"]["corpus"]["document_id_col"] == "doc_id"
    assert config["embedding"]["revision"] == "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    assert config["embedding"]["query_prefix"].endswith("Query: ")
