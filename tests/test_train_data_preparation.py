import json
import os
import tempfile
from pathlib import Path

from justatom.training import data as training_data


def _write_jsonl(rows: list[dict]) -> Path:
    fd, raw_path = tempfile.mkstemp(suffix=".jsonl", prefix="train_prepare_")
    os.close(fd)
    path = Path(raw_path)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def test_prepare_training_data_prefers_frame_batches(monkeypatch):
    path = _write_jsonl(
        [
            {"chunk_id": "a", "content": "doc-a", "queries": ["q1", "q2"]},
            {"chunk_id": "b", "content": "doc-b", "queries": ["q3"]},
        ]
    )
    monkeypatch.setattr(
        training_data.DatasetRecordAdapter,
        "from_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("frame-backed sources must not use the adapter fallback")),
    )
    try:
        frame, rows, lexical_lookup = training_data.prepare_training_data(
            dataset_name_or_path=path,
            num_samples=2,
            chunk_id_col="chunk_id",
        )
    finally:
        path.unlink(missing_ok=True)

    assert frame.height == len(rows) == 2
    assert all(row["content"] in lexical_lookup for row in rows)


def test_prepare_training_data_uses_lazy_adapter_for_iterable_sources(monkeypatch):
    raw_rows = [
        {"chunk_id": "a", "content": "doc-a", "queries": ["q1", "q2"]},
        {"chunk_id": "b", "content": "doc-b", "queries": ["q3"]},
    ]
    captured: dict[str, object] = {}

    class FakeDataset:
        def iterator(self, **kwargs):
            return iter(raw_rows)

    original = training_data.DatasetRecordAdapter.from_source

    def wrapped(*args, **kwargs):
        captured["lazy"] = kwargs.get("lazy")
        return original(*args, **kwargs)

    monkeypatch.setattr(training_data.DatasetApi, "named", lambda *args, **kwargs: FakeDataset())
    monkeypatch.setattr(training_data.DatasetRecordAdapter, "from_source", wrapped)

    frame, rows, _ = training_data.prepare_training_data(
        dataset_name_or_path="hf://dummy/dataset",
        num_samples=2,
        chunk_id_col="chunk_id",
    )

    assert captured["lazy"] is True
    assert frame.height == len(rows) == 2


def test_reservoir_sampling_is_bounded_and_deterministic():
    source = [{"chunk_id": str(index), "content": f"doc-{index}", "queries": [f"q-{index}"]} for index in range(20)]
    path = _write_jsonl(source)
    try:
        first, _ = training_data.sample_training_rows(
            dataset_name_or_path=path,
            num_samples=5,
            seed=17,
            chunk_id_col="chunk_id",
        )
        second, _ = training_data.sample_training_rows(
            dataset_name_or_path=path,
            num_samples=5,
            seed=17,
            chunk_id_col="chunk_id",
        )
    finally:
        path.unlink(missing_ok=True)

    assert len(first) == 5
    assert first == second


def test_duplicate_content_keeps_distinct_chunk_identity():
    path = _write_jsonl(
        [
            {"chunk_id": "a", "content": "same-doc", "queries": ["q1"]},
            {"chunk_id": "b", "content": "same-doc", "queries": ["q2"]},
        ]
    )
    try:
        _, rows, lexical_lookup = training_data.prepare_training_data(
            dataset_name_or_path=path,
            num_samples=2,
            chunk_id_col="chunk_id",
        )
    finally:
        path.unlink(missing_ok=True)

    assert [row["chunk_id"] for row in rows] == ["a", "b"]
    assert lexical_lookup == {"same-doc": "same-doc"}


def test_rebalance_reduces_avoidable_within_batch_duplicates():
    rows = [
        {"chunk_id": "a1", "content": "same-doc"},
        {"chunk_id": "a2", "content": "same-doc"},
        {"chunk_id": "b1", "content": "doc-b"},
        {"chunk_id": "c1", "content": "doc-c"},
    ]

    rebalanced = training_data.rebalance_rows_by_content(rows, batch_size=3)

    assert training_data.count_batches_with_duplicate_content(rows, batch_size=3) == 1
    assert training_data.count_batches_with_duplicate_content(rebalanced, batch_size=3) == 0
    assert sorted(row["chunk_id"] for row in rebalanced) == ["a1", "a2", "b1", "c1"]


def test_limit_is_applied_after_query_expansion():
    path = _write_jsonl(
        [
            {"chunk_id": "a", "content": "doc-a", "queries": ["q1", "q2"]},
            {"chunk_id": "b", "content": "doc-b", "queries": ["q3"]},
        ]
    )
    try:
        rows = list(
            training_data.iterate_training_rows(
                dataset_name_or_path=path,
                chunk_id_col="chunk_id",
                limit=2,
            )
        )
    finally:
        path.unlink(missing_ok=True)

    assert [row["queries"] for row in rows] == ["q1", "q2"]


def test_iterable_source_supports_custom_fields(monkeypatch):
    class FakeDataset:
        def iterator(self, **kwargs):
            return iter(
                [
                    {"passage": "doc-a", "question": "q1"},
                    {"passage": "doc-b", "question": "q2"},
                ]
            )

    monkeypatch.setattr(training_data.DatasetApi, "named", lambda *args, **kwargs: FakeDataset())

    rows = list(
        training_data.iterate_training_rows(
            dataset_name_or_path="hf://dummy/boolq-like",
            content_field="passage",
            labels_field="question",
            limit=2,
        )
    )

    assert rows == [
        {"content": "doc-a", "queries": "q1", "lexical_text": "doc-a"},
        {"content": "doc-b", "queries": "q2", "lexical_text": "doc-b"},
    ]
