import json
import tempfile
from pathlib import Path

from justatom.api import train as train_api


def _write_jsonl(rows: list[dict]) -> Path:
    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="train_prepare_")
    Path(path).unlink(missing_ok=True)
    out = Path(path)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def test_prepare_training_data_prefers_polars_batches_for_frame_backed_sources(
    monkeypatch,
):
    rows = [
        {"chunk_id": "a", "content": "doc-a", "queries": ["q1", "q2"]},
        {"chunk_id": "b", "content": "doc-b", "queries": ["q3"]},
        {"chunk_id": "c", "content": "doc-c", "queries": ["q4"]},
    ]
    path = _write_jsonl(rows)

    def fail(*args, **kwargs):
        raise AssertionError("frame-backed datasets should avoid from_source fallback")

    monkeypatch.setattr(train_api.DatasetRecordAdapter, "from_source", fail)

    try:
        pl_data, js_data, lexical_by_content = train_api.prepare_training_data(
            dataset_name_or_path=path,
            num_samples=2,
            content_field="content",
            labels_field="queries",
            chunk_id_col="chunk_id",
        )
    finally:
        path.unlink(missing_ok=True)

    assert pl_data.height == 2
    assert len(js_data) == 2
    assert set(lexical_by_content).issubset({"doc-a", "doc-b", "doc-c"})


def test_prepare_training_data_falls_back_to_lazy_adapter_for_iterable_sources(
    monkeypatch,
):
    rows = [
        {"chunk_id": "a", "content": "doc-a", "queries": ["q1", "q2"]},
        {"chunk_id": "b", "content": "doc-b", "queries": ["q3"]},
        {"chunk_id": "c", "content": "doc-c", "queries": ["q4"]},
    ]
    captured: dict[str, object] = {}

    class _FakeDataset:
        def iterator(self, **kwargs):
            return iter(rows)

    original_from_source = train_api.DatasetRecordAdapter.from_source

    def fake_named(*args, **kwargs):
        return _FakeDataset()

    def wrapped(*args, **kwargs):
        captured["lazy"] = kwargs.get("lazy")
        return original_from_source(*args, **kwargs)

    monkeypatch.setattr(train_api.DatasetApi, "named", fake_named)
    monkeypatch.setattr(train_api.DatasetRecordAdapter, "from_source", wrapped)

    pl_data, js_data, lexical_by_content = train_api.prepare_training_data(
        dataset_name_or_path="hf://dummy/dataset",
        num_samples=2,
        content_field="content",
        labels_field="queries",
        chunk_id_col="chunk_id",
    )

    assert captured["lazy"] is True
    assert pl_data.height == 2
    assert len(js_data) == 2
    assert set(lexical_by_content).issubset({"doc-a", "doc-b", "doc-c"})


def test_prepare_training_data_streams_and_bounds_sample_size():
    rows = [
        {"chunk_id": "a", "content": "doc-a", "queries": ["q1", "q2"]},
        {"chunk_id": "b", "content": "doc-b", "queries": ["q3", "q4"]},
        {"chunk_id": "c", "content": "doc-c", "queries": ["q5", "q6"]},
    ]
    path = _write_jsonl(rows)

    try:
        pl_data, js_data, lexical_by_content = train_api.prepare_training_data(
            dataset_name_or_path=path,
            num_samples=3,
            content_field="content",
            labels_field="queries",
            chunk_id_col="chunk_id",
        )
    finally:
        path.unlink(missing_ok=True)

    assert pl_data.height == 3
    assert len(js_data) == 3
    assert all(row["content"] in lexical_by_content for row in js_data)
    assert all(row["lexical_text"] == lexical_by_content[row["content"]] for row in js_data)


def test_iterate_training_rows_applies_limit_after_query_expansion():
    rows = [
        {"chunk_id": "a", "content": "doc-a", "queries": ["q1", "q2"]},
        {"chunk_id": "b", "content": "doc-b", "queries": ["q3"]},
    ]
    path = _write_jsonl(rows)

    try:
        sample_rows = list(
            train_api.iterate_training_rows(
                dataset_name_or_path=path,
                content_field="content",
                labels_field="queries",
                chunk_id_col="chunk_id",
                limit=2,
            )
        )
    finally:
        path.unlink(missing_ok=True)

    assert len(sample_rows) == 2
    assert [row["queries"] for row in sample_rows] == ["q1", "q2"]


def test_create_training_job_selects_gamma_only_mode():
    job = train_api.create_training_job(
        dataset_name_or_path="dummy",
        freeze_encoder=True,
        include_semantic_gamma=True,
        include_keywords_gamma=False,
    )

    assert isinstance(job, train_api.GammaOnlyTrainingJob)
    assert job.training_mode == "gamma-only"


def test_create_training_job_selects_encoder_gamma_mode():
    job = train_api.create_training_job(
        dataset_name_or_path="dummy",
        freeze_encoder=False,
        include_semantic_gamma=True,
        include_keywords_gamma=True,
    )

    assert isinstance(job, train_api.EncoderGammaTrainingJob)
    assert job.training_mode == "encoder+gamma"


def test_create_training_job_selects_encoder_only_mode():
    job = train_api.create_training_job(
        dataset_name_or_path="dummy",
        freeze_encoder=False,
        include_semantic_gamma=False,
        include_keywords_gamma=False,
    )

    assert isinstance(job, train_api.EncoderOnlyTrainingJob)
    assert job.training_mode == "encoder-only"
