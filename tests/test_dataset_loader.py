import json
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest

from justatom.storing.datasets import DatasetLoader
from justatom.storing.datasets.errors import (
    DatasetNotFoundError,
    DatasetStreamingUnsupportedError,
    UnsupportedDatasetFormatError,
    UnsupportedDatasetSourceError,
)
from justatom.storing.datasets.readers import iter_source_rows, source_to_frame
from justatom.storing.datasets.source import (
    DatasetReadOptions,
    HuggingFaceDatasetSource,
    LocalDatasetSource,
    PackagedDatasetSource,
    resolve_dataset_source,
)


def test_existing_owner_dataset_path_wins_over_hugging_face(tmp_path, monkeypatch):
    local = tmp_path / "owner" / "dataset"
    local.parent.mkdir()
    local.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    source = resolve_dataset_source("owner/dataset")

    assert source == LocalDatasetSource(local.resolve())


def test_local_path_expands_user_home(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"id": 1}\n', encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    source = resolve_dataset_source("~/dataset.jsonl")

    assert source == LocalDatasetSource(dataset_path.resolve())


def test_demo_resolves_to_packaged_resource():
    source = resolve_dataset_source("demo")

    assert isinstance(source, PackagedDatasetSource)
    assert source.name == "demo"
    assert source.resource.name == "demo_retrieval.jsonl"
    assert source.resource.is_file()


def test_owner_dataset_resolves_to_hugging_face():
    source = resolve_dataset_source("justatom/meme-russian-ir")

    assert source == HuggingFaceDatasetSource(repo_id="justatom/meme-russian-ir")


@pytest.mark.parametrize(
    ("value", "suggestion"),
    [
        ("builtin://datasets/demo_retrieval.jsonl", "demo"),
        ("hf://justatom/meme-russian-ir", "justatom/meme-russian-ir"),
    ],
)
def test_legacy_uri_is_rejected_with_new_syntax(value, suggestion):
    with pytest.raises(UnsupportedDatasetSourceError, match=suggestion):
        resolve_dataset_source(value)


@pytest.mark.parametrize("value", ["http://example.test/data.jsonl", "https://example.test/data.parquet"])
def test_http_source_is_rejected_with_download_guidance(value):
    with pytest.raises(UnsupportedDatasetSourceError, match="download"):
        resolve_dataset_source(value)


@pytest.mark.parametrize("value", ["justatom", "missing.jsonl", "not-a-dataset"])
def test_missing_bare_source_is_not_magic(value):
    with pytest.raises(DatasetNotFoundError, match="owner/dataset"):
        resolve_dataset_source(value)


def test_read_options_normalize_drop_columns():
    options = DatasetReadOptions(drop_columns=["photo", "embedding"])

    assert options.drop_columns == ("photo", "embedding")


def test_read_options_reject_negative_limit():
    with pytest.raises(ValueError, match="limit must be >= 0"):
        DatasetReadOptions(limit=-1)


def _write_tabular_dataset(path: Path) -> Path:
    frame = pl.DataFrame(
        [
            {"id": 1, "content": "one", "blob": "drop-a"},
            {"id": 2, "content": "two", "blob": "drop-b"},
        ]
    )
    if path.suffix == ".csv":
        frame.write_csv(path)
    elif path.suffix in {".jsonl", ".ndjson"}:
        frame.write_ndjson(path)
    elif path.suffix == ".parquet":
        frame.write_parquet(path)
    elif path.suffix == ".xlsx":
        frame.write_excel(path)
    else:
        raise AssertionError(f"Unsupported test suffix: {path.suffix}")
    return path


@pytest.mark.parametrize("suffix", [".csv", ".jsonl", ".ndjson", ".parquet"])
def test_streaming_local_formats_return_bounded_row_iterator(tmp_path, suffix):
    path = _write_tabular_dataset(tmp_path / f"dataset{suffix}")
    source = LocalDatasetSource(path.resolve())

    rows = iter_source_rows(
        source,
        DatasetReadOptions(limit=1, drop_columns=["blob"]),
    )

    assert isinstance(rows, Iterator)
    assert iter(rows) is rows
    assert list(rows) == [{"id": 1, "content": "one"}]


@pytest.mark.parametrize("suffix", [".csv", ".jsonl", ".ndjson", ".parquet", ".xlsx"])
def test_eager_local_formats_return_polars_frame(tmp_path, suffix):
    path = _write_tabular_dataset(tmp_path / f"dataset{suffix}")
    source = LocalDatasetSource(path.resolve())

    frame = source_to_frame(
        source,
        DatasetReadOptions(limit=1, drop_columns=["blob"]),
    )

    assert isinstance(frame, pl.DataFrame)
    assert frame.to_dicts() == [{"id": 1, "content": "one"}]


@pytest.mark.parametrize("suffix", [".json", ".xlsx"])
def test_non_streaming_formats_fail_before_parsing_file(tmp_path, suffix):
    path = tmp_path / f"broken{suffix}"
    path.write_text("not valid data", encoding="utf-8")

    with pytest.raises(DatasetStreamingUnsupportedError, match="lazy=False"):
        iter_source_rows(LocalDatasetSource(path.resolve()), DatasetReadOptions())


@pytest.mark.parametrize(
    "payload",
    [
        [{"id": 1, "content": "one"}, {"id": 2, "content": "two"}],
        {"data": [{"id": 1, "content": "one"}, {"id": 2, "content": "two"}]},
        {"id": 1, "content": "one"},
    ],
)
def test_eager_json_preserves_supported_shapes(tmp_path, payload):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    frame = source_to_frame(
        LocalDatasetSource(path.resolve()),
        DatasetReadOptions(limit=1),
    )

    assert isinstance(frame, pl.DataFrame)
    assert frame.to_dicts() == [{"id": 1, "content": "one"}]


def test_packaged_demo_supports_lazy_and_eager_reads():
    source = resolve_dataset_source("demo")

    rows = iter_source_rows(source, DatasetReadOptions(limit=1))
    frame = source_to_frame(source, DatasetReadOptions(limit=1))

    assert isinstance(rows, Iterator)
    assert list(rows) == frame.to_dicts()
    assert frame.height == 1


def test_unsupported_local_extension_lists_supported_formats(tmp_path):
    path = tmp_path / "dataset.txt"
    path.write_text("content", encoding="utf-8")

    with pytest.raises(UnsupportedDatasetFormatError, match="csv"):
        source_to_frame(LocalDatasetSource(path.resolve()), DatasetReadOptions())


@pytest.mark.parametrize(
    "source",
    ["justatom/meme-russian-ir?split=train", "unicamp-dl/mmarco?config=russian"],
)
def test_hugging_face_query_parameters_are_rejected(source):
    with pytest.raises(UnsupportedDatasetSourceError, match="separate config/split"):
        DatasetLoader.read(source, lazy=True)


@pytest.mark.parametrize("option", ["split", "config"])
def test_local_source_rejects_hugging_face_only_options(tmp_path, option):
    path = _write_tabular_dataset(tmp_path / "dataset.jsonl")
    kwargs = {option: "train"}

    with pytest.raises(UnsupportedDatasetSourceError, match=option):
        DatasetLoader.read(path, lazy=True, **kwargs)
