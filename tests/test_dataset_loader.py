from pathlib import Path

import pytest

from justatom.storing.datasets.errors import DatasetNotFoundError, UnsupportedDatasetSourceError
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
