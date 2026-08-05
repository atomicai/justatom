from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import pytest

from justatom.storing.datasets import DatasetLoader
from justatom.storing.datasets import readers
from justatom.storing.datasets.errors import DatasetReadError


@dataclass
class _FakeArrowData:
    table: object


class _FakeHFDataset:
    def __init__(self, rows: list[dict]):
        self._frame = pl.from_dicts(rows)
        self.data = _FakeArrowData(self._frame.to_arrow())
        self.column_names = self._frame.columns

    def __len__(self):
        return self._frame.height

    def select(self, indices):
        return _FakeHFDataset(self._frame[list(indices)].to_dicts())

    def remove_columns(self, columns):
        return _FakeHFDataset(self._frame.drop(columns).to_dicts())


def _clear_hf_tokens(monkeypatch):
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN", "HF_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_hf_lazy_uses_streaming_config_token_and_bounded_iteration(monkeypatch):
    calls = []
    consumed = []
    _clear_hf_tokens(monkeypatch)
    monkeypatch.setenv("HF_API_KEY", "hf_test")

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))

        def rows():
            for row in [
                {"id": 1, "content": "one", "blob": "a"},
                {"id": 2, "content": "two", "blob": "b"},
                {"id": 3, "content": "three", "blob": "c"},
            ]:
                consumed.append(row["id"])
                yield row

        return rows()

    monkeypatch.setattr(readers, "load_dataset", fake_load_dataset)

    rows = DatasetLoader.read(
        "owner/data",
        lazy=True,
        split="train",
        config="russian",
        limit=2,
        drop_columns=["blob"],
    )

    assert list(rows) == [
        {"id": 1, "content": "one"},
        {"id": 2, "content": "two"},
    ]
    assert consumed == [1, 2]
    assert calls == [
        (
            ("owner/data",),
            {
                "name": "russian",
                "split": "train",
                "streaming": True,
                "token": "hf_test",
            },
        )
    ]


def test_hf_eager_uses_arrow_to_polars_without_pandas(monkeypatch):
    calls = []
    _clear_hf_tokens(monkeypatch)

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return _FakeHFDataset(
            [
                {"id": 1, "content": "one", "blob": "a"},
                {"id": 2, "content": "two", "blob": "b"},
            ]
        )

    monkeypatch.setattr(readers, "load_dataset", fake_load_dataset)

    frame = DatasetLoader.read(
        "owner/data",
        lazy=False,
        split="validation",
        limit=1,
        drop_columns=["blob"],
    )

    assert isinstance(frame, pl.DataFrame)
    assert frame.to_dicts() == [{"id": 1, "content": "one"}]
    assert calls[0][1]["streaming"] is False


def test_hf_split_fallback_tries_candidates_in_order(monkeypatch):
    attempted = []

    def fake_load_dataset(*args, **kwargs):
        attempted.append(kwargs["split"])
        if kwargs["split"] == "dev":
            raise ValueError("missing dev")
        return iter([{"id": 1}])

    monkeypatch.setattr(readers, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(readers, "list_repo_files", None)
    monkeypatch.setattr(readers, "hf_hub_download", None)

    rows = DatasetLoader.read("owner/data", lazy=True, split="dev|test")

    assert list(rows) == [{"id": 1}]
    assert attempted == ["dev", "test"]


def test_hf_split_failure_preserves_backend_cause(monkeypatch):
    backend_error = ValueError("no requested split")

    def fake_load_dataset(*args, **kwargs):
        raise backend_error

    monkeypatch.setattr(readers, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(readers, "list_repo_files", None)
    monkeypatch.setattr(readers, "hf_hub_download", None)

    with pytest.raises(DatasetReadError, match="dev.*test") as exc_info:
        list(DatasetLoader.read("owner/data", lazy=True, split="dev|test"))

    assert exc_info.value.__cause__ is backend_error


@pytest.mark.parametrize("lazy", [True, False])
def test_hf_parquet_fallback_preserves_requested_contract(tmp_path, monkeypatch, lazy):
    parquet_path = tmp_path / "train-00000-of-00001.parquet"
    pl.DataFrame([{"id": 1, "content": "one", "blob": "drop"}]).write_parquet(parquet_path)
    list_calls = []
    download_calls = []
    _clear_hf_tokens(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_private")

    def fake_load_dataset(*args, **kwargs):
        raise TypeError("builder metadata failed")

    def fake_list_repo_files(*args, **kwargs):
        list_calls.append((args, kwargs))
        return ["data/train-00000-of-00001.parquet"]

    def fake_hf_hub_download(*args, **kwargs):
        download_calls.append((args, kwargs))
        return str(parquet_path)

    monkeypatch.setattr(readers, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(readers, "list_repo_files", fake_list_repo_files)
    monkeypatch.setattr(readers, "hf_hub_download", fake_hf_hub_download)
    readers._repo_files.cache_clear()

    result = DatasetLoader.read(
        "owner/private",
        lazy=lazy,
        split="train",
        drop_columns=["blob"],
    )

    rows = list(result) if lazy else result.to_dicts()
    assert rows == [{"id": 1, "content": "one"}]
    assert list_calls[0][1]["token"] == "hf_private"
    assert download_calls[0][1]["token"] == "hf_private"


def test_hf_requires_datasets_dependency(monkeypatch):
    monkeypatch.setattr(readers, "load_dataset", None)

    with pytest.raises(DatasetReadError, match="pip install datasets"):
        list(DatasetLoader.read("owner/data", lazy=True))
