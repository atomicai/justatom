import polars as pl

from justatom.storing import dataset as dataset_module


def test_hf_dataset_falls_back_to_cached_parquet_when_builder_breaks(tmp_path, monkeypatch):
    parquet_path = tmp_path / "train-00000-of-00001.parquet"
    pl.DataFrame(
        [
            {"id": 1, "description": "doc-1", "generated": ["q1"], "photos": ["blob"]},
            {"id": 2, "description": "doc-2", "generated": ["q2"], "photos": ["blob"]},
        ]
    ).write_parquet(parquet_path)

    def fake_load_dataset(*args, **kwargs):
        raise TypeError("'str' object is not a mapping")

    monkeypatch.setattr(dataset_module, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        dataset_module,
        "list_repo_files",
        lambda repo_id, repo_type: ["data/train-00000-of-00001.parquet"],
    )
    monkeypatch.setattr(
        dataset_module,
        "hf_hub_download",
        lambda repo_id, filename, repo_type: str(parquet_path),
    )
    dataset_module.HFDataset._repo_files.cache_clear()

    source = dataset_module.HFDataset("hf://justatom/meme-russian-ir")
    data = source.iterator(split="train", drop_columns=["photos"])

    assert isinstance(data, pl.DataFrame)
    assert data.columns == ["id", "description", "generated"]
    assert data.to_dicts() == [
        {"id": 1, "description": "doc-1", "generated": ["q1"]},
        {"id": 2, "description": "doc-2", "generated": ["q2"]},
    ]


def test_hf_dataset_lazy_fallback_returns_lazyframe(tmp_path, monkeypatch):
    parquet_path = tmp_path / "train-00000-of-00001.parquet"
    pl.DataFrame(
        [
            {"id": 1, "description": "doc-1", "generated": ["q1"]},
        ]
    ).write_parquet(parquet_path)

    def fake_load_dataset(*args, **kwargs):
        raise TypeError("builder metadata failed")

    monkeypatch.setattr(dataset_module, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        dataset_module,
        "list_repo_files",
        lambda repo_id, repo_type: ["data/train-00000-of-00001.parquet"],
    )
    monkeypatch.setattr(
        dataset_module,
        "hf_hub_download",
        lambda repo_id, filename, repo_type: str(parquet_path),
    )
    dataset_module.HFDataset._repo_files.cache_clear()

    source = dataset_module.HFDataset("hf://justatom/meme-russian-ir")
    data = source.iterator(lazy=True, split="train")

    assert isinstance(data, pl.LazyFrame)
    assert data.collect().to_dicts() == [{"id": 1, "description": "doc-1", "generated": ["q1"]}]
