from __future__ import annotations

import json
import os
from collections.abc import Iterator
from functools import lru_cache
from importlib.resources import as_file
from itertools import islice
from pathlib import Path
from typing import Any

import polars as pl

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None  # type: ignore[assignment]

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except Exception:
    hf_hub_download = None  # type: ignore[assignment]
    list_repo_files = None  # type: ignore[assignment]

from justatom.storing.datasets.errors import DatasetReadError, DatasetStreamingUnsupportedError, UnsupportedDatasetFormatError
from justatom.storing.datasets.source import (
    DatasetReadOptions,
    DatasetSource,
    HuggingFaceDatasetSource,
    LocalDatasetSource,
    PackagedDatasetSource,
)

_STREAMING_EXTENSIONS = {".csv", ".jsonl", ".ndjson", ".parquet"}
_EAGER_ONLY_EXTENSIONS = {".json", ".xlsx"}
_SUPPORTED_EXTENSIONS = _STREAMING_EXTENSIONS | _EAGER_ONLY_EXTENSIONS
_HF_TOKEN_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HF_HUB_TOKEN",
    "HUGGINGFACE_API_KEY",
    "HF_API_KEY",
)


def _unsupported_format(path: Path) -> UnsupportedDatasetFormatError:
    supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
    return UnsupportedDatasetFormatError(
        f"Unsupported dataset format {path.suffix or '<none>'!r} for {path}. Supported formats: {supported}."
    )


def _scan_local(path: Path) -> pl.LazyFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.scan_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pl.scan_ndjson(path)
    if suffix == ".parquet":
        return pl.scan_parquet(path)
    raise _unsupported_format(path)


def _validate_streaming_path(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in _EAGER_ONLY_EXTENSIONS:
        raise DatasetStreamingUnsupportedError(
            f"Streaming is not supported for {suffix} files. Use lazy=False for a small file " "or convert it to JSONL or Parquet."
        )
    if suffix not in _STREAMING_EXTENSIONS:
        raise _unsupported_format(path)


def _apply_lazy_options(frame: pl.LazyFrame, options: DatasetReadOptions) -> pl.LazyFrame:
    available = set(frame.collect_schema().names())
    removable = [column for column in options.drop_columns if column in available]
    if removable:
        frame = frame.drop(removable)
    if options.limit is not None:
        frame = frame.limit(options.limit)
    return frame


def _apply_eager_options(frame: pl.DataFrame, options: DatasetReadOptions) -> pl.DataFrame:
    removable = [column for column in options.drop_columns if column in frame.columns]
    if removable:
        frame = frame.drop(removable)
    if options.limit is not None:
        frame = frame.head(options.limit)
    return frame


def _iter_lazy_frame(frame: pl.LazyFrame) -> Iterator[dict[str, Any]]:
    for batch in frame.collect_batches(maintain_order=True):
        yield from batch.iter_rows(named=True)


def _iter_local_rows(path: Path, options: DatasetReadOptions) -> Iterator[dict[str, Any]]:
    try:
        yield from _iter_lazy_frame(_apply_lazy_options(_scan_local(path), options))
    except (DatasetStreamingUnsupportedError, UnsupportedDatasetFormatError):
        raise
    except Exception as exc:
        raise DatasetReadError(f"Failed to stream dataset file {path}: {exc}") from exc


def _json_frame(path: Path) -> pl.DataFrame:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, list):
        rows = [row for row in payload if row is not None]
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        rows = [row for row in payload["data"] if row is not None]
    elif isinstance(payload, dict):
        rows = [payload]
    else:
        rows = []
    return pl.from_dicts(rows) if rows else pl.DataFrame()


def _local_to_frame(path: Path, options: DatasetReadOptions) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        raise _unsupported_format(path)
    try:
        if suffix in _STREAMING_EXTENSIONS:
            return _apply_lazy_options(_scan_local(path), options).collect(engine="streaming")
        if suffix == ".json":
            return _apply_eager_options(_json_frame(path), options)
        return _apply_eager_options(pl.read_excel(path, engine="xlsx2csv"), options)
    except UnsupportedDatasetFormatError:
        raise
    except Exception as exc:
        raise DatasetReadError(f"Failed to read dataset file {path}: {exc}") from exc


def _iter_packaged_rows(source: PackagedDatasetSource, options: DatasetReadOptions) -> Iterator[dict[str, Any]]:
    with as_file(source.resource) as path:
        yield from _iter_local_rows(Path(path), options)


def _hf_token() -> str | None:
    for name in _HF_TOKEN_ENV_NAMES:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _split_candidates(split: str | None) -> tuple[str, ...]:
    raw = "train" if split is None else split
    candidates = tuple(candidate.strip() for candidate in raw.split("|") if candidate.strip())
    return candidates or ("train",)


@lru_cache(maxsize=64)
def _repo_files(repo_id: str, token: str | None, revision: str | None) -> tuple[str, ...]:
    if list_repo_files is None:
        return ()
    try:
        return tuple(list_repo_files(repo_id=repo_id, repo_type="dataset", token=token, revision=revision))
    except Exception:
        return ()


def _matches_config(repo_file: str, config: str) -> bool:
    normalized_file = repo_file.strip("/").lower()
    normalized_config = config.strip("/").lower()
    if normalized_config in Path(normalized_file).parts:
        return True
    basename = Path(normalized_file).name
    return basename.startswith(f"{normalized_config}-") or f"/{normalized_config}-" in normalized_file


def _parquet_files_for_split(
    repo_files: tuple[str, ...],
    split: str,
    config: str | None = None,
) -> list[str]:
    normalized_split = split.strip().lower()
    matches: list[str] = []
    for repo_file in repo_files:
        normalized_file = repo_file.lower()
        basename = Path(normalized_file).name
        if not normalized_file.endswith(".parquet"):
            continue
        if config is not None and not _matches_config(repo_file, config):
            continue
        if basename == f"{normalized_split}.parquet" or basename.startswith(f"{normalized_split}-"):
            matches.append(repo_file)
            continue
        if f"/{normalized_split}/" in normalized_file or f"/{normalized_split}-" in normalized_file:
            matches.append(repo_file)
    matches = sorted(matches)
    # Canonical HF-generated shards live under data/. Repositories may also
    # publish auxiliary parquet files such as artifacts/qrels/train.parquet;
    # mixing those schemas into the dataset split corrupts the fallback load.
    data_matches = [repo_file for repo_file in matches if Path(repo_file).parts[:1] == ("data",)]
    return data_matches or matches


def _load_parquet_fallback(
    source: HuggingFaceDatasetSource,
    split: str,
    *,
    config: str | None,
    lazy: bool,
    token: str | None,
    revision: str | None,
) -> pl.LazyFrame | pl.DataFrame | None:
    if hf_hub_download is None:
        return None
    parquet_files = _parquet_files_for_split(_repo_files(source.repo_id, token, revision), split, config)
    if not parquet_files:
        return None
    local_paths: list[str] = []
    for repo_file in parquet_files:
        try:
            local_paths.append(
                hf_hub_download(
                    repo_id=source.repo_id,
                    filename=repo_file,
                    repo_type="dataset",
                    token=token,
                    revision=revision,
                )
            )
        except Exception:
            return None
    return pl.scan_parquet(local_paths) if lazy else pl.read_parquet(local_paths)


def _load_hf_source(
    source: HuggingFaceDatasetSource,
    options: DatasetReadOptions,
    *,
    streaming: bool,
):
    if load_dataset is None:
        raise DatasetReadError("Hugging Face dataset support requires the datasets package; run `pip install datasets`.")
    token = _hf_token()
    candidates = _split_candidates(options.split)
    last_error: Exception | None = None
    for candidate in candidates:
        kwargs: dict[str, Any] = {
            "name": options.config,
            "split": candidate,
            "streaming": streaming,
        }
        if options.revision is not None:
            kwargs["revision"] = options.revision
        if token is not None:
            kwargs["token"] = token
        try:
            return load_dataset(source.repo_id, **kwargs)
        except Exception as exc:
            last_error = exc
            fallback = _load_parquet_fallback(
                source,
                candidate,
                config=options.config,
                lazy=streaming,
                token=token,
                revision=options.revision,
            )
            if fallback is not None:
                return fallback
    assert last_error is not None
    rendered_candidates = "|".join(candidates)
    raise DatasetReadError(
        f"Failed to load Hugging Face dataset {source.repo_id!r} for split candidates {rendered_candidates!r}: " f"{last_error}"
    ) from last_error


def _iter_hf_rows(source: HuggingFaceDatasetSource, options: DatasetReadOptions) -> Iterator[dict[str, Any]]:
    dataset = _load_hf_source(source, options, streaming=True)
    if isinstance(dataset, pl.LazyFrame):
        yield from _iter_lazy_frame(_apply_lazy_options(dataset, options))
        return
    dropped = set(options.drop_columns)
    rows = ({key: value for key, value in dict(row).items() if key not in dropped} for row in dataset)
    yield from rows if options.limit is None else islice(rows, options.limit)


def _hf_to_frame(source: HuggingFaceDatasetSource, options: DatasetReadOptions) -> pl.DataFrame:
    dataset = _load_hf_source(source, options, streaming=False)
    if isinstance(dataset, pl.DataFrame):
        return _apply_eager_options(dataset, options)
    if options.drop_columns and hasattr(dataset, "remove_columns"):
        existing = set(getattr(dataset, "column_names", ()))
        removable = [column for column in options.drop_columns if column in existing]
        if removable:
            dataset = dataset.remove_columns(removable)
    if options.limit is not None and hasattr(dataset, "select") and hasattr(dataset, "__len__"):
        dataset = dataset.select(range(min(options.limit, len(dataset))))
    arrow_table = getattr(getattr(dataset, "data", None), "table", None)
    if arrow_table is not None:
        frame = pl.from_arrow(arrow_table)
        if not isinstance(frame, pl.DataFrame):
            raise DatasetReadError(f"Hugging Face dataset {source.repo_id!r} did not produce a tabular Arrow result.")
        return _apply_eager_options(frame, options)
    try:
        return _apply_eager_options(pl.from_dicts([dict(row) for row in dataset]), options)
    except Exception as exc:
        raise DatasetReadError(f"Failed to convert Hugging Face dataset {source.repo_id!r} to Polars: {exc}") from exc


def iter_source_rows(source: DatasetSource, options: DatasetReadOptions) -> Iterator[dict[str, Any]]:
    if isinstance(source, LocalDatasetSource):
        _validate_streaming_path(source.path)
        return _iter_local_rows(source.path, options)
    if isinstance(source, PackagedDatasetSource):
        _validate_streaming_path(Path(source.resource.name))
        return _iter_packaged_rows(source, options)
    if isinstance(source, HuggingFaceDatasetSource):
        return _iter_hf_rows(source, options)
    raise TypeError(f"Unsupported dataset source type: {type(source)!r}")


def source_to_frame(source: DatasetSource, options: DatasetReadOptions) -> pl.DataFrame:
    if isinstance(source, LocalDatasetSource):
        return _local_to_frame(source.path, options)
    if isinstance(source, PackagedDatasetSource):
        with as_file(source.resource) as path:
            return _local_to_frame(Path(path), options)
    if isinstance(source, HuggingFaceDatasetSource):
        return _hf_to_frame(source, options)
    raise TypeError(f"Unsupported dataset source type: {type(source)!r}")


__all__ = ["iter_source_rows", "source_to_frame"]
