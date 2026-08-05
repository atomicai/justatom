from __future__ import annotations

import json
from collections.abc import Iterator
from importlib.resources import as_file
from pathlib import Path
from typing import Any

import polars as pl

from justatom.storing.datasets.errors import (
    DatasetReadError,
    DatasetStreamingUnsupportedError,
    UnsupportedDatasetFormatError,
    UnsupportedDatasetSourceError,
)
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
            f"Streaming is not supported for {suffix} files. Use lazy=False for a small file "
            "or convert it to JSONL or Parquet."
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


def iter_source_rows(source: DatasetSource, options: DatasetReadOptions) -> Iterator[dict[str, Any]]:
    if isinstance(source, LocalDatasetSource):
        _validate_streaming_path(source.path)
        return _iter_local_rows(source.path, options)
    if isinstance(source, PackagedDatasetSource):
        _validate_streaming_path(Path(source.resource.name))
        return _iter_packaged_rows(source, options)
    if isinstance(source, HuggingFaceDatasetSource):
        raise UnsupportedDatasetSourceError("Hugging Face reading must go through DatasetLoader.")
    raise TypeError(f"Unsupported dataset source type: {type(source)!r}")


def source_to_frame(source: DatasetSource, options: DatasetReadOptions) -> pl.DataFrame:
    if isinstance(source, LocalDatasetSource):
        return _local_to_frame(source.path, options)
    if isinstance(source, PackagedDatasetSource):
        with as_file(source.resource) as path:
            return _local_to_frame(Path(path), options)
    if isinstance(source, HuggingFaceDatasetSource):
        raise UnsupportedDatasetSourceError("Hugging Face reading must go through DatasetLoader.")
    raise TypeError(f"Unsupported dataset source type: {type(source)!r}")


__all__ = ["iter_source_rows", "source_to_frame"]
