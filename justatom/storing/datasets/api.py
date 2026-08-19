from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, overload

import polars as pl

from justatom.storing.datasets.errors import UnsupportedDatasetSourceError
from justatom.storing.datasets.readers import iter_source_rows, source_to_frame
from justatom.storing.datasets.source import DatasetReadOptions, HuggingFaceDatasetSource, resolve_dataset_source


class DatasetLoader:
    @staticmethod
    @overload
    def read(
        source: str | Path,
        *,
        lazy: Literal[True],
        split: str | None = None,
        config: str | None = None,
        revision: str | None = None,
        limit: int | None = None,
        drop_columns: Sequence[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        pass

    @staticmethod
    @overload
    def read(
        source: str | Path,
        *,
        lazy: Literal[False],
        split: str | None = None,
        config: str | None = None,
        revision: str | None = None,
        limit: int | None = None,
        drop_columns: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        pass

    @staticmethod
    def read(
        source: str | Path,
        *,
        lazy: bool,
        split: str | None = None,
        config: str | None = None,
        revision: str | None = None,
        limit: int | None = None,
        drop_columns: Sequence[str] | None = None,
    ) -> Iterator[dict[str, Any]] | pl.DataFrame:
        if not isinstance(lazy, bool):
            raise TypeError("lazy must be a boolean")
        resolved = resolve_dataset_source(source)
        if not isinstance(resolved, HuggingFaceDatasetSource):
            if split is not None:
                raise UnsupportedDatasetSourceError("split is only supported for Hugging Face dataset sources.")
            if config is not None:
                raise UnsupportedDatasetSourceError("config is only supported for Hugging Face dataset sources.")
            if revision is not None:
                raise UnsupportedDatasetSourceError("revision is only supported for Hugging Face dataset sources.")
        options = DatasetReadOptions(
            split=split,
            config=config,
            revision=revision,
            limit=limit,
            drop_columns=tuple(drop_columns or ()),
        )
        return iter_source_rows(resolved, options) if lazy else source_to_frame(resolved, options)


__all__ = ["DatasetLoader"]
