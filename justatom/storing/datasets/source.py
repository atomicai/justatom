from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import TypeAlias

from justatom.storing.datasets.errors import DatasetNotFoundError, UnsupportedDatasetSourceError

_HF_REPO_ID = re.compile(r"^[^/?\s:]+/[^/?\s:]+$")


@dataclass(frozen=True)
class DatasetReadOptions:
    split: str | None = None
    config: str | None = None
    limit: int | None = None
    drop_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.limit is not None and self.limit < 0:
            raise ValueError("dataset limit must be >= 0")
        normalized_columns = tuple(str(column) for column in self.drop_columns if str(column).strip())
        object.__setattr__(self, "drop_columns", normalized_columns)


@dataclass(frozen=True)
class LocalDatasetSource:
    path: Path


@dataclass(frozen=True)
class PackagedDatasetSource:
    name: str
    resource: Traversable


@dataclass(frozen=True)
class HuggingFaceDatasetSource:
    repo_id: str


DatasetSource: TypeAlias = LocalDatasetSource | PackagedDatasetSource | HuggingFaceDatasetSource


def _demo_resource() -> Traversable:
    return files("justatom.builtins").joinpath("datasets", "demo_retrieval.jsonl")


def resolve_dataset_source(value: str | Path) -> DatasetSource:
    raw = str(value).strip()
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return LocalDatasetSource(path=candidate.resolve())
    if raw == "demo":
        return PackagedDatasetSource(name="demo", resource=_demo_resource())
    if raw.startswith(("http://", "https://")):
        raise UnsupportedDatasetSourceError(
            "HTTP dataset sources are not supported yet; download the file and pass its local path."
        )
    if raw.startswith("builtin://"):
        raise UnsupportedDatasetSourceError(
            "The builtin:// dataset syntax was removed; use 'demo' or a normal local path."
        )
    if raw.startswith("hf://"):
        replacement = raw.removeprefix("hf://").split("?", maxsplit=1)[0]
        raise UnsupportedDatasetSourceError(
            f"The hf:// dataset syntax was removed; use '{replacement}' and separate config/split fields."
        )
    if "://" in raw:
        raise UnsupportedDatasetSourceError(
            "Dataset custom URI syntax is not supported; use owner/dataset for Hugging Face or a local path."
        )
    if _HF_REPO_ID.fullmatch(raw):
        return HuggingFaceDatasetSource(repo_id=raw)
    raise DatasetNotFoundError(
        f"Dataset source {raw!r} is neither an existing file, 'demo', nor a Hugging Face owner/dataset ID."
    )


__all__ = [
    "DatasetReadOptions",
    "DatasetSource",
    "HuggingFaceDatasetSource",
    "LocalDatasetSource",
    "PackagedDatasetSource",
    "resolve_dataset_source",
]
