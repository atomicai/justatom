from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import polars as pl
from huggingface_hub import hf_hub_download, list_repo_files


HABR_SOURCE_COLUMNS = (
    "id",
    "language",
    "url",
    "title",
    "text_markdown",
    "type",
    "time_published",
    "statistics",
    "labels",
    "hubs",
    "flows",
    "tags",
    "reading_time",
    "format",
    "complexity",
)
HF_TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN", "HF_API_KEY")


def resolve_hf_token(env: dict[str, str] | None = None) -> str | None:
    active_env = os.environ if env is None else env
    for name in HF_TOKEN_ENV_NAMES:
        value = active_env.get(name)
        if value and str(value).strip():
            return str(value).strip()
    return None


class HabrSource:
    def __init__(
        self,
        *,
        repo_id: str = "justatom/habr-ds",
        config: str = "default",
        split: str = "train",
        revision: str = "main",
        cache_dir: str | Path | None = None,
    ) -> None:
        self.repo_id = str(repo_id)
        self.config = str(config)
        self.split = str(split)
        self.revision = str(revision)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)

    def _repo_files(self) -> list[str]:
        return list(
            list_repo_files(
                repo_id=self.repo_id,
                repo_type="dataset",
                revision=self.revision,
                token=resolve_hf_token(),
            )
        )

    def _matching_parquet_files(self) -> list[str]:
        parquet_files = sorted(path for path in self._repo_files() if path.lower().endswith(".parquet"))
        if not parquet_files:
            raise RuntimeError(f"No parquet files found in dataset repo {self.repo_id!r}.")

        split = self.split.casefold()
        matching = []
        for path in parquet_files:
            lowered = path.casefold()
            basename = Path(lowered).name
            if basename == f"{split}.parquet" or basename.startswith(f"{split}-"):
                matching.append(path)
            elif f"/{split}/" in lowered or f"/{split}-" in lowered:
                matching.append(path)
        return matching or parquet_files

    def _parquet_paths(self) -> list[Path]:
        token = resolve_hf_token()
        return [
            Path(
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=repo_file,
                    repo_type="dataset",
                    revision=self.revision,
                    token=token,
                    cache_dir=None if self.cache_dir is None else str(self.cache_dir),
                )
            )
            for repo_file in self._matching_parquet_files()
        ]

    def fingerprint(self) -> str:
        files = self._matching_parquet_files()
        payload = {
            "repo_id": self.repo_id,
            "config": self.config,
            "split": self.split,
            "revision": self.revision,
            "files": files,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def iter_rows(self, *, limit: int | None = None, batch_size: int = 1024) -> Iterator[dict[str, Any]]:
        if limit is not None and int(limit) < 0:
            raise ValueError("source limit must be >= 0")
        if batch_size <= 0:
            raise ValueError("source batch_size must be > 0")

        paths = self._parquet_paths()
        lazy = pl.scan_parquet(paths).select(list(HABR_SOURCE_COLUMNS))
        if limit is not None:
            lazy = lazy.limit(int(limit))

        for batch in lazy.collect_batches(chunk_size=batch_size, maintain_order=True):
            yield from batch.iter_rows(named=True)


__all__ = ["HABR_SOURCE_COLUMNS", "HF_TOKEN_ENV_NAMES", "HabrSource", "resolve_hf_token"]
