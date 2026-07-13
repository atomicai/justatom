from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import polars as pl
from huggingface_hub import HfApi, hf_hub_download, list_repo_files


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


def promote_hf_token_env(env: dict[str, str] | None = None) -> bool:
    active_env = os.environ if env is None else env
    if active_env.get("HF_TOKEN"):
        return False
    token = resolve_hf_token(active_env)
    if not token:
        return False
    active_env["HF_TOKEN"] = token
    return True


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
        self._resolved_revision: str | None = None

    def resolved_revision(self) -> str:
        if self._resolved_revision is None:
            info = HfApi(token=resolve_hf_token()).dataset_info(
                repo_id=self.repo_id,
                revision=self.revision,
            )
            if not info.sha:
                raise RuntimeError(f"Could not resolve dataset revision {self.revision!r} for {self.repo_id!r}.")
            self._resolved_revision = str(info.sha)
        return self._resolved_revision

    def _repo_files(self) -> list[str]:
        return list(
            list_repo_files(
                repo_id=self.repo_id,
                repo_type="dataset",
                revision=self.resolved_revision(),
                token=resolve_hf_token(),
            )
        )

    def _matching_parquet_files(self) -> list[str]:
        parquet_files = sorted(path for path in self._repo_files() if path.lower().endswith(".parquet"))
        if not parquet_files:
            raise RuntimeError(f"No parquet files found in dataset repo {self.repo_id!r}.")

        candidates = parquet_files
        config = self.config.casefold()
        if config != "default":
            candidates = [
                path
                for path in candidates
                if config in {part.casefold() for part in Path(path).parts[:-1]}
            ]
            if not candidates:
                raise RuntimeError(
                    f"No parquet files found for config {self.config!r} in dataset repo {self.repo_id!r}."
                )

        split = self.split.casefold()
        matching = []
        for path in candidates:
            lowered = path.casefold()
            basename = Path(lowered).name
            if basename == f"{split}.parquet" or basename.startswith(f"{split}-"):
                matching.append(path)
            elif f"/{split}/" in lowered or f"/{split}-" in lowered:
                matching.append(path)
        if not matching:
            raise RuntimeError(
                f"No parquet files found for split {self.split!r} and config {self.config!r} "
                f"in dataset repo {self.repo_id!r}."
            )
        return matching

    def _parquet_paths(self) -> list[Path]:
        return list(self._iter_parquet_paths())

    def _download_parquet(self, repo_file: str) -> Path:
        token = resolve_hf_token()
        return Path(
            hf_hub_download(
                repo_id=self.repo_id,
                filename=repo_file,
                repo_type="dataset",
                revision=self.resolved_revision(),
                token=token,
                cache_dir=None if self.cache_dir is None else str(self.cache_dir),
            )
        )

    def _iter_parquet_paths(self) -> Iterator[Path]:
        for repo_file in self._matching_parquet_files():
            yield self._download_parquet(repo_file)

    def fingerprint(self) -> str:
        files = self._matching_parquet_files()
        payload = {
            "repo_id": self.repo_id,
            "config": self.config,
            "split": self.split,
            "revision": self.revision,
            "resolved_revision": self.resolved_revision(),
            "files": files,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def iter_rows(self, *, limit: int | None = None, batch_size: int = 1024) -> Iterator[dict[str, Any]]:
        if limit is not None and int(limit) < 0:
            raise ValueError("source limit must be >= 0")
        if batch_size <= 0:
            raise ValueError("source batch_size must be > 0")

        remaining = None if limit is None else int(limit)
        if remaining == 0:
            return
        for path in self._iter_parquet_paths():
            lazy = pl.scan_parquet(path).select(list(HABR_SOURCE_COLUMNS))
            if remaining is not None:
                lazy = lazy.limit(remaining)
            for batch in lazy.collect_batches(chunk_size=batch_size, maintain_order=True):
                rows = batch.to_dicts()
                yield from rows
                if remaining is not None:
                    remaining -= len(rows)
                    if remaining <= 0:
                        return


__all__ = [
    "HABR_SOURCE_COLUMNS",
    "HF_TOKEN_ENV_NAMES",
    "HabrSource",
    "promote_hf_token_env",
    "resolve_hf_token",
]
