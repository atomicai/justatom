import os
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import polars as pl
import simplejson as json
from loguru import logger

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None  # type: ignore

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except Exception:
    hf_hub_download = None  # type: ignore
    list_repo_files = None  # type: ignore

from justatom.configuring.builtins import resolve_builtin_path
from justatom.etc.pattern import singleton
from justatom.storing.mask import IDataset


_HF_REPO_DATASET_RE = re.compile(r"^[^/\s]+/[^/\s]+(?:\?.*)?$")


def _looks_like_hf_repo_dataset(value: str) -> bool:
    if value.startswith(("hf://", "builtin://", "http://", "https://")):
        return False
    if ":" in value:
        return False
    return bool(_HF_REPO_DATASET_RE.match(value))


class URLInJSONDataset(IDataset):
    """Dataset to fetch via url in json format"""

    def iterator(self, **kwargs):
        pass


class JUSTATOMDataset(IDataset):
    def iterator(self, lazy: bool = False, **kwargs):
        kwargs.pop("split", None)
        kwargs.pop("limit", None)
        dataset_path = Path(os.getcwd()) / ".data" / "polaroids.ai.data.json"
        return JSONDataset(fp=str(dataset_path)).iterator(lazy=lazy, **kwargs)


class JSONDataset(IDataset):
    def __init__(self, fp: str):
        self.fp = fp

    def iterator(self, lazy: bool = False, **kwargs) -> pl.LazyFrame | pl.DataFrame:
        kwargs.pop("split", None)
        kwargs.pop("limit", None)
        if lazy:
            logger.warning(
                "lazy=True for .json is unsupported; using eager read for [{}]. " "Prefer .jsonl/.parquet for large datasets.",
                self.fp,
            )

        with open(self.fp, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return pl.from_dicts([row for row in data if row is not None])
        if isinstance(data, dict):
            maybe_rows = data.get("data")
            if isinstance(maybe_rows, list):
                return pl.from_dicts([row for row in maybe_rows if row is not None])
            return pl.from_dicts([data])
        return pl.from_dicts([])


class JSONLinesDataset(IDataset):
    def __init__(self, fp: str):
        self.fp = fp

    def iterator(self, lazy: bool = False, **kwargs) -> pl.LazyFrame | pl.DataFrame:
        kwargs.pop("split", None)
        kwargs.pop("limit", None)
        if lazy:
            try:
                return pl.scan_ndjson(self.fp, **kwargs)
            except Exception:
                logger.warning("Falling back to eager NDJSON loading for file [{}].", self.fp)

        try:
            import jsonlines
        except Exception as ex:
            msg = "To read .jsonl files please install `jsonlines` package."
            logger.error(msg)
            raise ImportError(msg) from ex

        with jsonlines.open(self.fp, mode="r") as reader:
            rows = [row for row in reader if row is not None]
        frame = pl.from_dicts(rows)
        return frame.lazy() if lazy else frame


class PARQUETDataset(IDataset):
    def __init__(self, fp: str):
        self.fp = fp

    def iterator(self, lazy: bool = False, **kwargs) -> pl.LazyFrame | pl.DataFrame:
        kwargs.pop("split", None)
        kwargs.pop("limit", None)
        kwargs.pop("drop_columns", None)
        if lazy:
            try:
                return pl.scan_parquet(self.fp, **kwargs)
            except Exception:
                logger.warning("Falling back to eager Parquet loading for file [{}].", self.fp)
        try:
            return pl.scan_parquet(self.fp, **kwargs) if lazy else pl.read_parquet(self.fp, **kwargs)
        except Exception as ex:
            logger.error("Failed to load Parquet file [{}]: {}", self.fp, ex)
            raise ex


class CSVDataset(IDataset):
    def __init__(self, fp: str):
        self.fp = fp

    def iterator(self, lazy: bool = False, **kwargs) -> pl.LazyFrame | pl.DataFrame:
        kwargs.pop("split", None)
        kwargs.pop("limit", None)
        if lazy:
            return pl.scan_csv(self.fp, **kwargs)
        return pl.read_csv(self.fp, **kwargs)


class XLSXDataset(IDataset):
    def __init__(self, fp: str):
        self.fp = fp

    def iterator(self, **kwargs) -> pl.DataFrame:
        kwargs.pop("split", None)
        kwargs.pop("limit", None)
        if "lazy" in kwargs:
            logger.warning("Lazy loading is not supported for XLSXDataset, ignoring the `lazy` argument.")
            kwargs.pop("lazy", None)
        return pl.read_excel(self.fp, **kwargs)


class HFDataset(IDataset):
    def __init__(self, ref: str):
        self.ref = ref

    @staticmethod
    def _parse_ref(ref: str) -> tuple[str, str | None, str | None, bool | None]:
        parsed = urlparse(ref)
        dataset_name = f"{parsed.netloc}{parsed.path}".strip("/")
        query = parse_qs(parsed.query)
        split = query.get("split", [None])[0]
        config_name = query.get("config", [None])[0]
        streaming_raw = query.get("streaming", [None])[0]
        streaming = None
        if streaming_raw is not None:
            streaming = str(streaming_raw).strip().lower() in {"1", "true", "yes"}
        return dataset_name, config_name, split, streaming

    @staticmethod
    def _split_candidates(split: str | None) -> list[str]:
        raw = "train" if split is None else str(split)
        candidates = [part.strip() for part in raw.split("|") if part.strip()]
        return candidates or ["train"]

    @staticmethod
    def _format_split_error(dataset_name: str, split: str, exc: Exception) -> ValueError:
        return ValueError(
            "Failed to load HF dataset "
            f"{dataset_name!r} with split {split!r}. {exc} "
            "If split names differ, pass fallback chain like `train|test` or `dev|test`."
        )

    @staticmethod
    @lru_cache(maxsize=64)
    def _repo_files(dataset_name: str) -> tuple[str, ...]:
        if list_repo_files is None:
            return ()
        try:
            return tuple(list_repo_files(repo_id=dataset_name, repo_type="dataset"))
        except Exception as ex:
            logger.warning("Failed to inspect HF dataset repo files for [{}]: {}", dataset_name, ex)
            return ()

    @staticmethod
    def _parquet_files_for_split(repo_files: tuple[str, ...], split: str) -> list[str]:
        split_name = split.strip().lower()
        matches: list[str] = []
        for repo_file in repo_files:
            repo_file_lower = repo_file.lower()
            basename = Path(repo_file_lower).name
            if not repo_file_lower.endswith(".parquet"):
                continue
            if basename == f"{split_name}.parquet" or basename.startswith(f"{split_name}-"):
                matches.append(repo_file)
                continue
            if f"/{split_name}/" in repo_file_lower or f"/{split_name}-" in repo_file_lower:
                matches.append(repo_file)
        return sorted(matches)

    @staticmethod
    def _drop_columns_from_result(data, drop_columns: list[str]):
        if not drop_columns:
            return data
        if isinstance(data, pl.DataFrame):
            existing_to_drop = [col for col in drop_columns if col in data.columns]
            return data.drop(existing_to_drop) if existing_to_drop else data
        if isinstance(data, pl.LazyFrame):
            existing_to_drop = [col for col in drop_columns if col in data.collect_schema().names()]
            return data.drop(existing_to_drop) if existing_to_drop else data
        existing_to_drop = [col for col in drop_columns if col in getattr(data, "column_names", [])]
        if existing_to_drop:
            return data.remove_columns(existing_to_drop)
        return data

    def _load_parquet_fallback(self, dataset_name: str, split: str, lazy: bool):
        if hf_hub_download is None:
            return None

        repo_files = self._repo_files(dataset_name)
        parquet_files = self._parquet_files_for_split(repo_files, split)
        if not parquet_files:
            return None

        local_paths: list[str] = []
        for repo_file in parquet_files:
            try:
                local_paths.append(hf_hub_download(repo_id=dataset_name, filename=repo_file, repo_type="dataset"))
            except Exception as ex:
                logger.warning(
                    "Failed to cache HF parquet shard [{}] from [{}]: {}",
                    repo_file,
                    dataset_name,
                    ex,
                )
                return None

        logger.warning(
            "HF dataset [{}] split [{}] is falling back to [{}] cached parquet shard(s).",
            dataset_name,
            split,
            len(local_paths),
        )
        return pl.scan_parquet(local_paths) if lazy else pl.read_parquet(local_paths)

    def iterator(self, lazy: bool = False, **kwargs):
        if load_dataset is None:
            msg = "To read Hugging Face datasets please install `datasets` package. " "Example: pip install datasets"
            logger.error(msg)
            raise ImportError(msg)

        split_override = kwargs.pop("split", None)
        kwargs.pop("limit", None)
        drop_columns_raw = kwargs.pop("drop_columns", None)
        drop_columns: list[str] = []
        if isinstance(drop_columns_raw, str):
            drop_columns = [drop_columns_raw]
        elif isinstance(drop_columns_raw, (list, tuple, set)):
            drop_columns = [str(col) for col in drop_columns_raw if str(col).strip()]
        dataset_name, config_name, split, streaming = self._parse_ref(self.ref)
        if streaming is not None:
            logger.warning(
                "HF query param `streaming` is ignored by design. "
                "Dataset is always loaded with streaming=False (local cache semantics)."
            )
        effective_streaming = False
        split_candidates = self._split_candidates(split_override or split)

        last_error: Exception | None = None
        for candidate in split_candidates:
            try:
                ds = load_dataset(
                    dataset_name,
                    name=config_name,
                    split=candidate,
                    streaming=effective_streaming,
                    **kwargs,
                )
                return self._drop_columns_from_result(ds, drop_columns)
            except Exception as ex:
                last_error = ex
                parquet_fallback = self._load_parquet_fallback(dataset_name=dataset_name, split=candidate, lazy=lazy)
                if parquet_fallback is not None:
                    return self._drop_columns_from_result(parquet_fallback, drop_columns)
                if isinstance(ex, ValueError) and len(split_candidates) == 1:
                    raise self._format_split_error(dataset_name, candidate, ex) from ex
                logger.warning(
                    "Failed to load HF dataset [{}] split [{}], trying next candidate. Error: {}",
                    dataset_name,
                    candidate,
                    ex,
                )

        assert last_error is not None
        raise last_error


@singleton
class ByName:
    def named(self, name: str, **kwargs):
        ops = ["url", "justatom", "hf://<dataset>?split=train", "<owner>/<dataset>"]

        if name == "justatom":
            return JUSTATOMDataset(**kwargs)
        if name == "url":
            return URLInJSONDataset(**kwargs)
        if str(name).startswith("hf://"):
            return HFDataset(ref=str(name))

        fp = resolve_builtin_path(name)
        if not fp.exists() and _looks_like_hf_repo_dataset(str(name)):
            return HFDataset(ref=f"hf://{name}")
        if not fp.exists():
            msg = (
                f"Unknown dataset_name_or_path=[{name}] to init IDataset instance. "
                f"Use one of {','.join(ops)} or provide valid dataset path"
            )
            logger.error(msg)
            raise ValueError(msg)

        if fp.suffix in [".csv"]:
            return CSVDataset(fp=str(fp))
        if fp.suffix in [".xlsx"]:
            return XLSXDataset(fp=str(fp))
        if fp.suffix in [".json"]:
            return JSONDataset(fp=str(fp))
        if fp.suffix in [".jsonl"]:
            return JSONLinesDataset(fp=str(fp))
        if fp.suffix in [".parquet"]:
            return PARQUETDataset(fp=str(fp))

        msg = f"File exists however loading from the [{fp.suffix}] file is not supported"
        logger.error(msg)
        raise ValueError(msg)


API = ByName()


__all__ = ["API"]
