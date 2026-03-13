import os
from collections.abc import Generator
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import polars as pl
import simplejson as json
from datasets import load_dataset
from loguru import logger

from justatom.configuring.builtins import resolve_builtin_path
from justatom.etc.pattern import singleton
from justatom.storing.mask import IDataset


class URLInJSONDataset(IDataset):
    """Dataset to fetch via url in json format"""

    def iterator(self, **kwargs) -> Generator:
        pass


class JUSTATOMDataset(IDataset):
    def iterator(self, **kwargs) -> Generator:
        kwargs.pop("split", None)
        kwargs.pop("limit", None)
        dataset_path = Path(os.getcwd()) / ".data" / "polaroids.ai.data.json"
        if kwargs.get("lazy"):
            return JSONDataset(fp=dataset_path).iterator(lazy=True)

        with open(dataset_path, encoding="utf-8") as fp:
            docs = json.load(fp)
        return [doc for doc in docs if doc is not None]


class JSONDataset(IDataset):
    def __init__(self, fp):
        self.fp = fp

    def iterator(self, lazy: bool = False, **kwargs) -> pl.LazyFrame | pl.DataFrame:
        kwargs.pop("split", None)
        kwargs.pop("limit", None)
        if lazy:
            logger.warning(
                "lazy=True for .json is unsupported; falling back to eager JSON loading "
                f"for file [{self.fp}]. Use .jsonl or .parquet for large datasets."
            )

        with open(self.fp, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            frame = pl.from_dicts([row for row in data if row is not None])
        elif isinstance(data, dict):
            maybe_rows = data.get("data")
            if isinstance(maybe_rows, list):
                frame = pl.from_dicts([row for row in maybe_rows if row is not None])
            else:
                frame = pl.from_dicts([data])
        else:
            frame = pl.from_dicts([])

        return frame


class JSONLinesDataset(IDataset):
    def __init__(self, fp):
        self.fp = fp

    def iterator(self, lazy: bool = False, **kwargs) -> pl.LazyFrame | pl.DataFrame:
        kwargs.pop("split", None)
        kwargs.pop("limit", None)
        if lazy:
            try:
                return pl.scan_ndjson(self.fp, **kwargs)
            except Exception:
                logger.warning(f"Falling back to eager NDJSON loading for file [{self.fp}].")

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
    def __init__(self, fp):
        self.fp = fp

    def iterator(self, lazy: bool = False, **kwargs) -> pl.LazyFrame | pl.DataFrame:
        kwargs.pop("split", None)
        kwargs.pop("limit", None)
        if lazy:
            try:
                return pl.scan_parquet(self.fp, **kwargs)
            except Exception:
                logger.warning(f"Falling back to eager Parquet loading for file [{self.fp}].")
        try:
            return pl.scan_parquet(self.fp, **kwargs) if lazy else pl.read_parquet(self.fp, **kwargs)
        except Exception as ex:
            logger.error(f"Failed to load Parquet file [{self.fp}]: {ex}")
            raise ex


class CSVDataset(IDataset):
    def __init__(self, fp):
        self.fp = fp

    def iterator(self, lazy: bool = False, **kwargs) -> pl.LazyFrame | pl.DataFrame:
        kwargs.pop("split", None)
        kwargs.pop("limit", None)
        if lazy:
            return pl.scan_csv(self.fp, **kwargs)
        return pl.read_csv(self.fp, **kwargs)


class XLSXDataset(IDataset):
    def __init__(self, fp):
        self.fp = fp

    def iterator(self, **kwargs) -> pl.DataFrame:
        kwargs.pop("split", None)
        kwargs.pop("limit", None)
        if "lazy" in kwargs:
            logger.warning("Lazy loading is not supported for XLSXDataset, ignoring the 'lazy' argument.")
        pl_view = pl.read_excel(self.fp, **kwargs)
        return pl_view


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
    def _format_split_error(
        dataset_name: str,
        split: str,
        exc: Exception,
    ) -> ValueError:
        return ValueError(
            "Failed to load HF dataset "
            f"{dataset_name!r} with split {split!r}. {exc} "
            "If the dataset may expose different split names, use a fallback chain "
            "such as `train|test` or `dev|test`."
        )

    def iterator(self, lazy: bool = False, **kwargs):
        split_override = kwargs.pop("split", None)
        kwargs.pop("limit", None)
        dataset_name, config_name, split, streaming = self._parse_ref(self.ref)
        effective_streaming = lazy if streaming is None else streaming
        split_candidates = self._split_candidates(split_override or split)

        last_error: Exception | None = None
        for candidate in split_candidates:
            try:
                return load_dataset(
                    dataset_name,
                    name=config_name,
                    split=candidate,
                    streaming=effective_streaming,
                    **kwargs,
                )
            except ValueError as ex:
                last_error = ex
                if len(split_candidates) == 1:
                    raise self._format_split_error(dataset_name, candidate, ex) from ex
                logger.warning(
                    "Failed to load HF dataset [{}] split [{}], trying next candidate.",
                    dataset_name,
                    candidate,
                )

        assert last_error is not None
        raise last_error


@singleton
class ByName:
    def named(self, name: str, **kwargs):
        OPS = ["url", "justatom", "hf://<dataset>?split=train"]

        if name == "justatom":
            klass = JUSTATOMDataset
        elif name == "url":
            klass = URLInJSONDataset
        elif str(name).startswith("hf://"):
            klass = HFDataset
        else:
            fp = resolve_builtin_path(name)
            if not fp.exists():
                msg = f"Unknown dataset_name_or_path=[{name}] to init IDataset instance. Use one of {','.join(OPS)} or provide valid dataset path"  # noqa: E501
                logger.error(msg)
                raise ValueError(msg)
            if fp.suffix in [".csv"]:
                return CSVDataset(fp=str(fp))
            elif fp.suffix in [".xlsx"]:
                return XLSXDataset(fp=str(fp))
            elif fp.suffix in [".json"]:
                return JSONDataset(fp=str(fp))
            elif fp.suffix in [".jsonl"]:
                return JSONLinesDataset(fp=str(fp))
            elif fp.suffix in [".parquet"]:
                return PARQUETDataset(fp=str(fp))
            else:
                msg = f"File exists however loading from the [{fp.suffix}] file is not supported"
                logger.error(msg)
                raise ValueError(msg)
        if klass is HFDataset:
            return klass(ref=str(name))
        return klass(**kwargs)


API = ByName()


__all__ = ["API"]
