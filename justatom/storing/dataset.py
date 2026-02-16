import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import polars as pl
import simplejson as json
from loguru import logger

from justatom.etc.pattern import singleton
from justatom.storing.mask import IDataset


class URLInJSONDataset(IDataset):
    """Dataset to fetch via url in json format"""

    def iterator(self, **kwargs) -> Generator:
        pass


class JUSTATOMDataset(IDataset):
    def iterator(self, **kwargs) -> Generator:
        with open(Path(os.getcwd()) / ".data" / "polaroids.ai.data.json") as fp:
            docs = json.load(fp)
            for doc in docs:  # noqa: UP028
                yield doc


class JSONDataset(IDataset):
    def __init__(self, fp):
        self.fp = fp

    def iterator(
        self, lazy: bool = False, **kwargs
    ) -> pl.LazyFrame | pl.DataFrame:
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

        return frame.lazy() if lazy else frame


class JSONLinesDataset(IDataset):
    def __init__(self, fp):
        self.fp = fp

    def iterator(
        self, lazy: bool = False, **kwargs
    ) -> pl.LazyFrame | pl.DataFrame:
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

    def iterator(
        self, lazy: bool = False, **kwargs
    ) -> pl.LazyFrame | pl.DataFrame:
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

    def iterator(
        self, lazy: bool = False, **kwargs
    ) -> pl.LazyFrame | pl.DataFrame:
        if lazy:
            return pl.scan_csv(self.fp, **kwargs)
        return pl.read_csv(self.fp, **kwargs)


class XLSXDataset(IDataset):
    def __init__(self, fp):
        self.fp = fp

    def iterator(self, **kwargs) -> pl.DataFrame:
        if "lazy" in kwargs:
            logger.warning("Lazy loading is not supported for XLSXDataset, ignoring the 'lazy' argument.")
        pl_view = pl.read_excel(self.fp, **kwargs)
        return pl_view


@singleton
class ByName:
    def named(self, name: str, **kwargs):
        OPS = ["url", "justatom"]

        if name == "justatom":
            klass = JUSTATOMDataset
        elif name == "url":
            klass = URLInJSONDataset
        else:
            fp = Path(name)
            if not fp.exists():
                msg = f"Unknown dataset_name_or_path=[{name}] to init IDataset instance. Use one of {','.join(OPS)} or provide valid dataset path"  # noqa: E501
                logger.error(msg)
                raise ValueError(msg)
            if fp.suffix in [".csv"]:
                return CSVDataset(fp=name)
            elif fp.suffix in [".xlsx"]:
                return XLSXDataset(fp=name)
            elif fp.suffix in [".json"]:
                return JSONDataset(fp=name)
            elif fp.suffix in [".jsonl"]:
                return JSONLinesDataset(fp=name)
            elif fp.suffix in [".parquet"]:
                return PARQUETDataset(fp=name)
            else:
                msg = f"File exists however loading from the [{fp.suffix}] file is not supported"
                logger.error(msg)
                raise ValueError(msg)
        return klass(**kwargs)


API = ByName()


__all__ = ["API"]
