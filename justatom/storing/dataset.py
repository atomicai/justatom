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
    ) -> pl.DataFrame | Generator[dict[str, Any], None, None]:
        with open(self.fp, encoding="utf-8") as f:
            data = json.load(f)

        if not lazy:
            if isinstance(data, list):
                return pl.from_dicts(data)
            if isinstance(data, dict):
                maybe_rows = data.get("data")
                if isinstance(maybe_rows, list):
                    return pl.from_dicts(maybe_rows)
                return pl.from_dicts([data])
            return pl.from_dicts([])

        def _iter() -> Generator[dict[str, Any], None, None]:
            if isinstance(data, list):
                for row in data:
                    if row is not None:
                        yield row
            elif isinstance(data, dict):
                maybe_rows = data.get("data")
                if isinstance(maybe_rows, list):
                    for row in maybe_rows:
                        if row is not None:
                            yield row
                else:
                    yield data

        return _iter()


class JSONLinesDataset(IDataset):
    def __init__(self, fp):
        self.fp = fp

    def iterator(
        self, lazy: bool = False, **kwargs
    ) -> pl.DataFrame | Generator[dict[str, Any], None, None]:
        try:
            import jsonlines
        except Exception as ex:
            msg = "To read .jsonl files please install `jsonlines` package."
            logger.error(msg)
            raise ImportError(msg) from ex

        if not lazy:
            with jsonlines.open(self.fp, mode="r") as reader:
                rows = [row for row in reader if row is not None]
            return pl.from_dicts(rows)

        def _iter() -> Generator[dict[str, Any], None, None]:
            with jsonlines.open(self.fp, mode="r") as reader:
                for row in reader:
                    if row is not None:
                        yield row

        return _iter()


class CSVDataset(IDataset):
    def __init__(self, fp):
        self.fp = fp

    def iterator(
        self, lazy: bool = False, **kwargs
    ) -> pl.LazyFrame | pl.DataFrame | Generator[dict[str, Any], None, None]:
        if lazy:
            pl_view = pl.scan_csv(self.fp, **kwargs)

            def _iter() -> Generator[dict[str, Any], None, None]:
                for row in pl_view.collect(streaming=True).iter_rows(named=True):
                    yield row

            return _iter()
        else:
            pl_view = pl.read_csv(self.fp, **kwargs)
        return pl_view


class XLSXDataset(IDataset):
    def __init__(self, fp):
        self.fp = fp

    def iterator(self, **kwargs) -> pl.DataFrame:
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
            else:
                msg = f"File exists however loading from the [{fp.suffix}] file is not supported"
                logger.error(msg)
                raise ValueError(msg)
        return klass(**kwargs)


API = ByName()


__all__ = ["API"]
