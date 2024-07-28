from justatom.storing.mask import IDataset
from justatom.etc.pattern import singleton
from typing import Generator
import simplejson as json
from pathlib import Path
from loguru import logger
import polars as pl
import os


class URLInJSONDataset(IDataset):
    """Dataset to fetch via url in json format"""

    def iterator(self, **kwargs) -> Generator:
        pass


class JUSTATOMDataset(IDataset):

    def iterator(self, **kwargs) -> Generator:
        with open(Path(os.getcwd()) / ".data" / "polaroids.ai.data.json") as fp:
            docs = json.load(fp)
            for doc in docs:
                yield doc


class CSVDataset(IDataset):

    def __init__(self, fp):
        self.fp = fp

    def iterator(self, **kwargs) -> pl.DataFrame:
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
                msg = f"Unknown dataset_name_or_path=[{name}] to init IDataset instance. Use one of {','.join(OPS)} or provide valid dataset path"
                logger.error(msg)
                raise ValueError(msg)
            if fp.suffix in [".csv"]:
                return CSVDataset(fp=name)
            elif fp.suffix in [".xlsx"]:
                return XLSXDataset(fp=name)
            else:
                msg = f"File exists however loading from the [{fp.suffix}] file is not supported"
                logger.error(msg)
                raise ValueError(msg)
        return klass(**kwargs)


API = ByName()


__all__ = ["API"]
