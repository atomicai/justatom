from justatom.storing.mask import IDataset
from justatom.etc.pattern import singleton
from typing import Generator
import simplejson as json
from pathlib import Path
from loguru import logger
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


@singleton
class ByName:

    def named(self, name: str, **kwargs):
        OPS = ["url", "justatom"]

        if name == "justatom":
            klass = JUSTATOMDataset
        elif name == "url":
            klass = URLInJSONDataset
        else:
            msg = f"Unknown name=[{name}] to init IDataset instance. Use one of {','.join(OPS)}"
            logger.error(msg)
            raise ValueError(msg)

        return klass(**kwargs)


API = ByName()


__all__ = ["API"]
