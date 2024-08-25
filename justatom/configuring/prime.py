import os
from pathlib import Path

import dotenv
import yaml
from dotmap import DotMap
from loguru import logger

from justatom.etc.pattern import singleton


@singleton
class IConfig:
    DEFAULT_UMAP = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")

    DEFAULT_PCA = dict()

    DEFAULT_HDBSCAN = dict()

    DEFAULT_KMEANS = dict()

    def __init__(self):
        try:
            with open(str(Path(os.getcwd()) / "config.yaml")) as fp:
                config = DotMap(yaml.safe_load(fp))
        except:  # noqa: E722
            config = DotMap()
            logger.info("Config file is not loaded, be aware that no options are propagates from `config.yaml`")
        else:
            for k, v in config.items():
                if k.startswith("_"):
                    continue
                setattr(self, k, v)
        dotenv.load_dotenv()


Config = IConfig()


__all__ = ["Config"]
