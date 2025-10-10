import os
from pathlib import Path

import dotenv
from envyaml import EnvYAML
from loguru import logger

from justatom.etc.pattern import singleton


@singleton
class IConfig:
    DEFAULT_UMAP = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")

    DEFAULT_PCA = dict()

    DEFAULT_HDBSCAN = dict()

    DEFAULT_KMEANS = dict()

    def __init__(self):
        config = dict(EnvYAML(Path(os.getcwd()) / "config.yaml"))
        for k, v in config.items():
            if k.startswith("_"):
                continue
            setattr(self, k, v)
        dotenv.load_dotenv()


Config = IConfig()


__all__ = ["Config"]
