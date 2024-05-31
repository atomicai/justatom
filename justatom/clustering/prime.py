import numpy as np
from umap import UMAP

from justatom.clustering.mask import IDimReducer


class IUMAPDimReducer(IDimReducer):
    def __init__(self, **kwargs):
        self.umap = UMAP(**kwargs)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        embs = self.umap.fit_transform(embeddings)
        return embs

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        embs = self.umap.transform(embeddings)
        return embs


__all__ = ["IUMAPDimReducer"]
