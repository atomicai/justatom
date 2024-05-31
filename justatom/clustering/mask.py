import abc

import numpy as np


class IDimReducer(abc.ABC):

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        pass

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        pass


__all__ = ["IDimReducer"]
