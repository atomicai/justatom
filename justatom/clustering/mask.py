import abc

import numpy as np


class IDimReducer(abc.ABC):  # noqa: B024
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:  # noqa: B027
        pass

    def transform(self, embeddings: np.ndarray) -> np.ndarray:  # noqa: B027
        pass


__all__ = ["IDimReducer"]
