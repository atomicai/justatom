import abc

import numpy as np
from loguru import logger

try:
    from numba import njit  # pylint: disable=import-error
except (ImportError, ModuleNotFoundError):
    logger.debug("Numba not found, replacing njit() with no-op implementation. Enable it with 'pip install numba'.")

    def njit(f):
        return f


@njit  # (fastmath=True)
def expit(x: float) -> float:
    return 1 / (1 + np.exp(-x))


@njit  # (fastmath=True)
def _normalize_embedding_1D(emb: np.ndarray) -> None:
    norm = np.sqrt(emb.dot(emb))  # faster than np.linalg.norm()
    if norm != 0.0:
        emb /= norm


@njit  # (fastmath=True)
def _normalize_embedding_2D(emb: np.ndarray) -> None:
    for vec in emb:
        vec = np.ascontiguousarray(vec)
        norm = np.sqrt(vec.dot(vec))
        if norm != 0.0:
            vec /= norm


def scale_to_unit_interval(self, score: float, similarity: str | None) -> float:
    if similarity == "cosine":
        return (score + 1) / 2
    else:
        return float(expit(score / 100))


class IEVENTDocStore(abc.ABC):
    @abc.abstractmethod
    async def add_event(self, e):
        pass

    @abc.abstractmethod
    async def add_user(self, username, creds, uuid):
        pass

    @abc.abstractmethod
    async def del_user(self, uuid):
        pass

    @abc.abstractmethod
    async def add_document(self, doc):
        pass

    @abc.abstractmethod
    async def del_document(self, uuid):
        pass


class IDFDocStore(abc.ABC):
    @abc.abstractmethod
    def counts_per_col(self, col):
        pass

    @abc.abstractmethod
    def parse_metrics_per_col(self, col):
        pass

    @abc.abstractmethod
    def samples_per_col(self, col, n_samples):
        pass


__all__ = ["IEVENTDocStore", "IDFDocStore"]
