import abc
from typing import Dict, List, Optional
from loguru import logger
import numpy as np

from justatom.etc.filter import FilterType
from justatom.etc.schema import Document
from justatom.etc.errors import DuplicateDocumentError


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


def scale_to_unit_interval(self, score: float, similarity: Optional[str]) -> float:
    if similarity == "cosine":
        return (score + 1) / 2
    else:
        return float(expit(score / 100))


class IDBDocStore(abc.ABC):

    @abc.abstractmethod
    def add_event(self, e):
        pass

    @abc.abstractmethod
    def add_user(self, username, creds, uuid):
        pass

    @abc.abstractmethod
    def del_user(self, uuid):
        pass

    @abc.abstractmethod
    def add_document(self, doc):
        pass

    @abc.abstractmethod
    def del_document(self, uuid):
        pass


class INNDocStore(abc.ABC):

    index: Optional[str]
    similarity: Optional[str]
    duplicate_documents_options: tuple = ("skip", "overwrite", "fail")
    ids_iterator = None

    @abc.abstractclassmethod
    def write_documents(self, docs, index: str = None):
        pass

    @abc.abstractclassmethod
    def get_document_by_id(
        self, id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ) -> Optional[Document]:
        pass

    @abc.abstractmethod
    def get_document_count(
        self,
        filters: Optional[FilterType] = None,
        index: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> int:
        pass

    def __iter__(self):
        if not self.ids_iterator:
            self.ids_iterator = [x.id for x in self.get_all_documents()]
        return self

    def describe_documents(self, index=None):
        """
        Statistics of the documents
        """
        if index is None:
            index = self.index
        docs = self.get_all_documents(index)
        lens = [len(d.content) for d in docs]
        response = dict(
            count=len(docs),
            chars_mean=np.mean(lens),
            chars_max=max(lens),
            chars_min=min(lens),
            chars_median=np.median(lens),
        )
        return response

    def __next__(self):
        if len(self.ids_iterator) == 0:
            raise StopIteration
        curr_id = self.ids_iterator[0]
        ret = self.get_document_by_id(curr_id)
        self.ids_iterator = self.ids_iterator[1:]
        return ret

    def _drop_duplicate_documents(self, documents: List[Document], index: Optional[str] = None) -> List[Document]:
        """
        Drop duplicates documents based on same hash ID
        :param documents: A list of Document objects.
        :param index: name of the index
        :return: A list of Document objects.
        """
        _hash_ids = set([])
        _documents: List[Document] = []

        for document in documents:
            if document.id in _hash_ids:
                logger.info(
                    "Duplicate Documents: Document with id '%s' already exists in index '%s'",
                    document.id,
                    index or self.index,
                )
                continue
            _documents.append(document)
            _hash_ids.add(document.id)

        return _documents

    def _handle_duplicate_documents(
        self,
        documents: List[Document],
        index: Optional[str] = None,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Checks whether any of the passed documents is already existing in the chosen index and returns a list of
        documents that are not in the index yet.
        :param documents: A list of Document objects.
        :param index: name of the index
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip (default option): Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        :return: A list of Document objects.
        """

        index = index or self.index
        if duplicate_documents in ("skip", "fail"):
            documents = self._drop_duplicate_documents(documents, index)
            documents_found = self.get_documents_by_id(ids=[doc.id for doc in documents], index=index, headers=headers)
            ids_exist_in_db: List[str] = [doc.id for doc in documents_found]

            if len(ids_exist_in_db) > 0 and duplicate_documents == "fail":
                raise DuplicateDocumentError(
                    f"Document with ids '{', '.join(ids_exist_in_db)} already exists in index = '{index}'."
                )

            documents = list(filter(lambda doc: doc.id not in ids_exist_in_db, documents))

        return documents

    @abc.abstractmethod
    def nn_search(self, **props):
        pass

    @abc.abstractmethod
    def sp_search(self, **props):
        pass

    @abc.abstractmethod
    def ai_search(self, **props):
        pass


__all__ = ["IDocStore"]
