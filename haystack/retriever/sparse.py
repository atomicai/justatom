import logging
from collections import OrderedDict
from typing import List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from haystack.document_store.base import BaseDocumentStore
from haystack import Document
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.base import BaseRetriever
from collections import namedtuple

import numpy as np

from scipy import sparse

logger = logging.getLogger(__name__)


class ElasticsearchRetriever(BaseRetriever):
    def __init__(self, document_store: ElasticsearchDocumentStore, top_k: int = 10, custom_query: str = None):
        """
        :param document_store: an instance of a DocumentStore to retrieve documents from.
        :param custom_query: query string as per Elasticsearch DSL with a mandatory query placeholder(query).

                             Optionally, ES `filter` clause can be added where the values of `terms` are placeholders
                             that get substituted during runtime. The placeholder(${filter_name_1}, ${filter_name_2}..)
                             names must match with the filters dict supplied in self.retrieve().
                             ::

                                 **An example custom_query:**
                                 ```python
                                |    {
                                |        "size": 10,
                                |        "query": {
                                |            "bool": {
                                |                "should": [{"multi_match": {
                                |                    "query": ${query},                 // mandatory query placeholder
                                |                    "type": "most_fields",
                                |                    "fields": ["text", "title"]}}],
                                |                "filter": [                                 // optional custom filters
                                |                    {"terms": {"year": ${years}}},
                                |                    {"terms": {"quarter": ${quarters}}},
                                |                    {"range": {"date": {"gte": ${date}}}}
                                |                    ],
                                |            }
                                |        },
                                |    }
                                 ```

                             **For this custom_query, a sample retrieve() could be:**
                             ```python
                            |    self.retrieve(query="Why did the revenue increase?",
                            |                  filters={"years": ["2019"], "quarters": ["Q1", "Q2"]})
                            ```
        :param top_k: How many documents to return per query.
        """
        self.document_store: ElasticsearchDocumentStore = document_store
        self.top_k = top_k
        self.custom_query = custom_query

    def retrieve(self, query: str, filters: dict = None, top_k: Optional[int] = None, index: str = None) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        """
        if top_k is None:
            top_k = self.top_k
        if index is None:
            index = self.document_store.index

        documents = self.document_store.query(query, filters, top_k, self.custom_query, index)
        return documents


class ElasticsearchFilterOnlyRetriever(ElasticsearchRetriever):
    """
    Naive "Retriever" that returns all documents that match the given filters. No impact of query at all.
    Helpful for benchmarking, testing and if you want to do QA on small documents without an "active" retriever.
    """

    def retrieve(self, query: str, filters: dict = None, top_k: Optional[int] = None, index: str = None) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.

        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        """
        if top_k is None:
            top_k = self.top_k
        if index is None:
            index = self.document_store.index
        documents = self.document_store.query(query=None, filters=filters, top_k=top_k,
                                              custom_query=self.custom_query, index=index)
        return documents

# TODO make Paragraph generic for configurable units of text eg, pages, paragraphs, or split by a char_limit
Paragraph = namedtuple("Paragraph", ["paragraph_id", "document_id", "text", "meta"])


class BM25Retriever(BaseRetriever):

    def __init__(self, doc_store: BaseDocumentStore, top_k: int = 10, b: int = 0.75, k1 = 1.6):
        self.document_store = doc_store
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 1),
            norm=None,
            smooth_idf=False
        )
        self.b = b
        self.k1 = k1
        self.fit()


    def _get_all_paragraphs(self) -> List[Paragraph]:
        """
        Split the list of documents in paragraphs
        """
        documents = self.document_store.get_all_documents(index='document')

        paragraphs = []
        p_id = 0
        for doc in documents:
            for p in doc.text.split("\n\n"):  # TODO: this assumes paragraphs are separated by "\n\n". Can be switched to paragraph tokenizer.
                if not p.strip():  # skip empty paragraphs
                    continue
                paragraphs.append(
                    Paragraph(document_id=doc.id, paragraph_id=p_id, text=(p,), meta=doc.meta)
                )
                p_id += 1
        logger.info(f"Found {len(paragraphs)} candidate paragraphs from {len(documents)} docs in DB")
        return paragraphs

    def _get_all_data(self) -> List[str]:
        documents = self.document_store.get_all_documents(index='document')
        return [d.text for d in documents]

    def fit(self):
        """ Fit IDF to documents X """
        X = self._get_all_data()
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def retrieve(self, query: str, filters: dict = None, top_k: Optional[int] = None, index: Optional[str] = None) -> List[Document]:
        docs = self.document_store.get_all_documents(index=index)
        scores = self.transform(query, [d.text for d in docs])
        idx_scores = [(idx, score) for idx, score in enumerate(scores)]
        indices_and_scores = OrderedDict(
            sorted(idx_scores, key=(lambda tup: tup[1]), reverse=True)
        )
        # TODO: Return specific ids from X
        docs[:] = [docs[i] for i in indices_and_scores.keys()]
        return docs[:top_k]

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be converted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1
