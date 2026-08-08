import asyncio as asio
import threading
from collections.abc import Coroutine, Iterator
from typing import Any, TypeVar

import numpy as np
import torch
from tqdm.autonotebook import tqdm

from justatom.etc.schema import Document
from justatom.modeling.mask import ILanguageModel
from justatom.processing.loader import NamedDataLoader
from justatom.processing.mask import IProcessor
from justatom.processing.silo import igniset
from justatom.retrieval.contracts import Embedder
from justatom.running.mask import IClusteringRunner, ICLUSTERINGWrapperBackend, IDimReducer, IDocEmbedder

_T = TypeVar("_T")


def _load_bertopic():
    try:
        from bertopic import BERTopic
        from bertopic.backend import BaseEmbedder
    except ModuleNotFoundError as error:
        raise ImportError("BERTopic clustering requires `pip install 'justatom[clustering]'`") from error
    return BERTopic, BaseEmbedder


def _load_umap():
    try:
        from umap import UMAP
    except ModuleNotFoundError as error:
        raise ImportError("UMAP dimension reduction requires `pip install 'justatom[clustering]'`") from error
    return UMAP


class DocEmbedder(IDocEmbedder):
    """General class for embedding any NLP textual document"""

    def __init__(
        self,
        model: ILanguageModel,
        processor: IProcessor,
        device: str = "cpu",
    ):
        self.processor = processor
        self.model = model
        self.device = device

    @torch.no_grad()
    def encode(
        self,
        texts: list[dict],
        batch_size: int = 1,
        padding: bool = True,
        truncation: bool = True,
        normalize_embeddings: bool = True,
        verbose: bool = False,
        streaming_preprocessing: bool = False,
        **kwargs,
    ) -> Iterator[np.ndarray]:

        model = self.model.to(self.device).eval()

        dataset, tensor_names = igniset(
            texts,
            processor=self.processor,
            batch_size=batch_size,
            streaming=streaming_preprocessing,
        )

        loader = NamedDataLoader(dataset=dataset, batch_size=batch_size, tensor_names=tensor_names)

        batch_gen = range(0, len(texts), batch_size)
        if verbose:
            batch_gen = tqdm(batch_gen)

        for batch_begin, batch_features in zip(batch_gen, loader, strict=False):  # noqa: B007
            batch = {k: v.to(self.device) for k, v in batch_features.items()}

            embeddings = model(**batch)[0].cpu()

            yield embeddings.numpy()


class IHFWrapperBackend(ICLUSTERINGWrapperBackend):
    def __init__(self, model, batch_size: int = 16):
        super().__init__(model=model)
        self.batch_size = batch_size

    def embed(self, documents: list[str], verbose: bool = False) -> np.ndarray:
        """Embed a list of n documents/words into an n-dimensional
        matrix of embeddings

        Arguments:
            documents: A list of documents or words to be embedded
            verbose: Controls the verbosity of the process

        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        embeddings = list(self.model.encode(documents, batch_size=self.batch_size, verbose=verbose))
        embeddings = np.vstack(embeddings)
        return embeddings


class EmbeddingBackendAdapter(ICLUSTERINGWrapperBackend):
    def __init__(self, embedder: Embedder):
        super().__init__(model=embedder)
        self.embedder = embedder
        self._condition = threading.Condition()
        self._ready = threading.Event()
        self._loop: asio.AbstractEventLoop | None = None
        self._startup_error: BaseException | None = None
        self._active_calls = 0
        self._embedder_closing = False
        self._embedder_closed = False
        self._closing = False
        self._closed = False
        self._thread = threading.Thread(
            target=self._run_owner_loop,
            name="justatom-embedding-adapter",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError("Could not start embedding adapter event loop") from self._startup_error

    def _run_owner_loop(self) -> None:
        try:
            loop = asio.new_event_loop()
        except BaseException as error:  # pragma: no cover - platform startup failure
            self._startup_error = error
            self._ready.set()
            return

        asio.set_event_loop(loop)
        with self._condition:
            self._loop = loop
        self._ready.set()

        try:
            loop.run_forever()
        finally:
            try:
                pending = asio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
            finally:
                asio.set_event_loop(None)
                loop.close()
                with self._condition:
                    self._loop = None
                    self._closed = True
                    self._condition.notify_all()

    def _run_async(self, coro: Coroutine[Any, Any, _T]) -> _T:
        with self._condition:
            if self._closing or self._closed:
                coro.close()
                raise RuntimeError("Embedding backend adapter is closed")
            if self._embedder_closing or self._embedder_closed:
                coro.close()
                raise RuntimeError("Supplied embedder is closed")
            loop = self._loop
            if loop is None:  # pragma: no cover - guarded by startup synchronization
                coro.close()
                raise RuntimeError("Embedding backend adapter event loop is unavailable")
            self._active_calls += 1

        try:
            return self._submit(coro, loop)
        finally:
            with self._condition:
                self._active_calls -= 1
                if self._active_calls == 0:
                    self._condition.notify_all()

    @staticmethod
    def _submit(coro: Coroutine[Any, Any, _T], loop: asio.AbstractEventLoop) -> _T:
        try:
            future = asio.run_coroutine_threadsafe(coro, loop)
        except BaseException:
            coro.close()
            raise
        return future.result()

    def close_embedder(self) -> None:
        """Close the supplied embedder on its adapter loop before calling close().

        This operation is caller-controlled because the adapter does not own the
        embedder. Normal adapter close never invokes it implicitly.
        """
        if threading.current_thread() is self._thread:
            raise RuntimeError("Supplied embedder cannot close from the adapter owner thread")

        with self._condition:
            self._condition.wait_for(lambda: not self._embedder_closing)
            if self._embedder_closed:
                return
            if self._closing or self._closed:
                raise RuntimeError("Embedding backend adapter is closed")

            self._embedder_closing = True
            self._condition.wait_for(lambda: self._active_calls == 0)
            loop = self._loop
            if loop is None:  # pragma: no cover - guarded by adapter close state
                self._embedder_closing = False
                self._condition.notify_all()
                raise RuntimeError("Embedding backend adapter event loop is unavailable")
            self._active_calls += 1

        succeeded = False
        try:
            self._submit(self.embedder.close(), loop)
            succeeded = True
        finally:
            with self._condition:
                self._active_calls -= 1
                self._embedder_closing = False
                self._embedder_closed = succeeded
                self._condition.notify_all()

    def close(self) -> None:
        if threading.current_thread() is self._thread:
            raise RuntimeError("Embedding backend adapter cannot close from its owner thread")

        with self._condition:
            if self._closed:
                return
            if self._closing:
                self._condition.wait_for(lambda: self._closed)
                return
            self._closing = True
            self._condition.wait_for(lambda: self._active_calls == 0 and not self._embedder_closing)
            loop = self._loop

        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        self._thread.join()

    def embed(self, documents: list[str], verbose: bool = False) -> np.ndarray:
        del verbose
        vectors = self._run_async(self.embedder.embed_documents(documents))
        if len(vectors) == 0:
            return np.empty((0, 0), dtype=np.float32)
        return np.asarray(vectors, dtype=np.float32)


class IBTRunner(IClusteringRunner):
    """BERTopic class"""

    def __init__(self, model: ICLUSTERINGWrapperBackend, **kwargs):
        super().__init__(model=model)
        if "n_gram_range" in kwargs:
            kwargs["n_gram_range"] = tuple(kwargs["n_gram_range"])
        bertopic_type, base_embedder = _load_bertopic()
        if not isinstance(model, base_embedder):
            raise TypeError("BERTopic embedding model must implement its BaseEmbedder contract")
        self.topic_model = bertopic_type(embedding_model=model, **kwargs)

    def fit_transform(self, docs: list[str | Document], **kwargs) -> tuple[list[int], np.ndarray | None]:
        _docs = [str(d) if isinstance(d, str) else d.content for d in docs]

        topics, probs = self.topic_model.fit_transform(documents=_docs, **kwargs)
        return topics, probs


class IUMAPDimReducer(IDimReducer):
    def __init__(self, **kwargs):
        self.umap = _load_umap()(**kwargs)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        embs = self.umap.fit_transform(embeddings)
        return embs  # type: ignore

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        embs = self.umap.transform(embeddings)
        return embs  # type: ignore


__all__ = ["IBTRunner", "IHFWrapperBackend", "EmbeddingBackendAdapter"]
