from collections.abc import Iterator

import numpy as np
import torch
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from tqdm.autonotebook import tqdm
from umap import UMAP

from justatom.etc.schema import Document
from justatom.modeling.mask import ILanguageModel
from justatom.processing.loader import NamedDataLoader
from justatom.processing.mask import IProcessor
from justatom.processing.silo import igniset
from justatom.running.mask import IClusteringRunner, IDimReducer, IDocEmbedder


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
        **kwargs,
    ) -> Iterator[np.ndarray]:

        model = self.model.to(self.device).eval()

        dataset, tensor_names = igniset(
            texts,
            processor=self.processor,
            batch_size=batch_size,
        )

        loader = NamedDataLoader(
            dataset=dataset, batch_size=batch_size, tensor_names=tensor_names
        )

        batch_gen = range(0, len(texts), batch_size)
        if verbose:
            batch_gen = tqdm(batch_gen)

        for batch_begin, batch_features in zip(
            batch_gen, loader, strict=False
        ):  # noqa: B007
            batch = {k: v.to(self.device) for k, v in batch_features.items()}

            embeddings = model(**batch)[0].cpu()

            yield embeddings.numpy()


class IHFWrapperBackend(BaseEmbedder):
    def __init__(self, model, batch_size: int = 16):
        super().__init__()
        self.model = model
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
        embeddings = list(
            self.model.encode(documents, batch_size=self.batch_size, verbose=verbose)
        )
        embeddings = np.vstack(embeddings)
        return embeddings


class IBTRunner(IClusteringRunner):
    """BERTopic class"""

    def __init__(self, model: BaseEmbedder, **kwargs):
        super().__init__(model=model)
        if "n_gram_range" in kwargs:
            kwargs["n_gram_range"] = tuple(kwargs["n_gram_range"])
        self.topic_model = BERTopic(embedding_model=model, **kwargs)

    def fit_transform(
        self, docs: list[str | Document], **kwargs
    ) -> tuple[list[int], np.ndarray | None]:
        _docs = [str(d) if isinstance(d, str) else d.content for d in docs]

        topics, probs = self.topic_model.fit_transform(documents=_docs, **kwargs)
        return topics, probs


class IUMAPDimReducer(IDimReducer):
    def __init__(self, **kwargs):
        self.umap = UMAP(**kwargs)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        embs = self.umap.fit_transform(embeddings)
        return embs  # type: ignore

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        embs = self.umap.transform(embeddings)
        return embs  # type: ignore


__all__ = ["IBTRunner", "IHFWrapperBackend"]