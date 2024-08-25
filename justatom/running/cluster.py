import numpy as np
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder

from justatom.etc.schema import Document
from justatom.running.mask import ICLUSTERINGRunner


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
        embeddings = list(self.model.encode(documents, batch_size=self.batch_size, verbose=verbose))
        embeddings = np.vstack(embeddings)
        return embeddings


class IBTRunner(ICLUSTERINGRunner):
    """BERTopic class"""

    def __init__(self, model: BaseEmbedder, **kwargs):
        super().__init__(model=model)
        if "n_gram_range" in kwargs:
            kwargs["n_gram_range"] = tuple(kwargs["n_gram_range"])
        self.topic_model = BERTopic(embedding_model=model, **kwargs)

    def fit_transform(self, docs: list[str | Document], **kwargs) -> tuple[list[int], np.ndarray | None]:
        _docs = [str(d) if isinstance(d, str) else d.content for d in docs]

        topics, probs = self.topic_model.fit_transform(documents=_docs, **kwargs)
        return topics, probs


__all__ = ["IBTRunner", "IHFWrapperBackend"]
