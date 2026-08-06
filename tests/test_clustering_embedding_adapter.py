import numpy as np

from justatom.running.clusters import EmbeddingBackendAdapter


class FakeEmbedder:
    async def embed_documents(self, texts):
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


def test_bertopic_adapter_uses_document_embedding_role():
    result = EmbeddingBackendAdapter(FakeEmbedder()).embed(["a", "b"])
    np.testing.assert_array_equal(result, np.asarray([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32))
