from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx
import pytest

from justatom.api.embedding_server import EmbeddingServerSettings, create_embedding_app
from justatom.retrieval.contracts import EmbeddingProfile
from justatom.retrieval.embedders.openai_compatible import OpenAICompatibleEmbedder


class DeterministicEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.closed = 0

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        batch = list(texts)
        self.calls.append(batch)
        return [[float(len(text)), 1.0] for text in batch]

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.embed_documents(texts)

    async def close(self) -> None:
        self.closed += 1


@pytest.mark.integration
def test_builtin_server_satisfies_openai_compatible_embedder_contract():
    async def scenario() -> None:
        backend = DeterministicEmbedder()
        settings = EmbeddingServerSettings("model", "cpu", 8, 512)
        app = create_embedding_app(settings=settings, embedder=backend)
        transport = httpx.ASGITransport(app=app)

        async with app.test_app():
            client = OpenAICompatibleEmbedder(
                base_url="http://embedder.test/v1",
                model="model",
                profile=EmbeddingProfile(
                    document_prefix="doc: ",
                    query_prefix="query: ",
                    batch_size=2,
                ),
                transport=transport,
            )
            try:
                documents = await client.embed_documents(["a", "bbb", "ccccc"])
                queries = await client.embed_queries(["zz"])
            finally:
                await client.close()

        assert documents == [[6.0, 1.0], [8.0, 1.0], [10.0, 1.0]]
        assert queries == [[9.0, 1.0]]
        assert backend.calls == [
            ["doc: a", "doc: bbb"],
            ["doc: ccccc"],
            ["query: zz"],
        ]
        assert backend.closed == 1

    asyncio.run(scenario())
