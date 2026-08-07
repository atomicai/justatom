from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx
import pytest

from justatom.api.embedding_server import EmbeddingServerSettings, create_embedding_app
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
                transport=transport,
            )
            try:
                first = await client.embed_documents(["one", "two"])
                second = await client.embed_queries(["three"])
            finally:
                await client.close()

        assert first == [[3.0, 1.0], [3.0, 1.0]]
        assert second == [[5.0, 1.0]]
        assert backend.calls == [["one", "two"], ["three"]]
        assert backend.closed == 1

    asyncio.run(scenario())
