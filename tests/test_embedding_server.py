import asyncio

import pytest

from justatom.api import embedding_server as module
from justatom.retrieval.errors import ConfigurationError


class FakeEmbedder:
    def __init__(self, error=None):
        self.calls = []
        self.closed = 0
        self.error = error

    async def embed_documents(self, texts):
        self.calls.append(list(texts))
        if self.error is not None:
            raise self.error
        return [[float(index), float(len(text))] for index, text in enumerate(texts)]

    async def embed_queries(self, texts):
        return await self.embed_documents(texts)

    async def close(self):
        self.closed += 1


def test_settings_use_qwen_defaults():
    settings = module.EmbeddingServerSettings.from_env({})
    assert settings.model == "Qwen/Qwen3-Embedding-0.6B"
    assert settings.device == "cpu"
    assert settings.batch_size == 8
    assert settings.max_length == 512


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"EMBEDDING_MODEL": " "}, "EMBEDDING_MODEL"),
        ({"EMBEDDING_BATCH_SIZE": "0"}, "EMBEDDING_BATCH_SIZE"),
        ({"EMBEDDING_BATCH_SIZE": "true"}, "EMBEDDING_BATCH_SIZE"),
        ({"EMBEDDING_MAX_LENGTH": "-1"}, "EMBEDDING_MAX_LENGTH"),
    ],
)
def test_settings_reject_invalid_environment(env, message):
    with pytest.raises(ConfigurationError, match=message):
        module.EmbeddingServerSettings.from_env(env)


def test_build_local_embedder_uses_one_empty_prefix_profile(monkeypatch):
    calls = []

    class FakeEmbedder:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(module, "HuggingFaceEmbedder", FakeEmbedder)
    settings = module.EmbeddingServerSettings.from_env(
        {
            "EMBEDDING_MODEL": "model",
            "EMBEDDING_DEVICE": "mps",
            "EMBEDDING_BATCH_SIZE": "4",
            "EMBEDDING_MAX_LENGTH": "256",
        }
    )
    embedder = module.build_local_embedder(settings)
    assert isinstance(embedder, FakeEmbedder)
    assert calls[0]["model"] == "model"
    assert calls[0]["device"] == "mps"
    assert calls[0]["profile"].query_prefix == ""
    assert calls[0]["profile"].document_prefix == ""
    assert calls[0]["profile"].batch_size == 4
    assert calls[0]["profile"].max_length == 256


def test_embedding_endpoint_returns_ordered_openai_response_and_utf8():
    async def scenario():
        embedder = FakeEmbedder()
        settings = module.EmbeddingServerSettings("модель", "cpu", 8, 512)
        app = module.create_embedding_app(settings=settings, embedder=embedder)
        async with app.test_app() as test_app:
            response = await test_app.test_client().post(
                "/v1/embeddings",
                json={"model": "модель", "input": ["первый", "second"], "encoding_format": "float"},
            )
            body = await response.get_data()
            payload = await response.get_json()
        assert response.status_code == 200
        assert "модель".encode() in body
        assert b"\\u043c" not in body
        assert [item["index"] for item in payload["data"]] == [0, 1]
        assert [item["embedding"] for item in payload["data"]] == [[0.0, 6.0], [1.0, 6.0]]
        assert embedder.calls == [["первый", "second"]]
        assert embedder.closed == 1

    asyncio.run(scenario())


def test_models_and_health_report_configured_model():
    async def scenario():
        settings = module.EmbeddingServerSettings("model", "cpu", 8, 512)
        app = module.create_embedding_app(settings=settings, embedder=FakeEmbedder())
        async with app.test_app() as test_app:
            client = test_app.test_client()
            health = await client.get("/health")
            models = await client.get("/v1/models")
            assert await health.get_json() == {"status": "ok", "model": "model"}
            assert (await models.get_json())["data"][0]["id"] == "model"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"model": "other", "input": ["x"]},
        {"model": "model", "input": []},
        {"model": "model", "input": ["x", "y", "z"]},
        {"model": "model", "input": ["x", 7]},
        {"model": "model", "input": ["x"], "encoding_format": "base64"},
        {"model": "model", "input": ["x"], "dimensions": 128},
    ],
)
def test_embedding_endpoint_rejects_invalid_requests(payload):
    async def scenario():
        settings = module.EmbeddingServerSettings("model", "cpu", 2, 512)
        app = module.create_embedding_app(settings=settings, embedder=FakeEmbedder())
        async with app.test_app() as test_app:
            response = await test_app.test_client().post("/v1/embeddings", json=payload)
            assert 400 <= response.status_code < 500
            assert "error" in await response.get_json()

    asyncio.run(scenario())


def test_embedding_endpoint_sanitizes_backend_failure():
    async def scenario():
        secret = "secret-backend-detail"
        settings = module.EmbeddingServerSettings("model", "cpu", 8, 512)
        app = module.create_embedding_app(settings=settings, embedder=FakeEmbedder(RuntimeError(secret)))
        app.config["PROPAGATE_EXCEPTIONS"] = False
        async with app.test_app() as test_app:
            response = await test_app.test_client().post(
                "/v1/embeddings", json={"model": "model", "input": ["русский"]}
            )
            body = await response.get_data()
            assert response.status_code == 500
            assert secret.encode() not in body
            assert "embedding backend failed".encode() in body
            assert "русский".encode() not in body

    asyncio.run(scenario())


class FixedVectorsEmbedder(FakeEmbedder):
    def __init__(self, vectors):
        super().__init__()
        self.vectors = vectors

    async def embed_documents(self, texts):
        self.calls.append(list(texts))
        return self.vectors


@pytest.mark.parametrize(
    ("texts", "vectors"),
    [
        (["x"], []),
        (["x", "y"], [[1.0], [1.0, 2.0]]),
        (["x"], [[float("nan")]]),
    ],
)
def test_embedding_endpoint_rejects_invalid_backend_vectors(texts, vectors):
    async def scenario():
        settings = module.EmbeddingServerSettings("model", "cpu", 8, 512)
        app = module.create_embedding_app(settings=settings, embedder=FixedVectorsEmbedder(vectors))
        async with app.test_app() as test_app:
            response = await test_app.test_client().post("/v1/embeddings", json={"model": "model", "input": texts})
            assert response.status_code == 500
            assert await response.get_json() == {
                "error": {"message": "embedding backend failed", "type": "server_error"}
            }

    asyncio.run(scenario())
