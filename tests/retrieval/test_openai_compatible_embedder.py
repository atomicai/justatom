import asyncio
import json

import httpx
import pytest

from justatom.retrieval.contracts import EmbeddingProfile
from justatom.retrieval.embedders.openai_compatible import OpenAICompatibleEmbedder
from justatom.retrieval.errors import EmbeddingBackendError, EmbeddingResponseError


def test_remote_embedder_applies_role_prefix_and_restores_index_order():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/v1/embeddings"
        body = json.loads(request.content)
        assert body["input"] == ["query: first", "query: second"]
        return httpx.Response(
            200,
            json={"data": [{"index": 1, "embedding": [2, 0]}, {"index": 0, "embedding": [1, 0]}]},
        )

    embedder = OpenAICompatibleEmbedder(
        base_url="http://embedding.test/v1",
        model="test-model",
        api_key=None,
        timeout=12.5,
        profile=EmbeddingProfile(query_prefix="query: "),
        transport=httpx.MockTransport(handler),
    )
    vectors = asyncio.run(embedder.embed_queries(["first", "second"]))
    asyncio.run(embedder.close())

    assert vectors == [[1.0, 0.0], [2.0, 0.0]]
    assert embedder.timeout == 12.5
    assert "authorization" not in requests[0].headers


def test_remote_embedder_splits_requests_by_profile_batch_size():
    batches = []

    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        batches.append(inputs)
        return httpx.Response(
            200, json={"data": [{"index": index, "embedding": [float(len(text))]} for index, text in enumerate(inputs)]}
        )

    embedder = OpenAICompatibleEmbedder(
        base_url="http://embedding.test/v1",
        model="test-model",
        profile=EmbeddingProfile(batch_size=2),
        transport=httpx.MockTransport(handler),
    )
    assert len(asyncio.run(embedder.embed_documents(["a", "bb", "ccc"]))) == 3
    assert batches == [["a", "bb"], ["ccc"]]
    asyncio.run(embedder.close())


def test_remote_embedder_owns_a_deep_copy_of_extra_body():
    extra_body = {"options": {"pooling": "mean"}}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["options"] == {"pooling": "mean"}
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    embedder = OpenAICompatibleEmbedder(
        base_url="http://embedding.test/v1",
        model="test-model",
        extra_body=extra_body,
        transport=httpx.MockTransport(handler),
    )
    extra_body["options"]["pooling"] = "max"

    assert asyncio.run(embedder.embed_queries(["query"])) == [[1.0]]
    asyncio.run(embedder.close())


def test_remote_embedder_sanitizes_http_failures():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="provider failed: secret document body")

    embedder = OpenAICompatibleEmbedder(
        base_url="http://embedding.test/v1",
        model="test-model",
        api_key="top-secret",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(EmbeddingBackendError) as exc_info:
        asyncio.run(embedder.embed_documents(["private input text"]))
    message = str(exc_info.value)
    assert "503" in message and "test-model" in message
    assert "top-secret" not in message and "private input text" not in message
    asyncio.run(embedder.close())


def test_remote_embedder_rejects_missing_or_duplicate_response_indexes():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1]}, {"index": 0, "embedding": [2]}]})

    embedder = OpenAICompatibleEmbedder(
        base_url="http://embedding.test/v1",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(EmbeddingResponseError, match="indexes"):
        asyncio.run(embedder.embed_queries(["a", "b"]))
    asyncio.run(embedder.close())


def test_remote_embedder_sanitizes_malformed_json_responses():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{not valid json")

    embedder = OpenAICompatibleEmbedder(
        base_url="http://embedding.test/v1",
        model="test-model",
        api_key="top-secret",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(EmbeddingResponseError, match="valid JSON") as exc_info:
        asyncio.run(embedder.embed_documents(["private input text"]))
    assert "top-secret" not in str(exc_info.value)
    assert "private input text" not in str(exc_info.value)
    asyncio.run(embedder.close())


def test_remote_embedder_rejects_string_embeddings():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": "not a vector"}]})

    embedder = OpenAICompatibleEmbedder(
        base_url="http://embedding.test/v1",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(EmbeddingResponseError, match="embeddings"):
        asyncio.run(embedder.embed_documents(["text"]))
    asyncio.run(embedder.close())


def test_remote_embedder_rejects_non_numeric_embedding_values():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": ["not a number"]}]})

    embedder = OpenAICompatibleEmbedder(
        base_url="http://embedding.test/v1",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(EmbeddingResponseError, match="embeddings"):
        asyncio.run(embedder.embed_documents(["text"]))
    asyncio.run(embedder.close())


def test_remote_embedder_rejects_oversized_integer_embedding_values():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [10**400]}]})

    embedder = OpenAICompatibleEmbedder(
        base_url="http://embedding.test/v1",
        model="test-model",
        api_key="top-secret",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(EmbeddingResponseError, match="invalid embeddings") as exc_info:
        asyncio.run(embedder.embed_documents(["private input text"]))
    assert "top-secret" not in str(exc_info.value)
    assert "private input text" not in str(exc_info.value)
    asyncio.run(embedder.close())
