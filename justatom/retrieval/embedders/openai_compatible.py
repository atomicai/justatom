from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from justatom.retrieval.contracts import EmbeddingProfile, apply_prefix, validate_embeddings
from justatom.retrieval.errors import EmbeddingBackendError, EmbeddingResponseError


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        profile: EmbeddingProfile | None = None,
        encoding_format: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = f"{base_url.rstrip('/')}/"
        self.model = model
        self.profile = profile or EmbeddingProfile()
        self.encoding_format = encoding_format
        self.extra_body = dict(extra_body or {})
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers,
            transport=transport,
        )
        self._closed = False

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, prefix=self.profile.query_prefix)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, prefix=self.profile.document_prefix)

    async def close(self) -> None:
        if not self._closed:
            await self._client.aclose()
            self._closed = True

    async def _embed(self, texts: Sequence[str], *, prefix: str) -> list[list[float]]:
        if not texts:
            return []
        normalized = [
            apply_prefix(text, prefix, skip_if_present=self.profile.skip_prefix_if_present) for text in texts
        ]
        vectors: list[list[float]] = []
        for start in range(0, len(normalized), self.profile.batch_size):
            chunk = normalized[start : start + self.profile.batch_size]
            payload = {**self.extra_body, "model": self.model, "input": chunk}
            if self.encoding_format is not None:
                payload["encoding_format"] = self.encoding_format
            try:
                response = await self._client.post("embeddings", json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise EmbeddingBackendError(
                    f"Embedding endpoint {self.base_url!r} failed for model {self.model!r} "
                    f"with HTTP {exc.response.status_code}"
                ) from exc
            except httpx.HTTPError as exc:
                raise EmbeddingBackendError(
                    f"Embedding endpoint {self.base_url!r} failed for model {self.model!r}: {type(exc).__name__}"
                ) from exc
            try:
                payload = response.json()
            except ValueError as exc:
                raise EmbeddingResponseError("Embedding response was not valid JSON") from exc
            vectors.extend(self._parse_response(payload, expected_count=len(chunk)))
        try:
            return validate_embeddings(vectors, expected_count=len(texts))
        except (TypeError, ValueError, OverflowError) as exc:
            raise EmbeddingResponseError("Embedding response contains invalid embeddings") from exc

    @staticmethod
    def _parse_response(payload: Any, *, expected_count: int) -> list[list[float]]:
        try:
            data = payload["data"]
        except (KeyError, TypeError) as exc:
            raise EmbeddingResponseError("Embedding response must contain data") from exc
        if not isinstance(data, list):
            raise EmbeddingResponseError("Embedding response data must be a list")

        vectors_by_index: dict[int, list[float]] = {}
        for item in data:
            if not isinstance(item, Mapping) or type(item.get("index")) is not int or "embedding" not in item:
                raise EmbeddingResponseError("Embedding response must contain integer indexes and embeddings")
            index = item["index"]
            if index in vectors_by_index:
                raise EmbeddingResponseError("Embedding response indexes must be unique and complete")
            embedding = item["embedding"]
            if not isinstance(embedding, Sequence) or isinstance(embedding, (str, bytes, bytearray, Mapping)):
                raise EmbeddingResponseError("Embedding response embeddings must be non-string sequences")
            try:
                vectors_by_index[index] = list(embedding)
            except (TypeError, ValueError) as exc:
                raise EmbeddingResponseError("Embedding response embeddings must be sequences") from exc

        expected_indexes = set(range(expected_count))
        if set(vectors_by_index) != expected_indexes:
            raise EmbeddingResponseError("Embedding response indexes must match the request")
        return [vectors_by_index[index] for index in range(expected_count)]
