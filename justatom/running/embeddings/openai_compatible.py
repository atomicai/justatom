from __future__ import annotations

import asyncio as asio
import json
from urllib import request

from justatom.running.embeddings.base import IEmbeddingClient


class OpenAICompatibleEmbeddingClient(IEmbeddingClient):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        query_prefix: str = "",
        passage_prefix: str = "",
        default_input_type: str = "raw",
        prefix_enabled: bool = True,
        prefix_skip_if_present: bool = True,
        default_pooling: str | None = None,
        default_encoding_format: str | None = None,
        default_max_seq_len: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = float(timeout)
        self.query_prefix = query_prefix or ""
        self.passage_prefix = passage_prefix or ""
        self.default_input_type = (default_input_type or "raw").strip().lower()
        self.prefix_enabled = bool(prefix_enabled)
        self.prefix_skip_if_present = bool(prefix_skip_if_present)
        self.default_pooling = default_pooling
        self.default_encoding_format = default_encoding_format
        self.default_max_seq_len = (
            int(default_max_seq_len) if default_max_seq_len is not None and int(default_max_seq_len) > 0 else None
        )

    @staticmethod
    def _normalize_optional_int(value: object) -> int | None:
        if value is None:
            return None
        try:
            out = int(value)
        except (TypeError, ValueError):
            return None
        return out if out > 0 else None

    def _apply_prefix(self, text: str, input_type: str) -> str:
        input_type = (input_type or self.default_input_type or "raw").strip().lower()
        if input_type in {"query", "queries"}:
            prefix = self.query_prefix
        elif input_type in {"passage", "document", "doc", "content"}:
            prefix = self.passage_prefix
        else:
            prefix = ""

        if not prefix:
            return text
        if self.prefix_skip_if_present and text.startswith(prefix):
            return text
        return f"{prefix}{text}"

    def _post_embeddings(self, payload: dict) -> dict:
        endpoint = f"{self.base_url}/embeddings"
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
        **props,
    ) -> list[list[float]]:
        if len(texts) == 0:
            return []

        input_type = str(props.pop("input_type", self.default_input_type))
        apply_prefix = bool(props.pop("apply_prefix", self.prefix_enabled))
        normalized = [self._apply_prefix(t, input_type) for t in texts] if apply_prefix else list(texts)

        payload = {
            "model": model or self.model,
            "input": normalized,
        }
        if "pooling" in props:
            payload["pooling"] = props["pooling"]
        elif self.default_pooling:
            payload["pooling"] = self.default_pooling

        if "encoding_format" in props:
            payload["encoding_format"] = props["encoding_format"]
        elif self.default_encoding_format:
            payload["encoding_format"] = self.default_encoding_format

        payload_max_seq_len = self._normalize_optional_int(props.pop("max_seq_len", self.default_max_seq_len))
        if payload_max_seq_len is not None:
            payload["max_seq_len"] = payload_max_seq_len

        response = await asio.to_thread(self._post_embeddings, payload)
        data = response.get("data", [])
        data = sorted(data, key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in data]
