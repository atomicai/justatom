from __future__ import annotations

import asyncio
import os
from typing import Any

from quart import Quart, request

from justatom.api.hypercorn_server import serve_app


MODEL = os.getenv("FAKE_EMBEDDING_MODEL", "fixture-embedding-model")
PORT = int(os.getenv("FAKE_EMBEDDING_PORT", "18001"))
app = Quart(__name__, static_folder=None)
app.json.ensure_ascii = False
app.config.setdefault("PROVIDE_AUTOMATIC_OPTIONS", True)


def _vector(text: str) -> list[float]:
    normalized = text.lower()
    return [
        float("банк" in normalized or "негатив" in normalized),
        float("qwen" in normalized or "эмбед" in normalized),
        float("weaviate" in normalized or "вектор" in normalized),
    ]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL, "object": "model"}]}


@app.post("/v1/embeddings")
async def embeddings():
    payload: Any = await request.get_json(silent=True)
    if not isinstance(payload, dict) or payload.get("model") != MODEL:
        return {"error": {"message": "model not found", "type": "model_not_found"}}, 404

    source = payload.get("input")
    texts = [source] if isinstance(source, str) else source
    if not isinstance(texts, list) or not texts or any(not isinstance(text, str) for text in texts):
        return {"error": {"message": "invalid input", "type": "invalid_request_error"}}, 400

    return {
        "object": "list",
        "model": MODEL,
        "data": [
            {"object": "embedding", "index": index, "embedding": _vector(text)}
            for index, text in enumerate(texts)
        ],
    }


if __name__ == "__main__":
    asyncio.run(serve_app(app, host="0.0.0.0", port=PORT))
