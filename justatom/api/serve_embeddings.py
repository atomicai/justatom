from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping

from quart import Quart

from justatom.api.embedding_server import EmbeddingServerSettings, create_embedding_app
from justatom.api.hypercorn_server import serve_app


def build_embedding_app(env: Mapping[str, str] | None = None) -> Quart:
    values = os.environ if env is None else env
    return create_embedding_app(EmbeddingServerSettings.from_env(values))


def main() -> None:
    asyncio.run(serve_app(build_embedding_app(), host="0.0.0.0", port=8000))


if __name__ == "__main__":
    main()
