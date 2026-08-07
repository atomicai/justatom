from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping

from justatom.api.hypercorn_server import serve_app
from justatom.api.run import create_app
from justatom.retrieval.errors import ConfigurationError


def _boolean(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean")


def build_retrieval_app(env: Mapping[str, str] | None = None):
    values = os.environ if env is None else env
    config_path = values.get("JUSTATOM_CONFIG", "/etc/justatom/serve.yaml")
    start_mq = _boolean(values.get("JUSTATOM_START_MQ", "false"), "JUSTATOM_START_MQ")
    return create_app(config_path=config_path, start_mq=start_mq)


def main() -> None:
    app = build_retrieval_app()
    asyncio.run(serve_app(app, host="0.0.0.0", port=5555))


if __name__ == "__main__":
    main()
