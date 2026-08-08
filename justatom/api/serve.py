from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence

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


def validate_embedding_profiles(profiles: Sequence[str]) -> None:
    selected = {profile.strip().lower() for profile in profiles if profile.strip()}
    if {"cpu", "cuda"}.issubset(selected):
        raise ConfigurationError("cpu and cuda embedding profiles are mutually exclusive")


def build_retrieval_app(env: Mapping[str, str] | None = None):
    values = os.environ if env is None else env
    profiles = values.get("JUSTATOM_EMBEDDING_PROFILES", "external").split(",")
    validate_embedding_profiles(profiles)
    config_path = values.get("JUSTATOM_CONFIG", "/etc/justatom/serve.yaml")
    start_mq = _boolean(values.get("JUSTATOM_START_MQ", "false"), "JUSTATOM_START_MQ")
    return create_app(config_path=config_path, start_mq=start_mq)


def main() -> None:
    app = build_retrieval_app()
    asyncio.run(serve_app(app, host="0.0.0.0", port=5555))


if __name__ == "__main__":
    main()
