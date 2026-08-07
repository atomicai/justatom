from __future__ import annotations

from hypercorn.asyncio import serve
from hypercorn.config import Config


def build_hypercorn_config(host: str, port: int) -> Config:
    config = Config()
    config.bind = [f"{host}:{port}"]
    config.workers = 1
    config.accesslog = "-"
    config.errorlog = "-"
    return config


async def serve_app(app, *, host: str, port: int) -> None:
    await serve(app, build_hypercorn_config(host, port))
