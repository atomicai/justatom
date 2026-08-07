import asyncio

from justatom.api import hypercorn_server


def test_build_hypercorn_config_uses_one_worker_and_explicit_bind():
    config = hypercorn_server.build_hypercorn_config("0.0.0.0", 5555)
    assert config.bind == ["0.0.0.0:5555"]
    assert config.workers == 1


def test_serve_app_delegates_to_hypercorn(monkeypatch):
    calls = []

    async def fake_serve(app, config):
        calls.append((app, config.bind, config.workers))

    monkeypatch.setattr(hypercorn_server, "serve", fake_serve)
    app = object()
    asyncio.run(hypercorn_server.serve_app(app, host="127.0.0.1", port=7777))
    assert calls == [(app, ["127.0.0.1:7777"], 1)]
