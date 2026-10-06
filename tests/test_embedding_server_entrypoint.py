import pytest

from justatom.api import serve_embeddings as module


def test_build_embedding_app_passes_environment_settings(monkeypatch):
    calls = []
    monkeypatch.setattr(module, "create_embedding_app", lambda settings: calls.append(settings) or "app")
    app = module.build_embedding_app(
        {
            "EMBEDDING_MODEL": "model",
            "EMBEDDING_DEVICE": "cuda:0",
            "EMBEDDING_BATCH_SIZE": "4",
            "EMBEDDING_MAX_LENGTH": "256",
        }
    )
    assert app == "app"
    assert calls[0].model == "model"
    assert calls[0].device == "cuda:0"


@pytest.mark.parametrize("options, bind", [({}, "0.0.0.0:8000"), ({"host": "127.0.0.1", "port": 7777}, "127.0.0.1:7777")])
def test_embedding_server_starts_hypercorn(monkeypatch, options, bind):
    calls = []
    monkeypatch.setattr(module, "build_embedding_app", lambda: "app")

    async def fake_serve(app, config):
        calls.append((app, config.bind, config.workers, config.accesslog, config.errorlog))

    monkeypatch.setattr(module, "serve", fake_serve)
    module.main(**options)
    assert calls == [("app", [bind], 1, "-", "-")]
