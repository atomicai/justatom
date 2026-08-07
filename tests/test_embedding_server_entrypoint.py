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


def test_embedding_server_port_is_8000(monkeypatch):
    calls = []
    monkeypatch.setattr(module, "build_embedding_app", lambda: "app")

    async def fake_serve(app, *, host, port):
        calls.append((app, host, port))

    monkeypatch.setattr(module, "serve_app", fake_serve)
    module.main()
    assert calls == [("app", "0.0.0.0", 8000)]
