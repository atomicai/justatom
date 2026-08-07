from justatom.api import serve as module


def test_build_retrieval_app_uses_container_defaults(monkeypatch):
    calls = []

    def fake_create_app(**kwargs):
        calls.append(kwargs)
        return "app"

    monkeypatch.setattr(module, "create_app", fake_create_app)
    assert module.build_retrieval_app({}) == "app"
    assert calls == [{"config_path": "/etc/justatom/serve.yaml", "start_mq": False}]


def test_build_retrieval_app_allows_explicit_mq_boolean(monkeypatch):
    calls = []
    monkeypatch.setattr(module, "create_app", lambda **kwargs: calls.append(kwargs) or "app")
    module.build_retrieval_app({"JUSTATOM_CONFIG": "/cfg/serve.yaml", "JUSTATOM_START_MQ": "true"})
    assert calls == [{"config_path": "/cfg/serve.yaml", "start_mq": True}]
