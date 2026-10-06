import pytest

from justatom.api import serve as module
from justatom.retrieval.errors import ConfigurationError


def test_build_retrieval_app_uses_scenario_defaults_without_explicit_config(monkeypatch):
    calls = []

    def fake_create_app(**kwargs):
        calls.append(kwargs)
        return "app"

    monkeypatch.setattr(module, "create_app", fake_create_app)
    assert module.build_retrieval_app({}) == "app"
    assert calls == [{"config_path": None, "start_mq": False}]


def test_build_retrieval_app_keeps_explicit_container_config(monkeypatch):
    calls = []
    monkeypatch.setattr(module, "create_app", lambda **kwargs: calls.append(kwargs) or "app")
    module.build_retrieval_app({"JUSTATOM_CONFIG": "/etc/justatom/serve.yaml"})
    assert calls == [{"config_path": "/etc/justatom/serve.yaml", "start_mq": False}]


def test_build_retrieval_app_allows_explicit_mq_boolean(monkeypatch):
    calls = []
    monkeypatch.setattr(module, "create_app", lambda **kwargs: calls.append(kwargs) or "app")
    module.build_retrieval_app({"JUSTATOM_CONFIG": "/cfg/serve.yaml", "JUSTATOM_START_MQ": "true"})
    assert calls == [{"config_path": "/cfg/serve.yaml", "start_mq": True}]


def test_retrieval_entrypoint_rejects_cpu_and_cuda_profiles_together(monkeypatch):
    def unexpected_create_app(**kwargs):
        pytest.fail(f"create_app called before profile validation: {kwargs}")

    monkeypatch.setattr(module, "create_app", unexpected_create_app)

    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        module.build_retrieval_app({"JUSTATOM_EMBEDDING_PROFILES": "cpu,cuda"})


@pytest.mark.parametrize("options, bind", [({}, "0.0.0.0:5555"), ({"host": "127.0.0.1", "port": 7777}, "127.0.0.1:7777")])
def test_retrieval_server_starts_hypercorn(monkeypatch, options, bind):
    calls = []
    monkeypatch.setattr(module, "build_retrieval_app", lambda: "app")

    async def fake_serve(app, config):
        calls.append((app, config.bind, config.workers, config.accesslog, config.errorlog))

    monkeypatch.setattr(module, "serve", fake_serve)
    module.main(**options)
    assert calls == [("app", [bind], 1, "-", "-")]
