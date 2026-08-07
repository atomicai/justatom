import pytest

from justatom.api import embedding_server as module
from justatom.retrieval.errors import ConfigurationError


def test_settings_use_qwen_defaults():
    settings = module.EmbeddingServerSettings.from_env({})
    assert settings.model == "Qwen/Qwen3-Embedding-0.6B"
    assert settings.device == "cpu"
    assert settings.batch_size == 8
    assert settings.max_length == 512


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"EMBEDDING_MODEL": " "}, "EMBEDDING_MODEL"),
        ({"EMBEDDING_BATCH_SIZE": "0"}, "EMBEDDING_BATCH_SIZE"),
        ({"EMBEDDING_BATCH_SIZE": "true"}, "EMBEDDING_BATCH_SIZE"),
        ({"EMBEDDING_MAX_LENGTH": "-1"}, "EMBEDDING_MAX_LENGTH"),
    ],
)
def test_settings_reject_invalid_environment(env, message):
    with pytest.raises(ConfigurationError, match=message):
        module.EmbeddingServerSettings.from_env(env)


def test_build_local_embedder_uses_one_empty_prefix_profile(monkeypatch):
    calls = []

    class FakeEmbedder:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(module, "HuggingFaceEmbedder", FakeEmbedder)
    settings = module.EmbeddingServerSettings.from_env(
        {
            "EMBEDDING_MODEL": "model",
            "EMBEDDING_DEVICE": "mps",
            "EMBEDDING_BATCH_SIZE": "4",
            "EMBEDDING_MAX_LENGTH": "256",
        }
    )
    embedder = module.build_local_embedder(settings)
    assert isinstance(embedder, FakeEmbedder)
    assert calls[0]["model"] == "model"
    assert calls[0]["device"] == "mps"
    assert calls[0]["profile"].query_prefix == ""
    assert calls[0]["profile"].document_prefix == ""
    assert calls[0]["profile"].batch_size == 4
    assert calls[0]["profile"].max_length == 256
