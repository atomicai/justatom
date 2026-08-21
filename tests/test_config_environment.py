import os

from justatom.configuring.environment import load_runtime_environment


def test_runtime_environment_maps_huggingface_api_key_without_overwriting_standard_token(monkeypatch):
    monkeypatch.setattr("justatom.configuring.environment.dotenv.load_dotenv", lambda: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "  alias-token  ")

    load_runtime_environment()

    assert os.environ["HF_TOKEN"] == "alias-token"

    monkeypatch.setenv("HF_TOKEN", "standard-token")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "different-token")
    load_runtime_environment()
    assert os.environ["HF_TOKEN"] == "standard-token"
