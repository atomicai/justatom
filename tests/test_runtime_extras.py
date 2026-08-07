import tomllib
from pathlib import Path


def _extras():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def test_serve_extra_has_http_storage_and_data_dependencies_without_torch():
    serve = "\n".join(_extras()["serve"]).lower()
    assert "quart" in serve
    assert "hypercorn" in serve
    assert "weaviate-client" in serve
    assert "polars" in serve
    assert "torch" not in serve
    assert "transformers" not in serve


def test_embedder_extra_has_local_runner_dependencies_without_torch_wheel():
    embedder = "\n".join(_extras()["embedder"]).lower()
    assert "transformers" in embedder
    assert "pytorch-lightning" in embedder
    assert not any(line.startswith("torch==") for line in embedder.splitlines())
