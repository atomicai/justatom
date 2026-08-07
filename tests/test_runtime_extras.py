import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest


def _extras():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def _contains_requirement(declarations, project_name):
    normalized_project_name = canonicalize_name(project_name)
    return any(canonicalize_name(Requirement(declaration).name) == normalized_project_name for declaration in declarations)


def test_serve_extra_has_http_storage_and_data_dependencies_without_torch():
    serve = "\n".join(_extras()["serve"]).lower()
    assert "quart" in serve
    assert "hypercorn" in serve
    assert "weaviate-client" in serve
    assert "polars" in serve
    assert "torch" not in serve
    assert "transformers" not in serve


def test_embedder_extra_has_local_runner_dependencies_without_torch_wheel():
    embedder = _extras()["embedder"]
    embedder_text = "\n".join(embedder).lower()
    assert "transformers" in embedder_text
    assert "pytorch-lightning" in embedder_text
    assert not _contains_requirement(embedder, "torch")


@pytest.mark.parametrize("declaration", ["Torch>=2.8", "torch == 2.8", "torch[cpu]>=2.8", "TORCH [cpu] >= 2.8"])
def test_embedder_extra_rejects_direct_torch_requirement_variants(monkeypatch, declaration):
    monkeypatch.setattr(
        sys.modules[__name__],
        "_extras",
        lambda: {"embedder": ["transformers>=4.44,<6", "pytorch-lightning==2.2.1", declaration]},
    )

    with pytest.raises(AssertionError):
        test_embedder_extra_has_local_runner_dependencies_without_torch_wheel()
