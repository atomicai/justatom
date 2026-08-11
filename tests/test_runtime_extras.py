import sys
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by Python 3.10 CI
    import tomli as tomllib


def _project():
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]


def _extras():
    return _project()["optional-dependencies"]


def _contains_requirement(declarations, project_name):
    normalized_project_name = canonicalize_name(project_name)
    return any(canonicalize_name(Requirement(declaration).name) == normalized_project_name for declaration in declarations)


def _workflow_jobs():
    return yaml.safe_load(Path(".github/workflows/ci.yaml").read_text(encoding="utf-8"))["jobs"]


def _step(job, name):
    return next(step for step in job["steps"] if step.get("name") == name)


def _action_step(job, action):
    return next(step for step in job["steps"] if step.get("uses") == action)


def test_package_supports_python_310_through_313_only():
    assert _project()["requires-python"] == ">=3.10,<3.14"


def test_test_extra_supplies_tomli_only_for_python_310():
    tomli = next(Requirement(declaration) for declaration in _extras()["test"] if Requirement(declaration).name == "tomli")

    assert str(tomli.marker) == 'python_version < "3.11"'


def test_ci_covers_supported_python_range_and_representative_platforms():
    jobs = _workflow_jobs()

    assert jobs["pytest-ubuntu"]["strategy"]["matrix"]["python-version"] == ["3.10", "3.11", "3.12", "3.13"]
    assert _action_step(jobs["pytest-windows"], "actions/setup-python@v5")["with"]["python-version"] == "3.12"
    assert _action_step(jobs["pytest-macos"], "actions/setup-python@v5")["with"]["python-version"] == "3.12"


def test_windows_tests_run_from_git_bash():
    run_tests = _step(_workflow_jobs()["pytest-windows"], "Run tests")

    assert run_tests["shell"] == "bash"


@pytest.mark.parametrize(
    "job_name",
    ["pytest-ubuntu", "pytest-windows", "pytest-macos", "pytest-integration-weaviate"],
)
def test_python_test_jobs_install_serve_runtime_extra(job_name):
    install = _step(_workflow_jobs()[job_name], "Install requirements")["run"]

    assert 'python -m pip install ".[torch,serve,test]"' in install


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


def test_torch_extra_has_peft_and_qwen3_compatible_transformers():
    torch_extra = _extras()["torch"]
    transformers = next(Requirement(item) for item in torch_extra if Requirement(item).name == "transformers")

    assert _contains_requirement(torch_extra, "peft")
    assert "4.51" in str(transformers.specifier)


def test_clustering_extra_owns_bertopic_and_umap_dependencies():
    clustering = _extras()["clustering"]

    assert _contains_requirement(clustering, "bertopic")
    assert _contains_requirement(clustering, "umap-learn")


@pytest.mark.parametrize("declaration", ["Torch>=2.8", "torch == 2.8", "torch[cpu]>=2.8", "TORCH [cpu] >= 2.8"])
def test_embedder_extra_rejects_direct_torch_requirement_variants(monkeypatch, declaration):
    monkeypatch.setattr(
        sys.modules[__name__],
        "_extras",
        lambda: {"embedder": ["transformers>=4.44,<6", "pytorch-lightning==2.2.1", declaration]},
    )

    with pytest.raises(AssertionError):
        test_embedder_extra_has_local_runner_dependencies_without_torch_wheel()
