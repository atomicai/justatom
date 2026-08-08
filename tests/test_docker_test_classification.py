import os
import stat
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_standard_docker_asset_suite_does_not_invoke_docker(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        """#!/bin/sh
printf 'called\\n' >>"$DOCKER_CALLED"
exit 97
"""
    )
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)
    called = tmp_path / "docker-called.log"
    env = os.environ.copy()
    env["DOCKER_CALLED"] = str(called)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_docker_assets.py",
            "-m",
            "not integration",
            "-q",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not called.exists()


def test_ci_has_dedicated_docker_compose_contract_gate() -> None:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yaml").read_text())
    job = workflow["jobs"]["pytest-integration-docker-compose"]
    setup_python = next(step for step in job["steps"] if step.get("uses") == "actions/setup-python@v5")
    test_step = next(step for step in job["steps"] if step.get("name") == "Run Docker Compose contract tests")

    assert setup_python["with"]["python-version"] == "3.12"
    assert test_step["run"] == "python -m pytest tests/test_docker_assets.py -m integration"
