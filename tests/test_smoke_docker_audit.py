import os
import shlex
import stat
import subprocess
from pathlib import Path

import pytest

from tests.shell_utils import bash_executable

ROOT = Path(__file__).resolve().parents[1]
AUDIT_HELPER = ROOT / "scripts" / "smoke_docker_audit.sh"


def _write_fake_docker(bin_dir: Path) -> None:
    docker = bin_dir / "docker"
    docker.write_text(
        """#!/usr/bin/env bash
set -u

case "${1:-}:${2:-}" in
  ps:-a)
    query=ps-list
    output=container-project
    ;;
  ps:-aq)
    query=ps-project
    output=container-id
    ;;
  volume:ls)
    query=volume-list
    output=volume-id
    ;;
  volume:inspect)
    query=volume-inspect
    output=volume-project
    ;;
  network:ls)
    query=network-list
    output=network-id
    ;;
  network:inspect)
    query=network-inspect
    output=network-project
    ;;
  *)
    exit 99
    ;;
esac

printf '%s\\n' "$query" >>"$DOCKER_QUERY_LOG"
if [[ "${FAIL_QUERY:-}" == "$query" ]]; then
  exit 17
fi
printf '%s\\n' "$output"
""",
        encoding="utf-8",
    )
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)


def _run_helper(tmp_path: Path, helper: str, fail_query: str = "") -> tuple[int, bool, list[str], list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_docker(bin_dir)
    query_log = tmp_path / "queries.log"
    env = os.environ.copy()
    env.update(
        {
            "DOCKER_QUERY_LOG": str(query_log),
            "FAIL_QUERY": fail_query,
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
            "PROJECT": "audit-project",
        }
    )
    command = f"""
source {shlex.quote(AUDIT_HELPER.as_posix())}
if output="$({helper})"; then
  status=0
else
  status=$?
fi
printf 'status=%s\\n' "$status"
if ! {helper} >/dev/null; then
  printf 'failed_under_if_not=true\\n'
else
  printf 'failed_under_if_not=false\\n'
fi
printf '%s\\n' "$output"
"""
    result = subprocess.run(
        [bash_executable(), "-c", command],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    lines = result.stdout.splitlines()
    queries = query_log.read_text(encoding="utf-8").splitlines() if query_log.exists() else []
    failed_under_if_not = lines[1] == "failed_under_if_not=true"
    return int(lines[0].removeprefix("status=")), failed_under_if_not, lines[2:], queries


@pytest.mark.parametrize(
    ("helper", "fail_query"),
    [
        ("list_compose_projects", "ps-list"),
        ("list_compose_projects", "volume-list"),
        ("list_compose_projects", "volume-inspect"),
        ("list_compose_projects", "network-list"),
        ("list_compose_projects", "network-inspect"),
        ("list_preexisting_compose_projects", "ps-list"),
        ("list_preexisting_compose_projects", "volume-list"),
        ("list_preexisting_compose_projects", "volume-inspect"),
        ("list_preexisting_compose_projects", "network-list"),
        ("list_preexisting_compose_projects", "network-inspect"),
        ("project_resources", "ps-project"),
        ("project_resources", "volume-list"),
        ("project_resources", "network-list"),
    ],
)
def test_docker_audit_propagates_each_query_failure_under_if(tmp_path: Path, helper: str, fail_query: str) -> None:
    status, failed_under_if_not, _, queries = _run_helper(tmp_path, helper, fail_query)

    assert status == 17
    assert failed_under_if_not
    assert fail_query in queries


@pytest.mark.parametrize(
    ("helper", "expected"),
    [
        (
            "list_compose_projects",
            ["container-project", "network-project", "volume-project"],
        ),
        ("project_resources", ["container-id", "volume-id", "network-id"]),
    ],
)
def test_docker_audit_returns_complete_successful_results(tmp_path: Path, helper: str, expected: list[str]) -> None:
    status, failed_under_if_not, output, _ = _run_helper(tmp_path, helper)

    assert status == 0
    assert not failed_under_if_not
    assert output == expected


def test_container_smokes_share_the_hardened_docker_audit() -> None:
    for script_name in (
        "smoke_containerized_retrieval.sh",
        "smoke_api_external_backend.sh",
    ):
        script = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert "source scripts/smoke_docker_audit.sh" in script
        assert "list_compose_projects()" not in script
        assert "project_resources()" not in script
