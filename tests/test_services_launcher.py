import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "services.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _launcher_env(tmp_path: Path, **overrides: str) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_log = tmp_path / "docker.log"
    _write_executable(
        bin_dir / "docker",
        """#!/usr/bin/env bash
printf 'COMPOSE_PROFILES=%s\n' "$COMPOSE_PROFILES" > "$DOCKER_LOG"
printf 'arg=%s\n' "$@" >> "$DOCKER_LOG"
""",
    )
    env = os.environ.copy()
    env.pop("EMBEDDING_BASE_URL", None)
    env["COMPOSE_PROFILES"] = "cpu,cuda"
    env["DOCKER_LOG"] = str(docker_log)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env.update(overrides)
    return env, docker_log


def _run_launcher(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


@pytest.mark.parametrize("mode", ["external", "cpu", "cuda"])
def test_launcher_forwards_config_with_exact_selected_profile(tmp_path, mode):
    env, docker_log = _launcher_env(tmp_path)
    if mode == "cuda":
        bin_dir = Path(env["PATH"].split(os.pathsep)[0])
        _write_executable(bin_dir / "uname", "#!/bin/sh\necho Darwin\n")
        _write_executable(bin_dir / "nvidia-smi", "#!/bin/sh\nexit 1\n")

    result = _run_launcher([mode, "config", "--quiet"], env)

    assert result.returncode == 0, result.stderr
    assert docker_log.read_text(encoding="utf-8").splitlines() == [
        f"COMPOSE_PROFILES={mode}",
        "arg=compose",
        "arg=config",
        "arg=--quiet",
    ]


@pytest.mark.parametrize(
    "mode",
    ["cpu,cuda", "external,cpu"],
)
def test_launcher_rejects_comma_separated_backend_modes_before_docker(tmp_path, mode):
    env, docker_log = _launcher_env(tmp_path)

    result = _run_launcher([mode, "config"], env)

    assert result.returncode != 0
    assert "exactly one embedding mode" in result.stderr
    assert not docker_log.exists()


@pytest.mark.parametrize("mode", ["external", "cuda"])
def test_launcher_rejects_global_option_in_command_position_before_docker(tmp_path, mode):
    env, docker_log = _launcher_env(tmp_path)

    result = _run_launcher([mode, "--ansi", "never", "up"], env)

    assert result.returncode != 0
    assert "unsupported compose command" in result.stderr
    assert not docker_log.exists()


@pytest.mark.parametrize(
    "command",
    ["create", "start", "restart", "scale", "run", "exec", "watch"],
)
def test_launcher_rejects_unsupported_runtime_commands_before_docker(tmp_path, command):
    env, docker_log = _launcher_env(tmp_path)

    result = _run_launcher(["cpu", command, "api"], env)

    assert result.returncode != 0
    assert "unsupported compose command" in result.stderr
    assert not docker_log.exists()


@pytest.mark.parametrize(
    "args",
    [
        ["cpu", "config", "--profile", "cuda"],
        ["cpu", "--profile=cuda", "config"],
    ],
)
def test_launcher_rejects_manual_profile_arguments_before_docker(tmp_path, args):
    env, docker_log = _launcher_env(tmp_path)

    result = _run_launcher(args, env)

    assert result.returncode != 0
    assert "--profile is not supported" in result.stderr
    assert not docker_log.exists()


@pytest.mark.parametrize(
    ("command", "command_args"),
    [
        ("up", ["-d"]),
        ("down", ["--remove-orphans"]),
        ("config", ["--quiet"]),
        ("build", ["--pull"]),
        ("ps", ["--all"]),
        ("logs", ["--tail", "10", "api"]),
    ],
)
def test_launcher_forwards_supported_commands(tmp_path, command, command_args):
    env, docker_log = _launcher_env(tmp_path)

    result = _run_launcher(["cpu", command, *command_args], env)

    assert result.returncode == 0, result.stderr
    assert docker_log.read_text(encoding="utf-8").splitlines() == [
        "COMPOSE_PROFILES=cpu",
        "arg=compose",
        f"arg={command}",
        *(f"arg={arg}" for arg in command_args),
    ]


def test_launcher_does_not_treat_post_command_mode_value_as_another_selection(tmp_path):
    env, docker_log = _launcher_env(tmp_path)

    result = _run_launcher(["cpu", "logs", "cuda"], env)

    assert result.returncode == 0, result.stderr
    assert "arg=cuda" in docker_log.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize("command", ["down", "config", "build", "ps", "logs"])
def test_external_non_runtime_commands_do_not_require_embedding_url(tmp_path, command):
    env, docker_log = _launcher_env(tmp_path)

    result = _run_launcher(["external", command], env)

    assert result.returncode == 0, result.stderr
    assert docker_log.exists()


@pytest.mark.parametrize("command", ["down", "config", "build", "ps", "logs"])
def test_cuda_non_runtime_commands_do_not_require_runtime_host(tmp_path, command):
    env, docker_log = _launcher_env(tmp_path)
    bin_dir = Path(env["PATH"].split(os.pathsep)[0])
    _write_executable(bin_dir / "uname", "#!/bin/sh\necho Darwin\n")
    _write_executable(bin_dir / "nvidia-smi", "#!/bin/sh\nexit 1\n")

    result = _run_launcher(["cuda", command], env)

    assert result.returncode == 0, result.stderr
    assert docker_log.exists()


def test_external_up_requires_embedding_base_url_before_docker(tmp_path):
    env, docker_log = _launcher_env(tmp_path)

    result = _run_launcher(["external", "up", "-d"], env)

    assert result.returncode != 0
    assert "EMBEDDING_BASE_URL is required" in result.stderr
    assert not docker_log.exists()


def test_external_up_forwards_configured_embedding_base_url(tmp_path):
    env, docker_log = _launcher_env(
        tmp_path,
        EMBEDDING_BASE_URL="http://host.docker.internal:8000/v1",
    )

    result = _run_launcher(["external", "up", "-d"], env)

    assert result.returncode == 0, result.stderr
    assert docker_log.exists()


def test_cuda_up_rejects_non_linux_host_before_docker(tmp_path):
    env, docker_log = _launcher_env(tmp_path)
    _write_executable(Path(env["PATH"].split(os.pathsep)[0]) / "uname", "#!/bin/sh\necho Darwin\n")

    result = _run_launcher(["cuda", "up", "-d"], env)

    assert result.returncode != 0
    assert "CUDA mode requires a Linux host" in result.stderr
    assert not docker_log.exists()


def test_cuda_up_requires_working_nvidia_smi_before_docker(tmp_path):
    env, docker_log = _launcher_env(tmp_path)
    bin_dir = Path(env["PATH"].split(os.pathsep)[0])
    _write_executable(bin_dir / "uname", "#!/bin/sh\necho Linux\n")
    _write_executable(bin_dir / "nvidia-smi", "#!/bin/sh\nexit 1\n")

    result = _run_launcher(["cuda", "up", "-d"], env)

    assert result.returncode != 0
    assert "nvidia-smi must be available and succeed" in result.stderr
    assert not docker_log.exists()


def test_cuda_up_runs_after_linux_nvidia_preflight_succeeds(tmp_path):
    env, docker_log = _launcher_env(tmp_path)
    bin_dir = Path(env["PATH"].split(os.pathsep)[0])
    _write_executable(bin_dir / "uname", "#!/bin/sh\necho Linux\n")
    _write_executable(bin_dir / "nvidia-smi", "#!/bin/sh\nexit 0\n")

    result = _run_launcher(["cuda", "up", "-d"], env)

    assert result.returncode == 0, result.stderr
    assert docker_log.exists()


def test_launcher_is_executable():
    assert LAUNCHER.stat().st_mode & stat.S_IXUSR
