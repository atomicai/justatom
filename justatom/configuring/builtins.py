from __future__ import annotations

import os
import re
from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import yaml


_ENV_VAR_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")
_BUILTIN_URI_PREFIX = "builtin://"


def _builtins_root() -> Path:
    return Path(__file__).resolve().parents[1] / "builtins"


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _builtin_resource(relative_path: str):
    parts = [part for part in Path(relative_path).parts if part not in {".", ""}]
    resource = files("justatom.builtins")
    for part in parts:
        resource = resource.joinpath(part)
    return resource


def is_builtin_uri(value: str | Path | None) -> bool:
    if value is None:
        return False
    return str(value).startswith(_BUILTIN_URI_PREFIX)


def builtin_path(relative_path: str | Path) -> Path:
    relative = Path(relative_path)
    parts = [part for part in relative.parts if part not in {".", "", "/"}]
    return _builtins_root().joinpath(*parts)


def resolve_builtin_path(value: str | Path) -> Path:
    raw = str(value)
    if not is_builtin_uri(raw):
        return Path(raw)
    relative = raw[len(_BUILTIN_URI_PREFIX) :]
    return builtin_path(relative)


def _parse_yaml_text(raw: str) -> dict[str, Any]:
    rendered = render_env_template(raw)
    parsed = yaml.safe_load(rendered) or {}
    return parsed if isinstance(parsed, dict) else {}


@lru_cache(maxsize=64)
def load_builtin_yaml(relative_path: str) -> dict[str, Any]:
    resource = _builtin_resource(relative_path)
    if not resource.is_file():
        return {}
    with as_file(resource) as path:
        return _parse_yaml_text(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=64)
def load_repo_yaml(relative_or_absolute_path: str) -> dict[str, Any]:
    path = Path(relative_or_absolute_path)
    if not path.is_absolute():
        path = _workspace_root() / path
    if not path.exists():
        return {}
    return _parse_yaml_text(path.read_text(encoding="utf-8"))


def render_env_template(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        return os.getenv(key, match.group(0))

    return _ENV_VAR_RE.sub(repl, text)


@lru_cache(maxsize=64)
def load_builtin_prompt(relative_path: str) -> str:
    resource = _builtin_resource(relative_path)
    if not resource.is_file():
        return ""
    raw = resource.read_text(encoding="utf-8")
    return render_env_template(raw)


__all__ = [
    "builtin_path",
    "is_builtin_uri",
    "load_builtin_prompt",
    "load_builtin_yaml",
    "load_repo_yaml",
    "render_env_template",
    "resolve_builtin_path",
]
