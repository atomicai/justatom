from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from loguru import logger

from justatom.configuring.builtins import load_builtin_yaml, load_repo_yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def set_dotted(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = [k.replace("-", "_") for k in dotted_key.split(".") if k]
    if not keys:
        return
    cursor = cfg
    for part in keys[:-1]:
        node = cursor.get(part)
        if not isinstance(node, dict):
            node = {}
            cursor[part] = node
        cursor = node
    cursor[keys[-1]] = value


def parse_literal(raw: str) -> Any:
    text = raw.strip()
    low = text.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return raw


def parse_unknown_overrides(tokens: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            i += 1
            continue
        body = token[2:]
        if "=" in body:
            key, raw = body.split("=", 1)
            set_dotted(out, key, parse_literal(raw))
            i += 1
            continue

        key = body
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            value = parse_literal(tokens[i + 1])
            i += 2
        else:
            value = True
            i += 1
        set_dotted(out, key, value)
    return out


def _repo_default_config_path(scenario_name: str) -> Path:
    return Path.cwd() / "configs" / f"{scenario_name}.yaml"


def _builtins_default_relative_path(scenario_name: str) -> str:
    return f"configs/{scenario_name}.default.yaml"


def _scenario_base_dir(config_path: str | Path | None) -> Path | None:
    if config_path is None:
        return None
    path = Path(config_path)
    return path.resolve().parent if path.exists() else path.parent.resolve()


def _load_yaml_overlay(path: str | Path) -> dict[str, Any]:
    cfg = load_repo_yaml(str(path))
    return cfg if isinstance(cfg, dict) else {}


def _drop_none_values(node: Any) -> Any:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            compacted = _drop_none_values(value)
            if compacted is not None:
                out[key] = compacted
        return out
    return node


def _resolve_dataset_overlay(
    cfg: dict[str, Any],
    *,
    scenario_name: str,
    scenario_config_path: str | Path | None,
) -> dict[str, Any]:
    dataset_cfg = cfg.get("dataset") or {}
    if not isinstance(dataset_cfg, dict):
        return cfg

    overlay: dict[str, Any] = {}
    config_path = dataset_cfg.get("config_path")
    dataset_id = dataset_cfg.get("id")

    if config_path:
        overlay = _load_yaml_overlay(str(config_path))
    elif dataset_id:
        candidates: list[Path] = []
        base_dir = _scenario_base_dir(scenario_config_path)
        if base_dir is not None:
            candidates.append(base_dir / "dataset" / f"{dataset_id}.yaml")
        candidates.append(Path.cwd() / "configs" / "dataset" / f"{dataset_id}.yaml")

        for candidate in candidates:
            if candidate.exists():
                overlay = _load_yaml_overlay(candidate)
                break
        else:
            builtin_overlay = load_builtin_yaml(f"configs/dataset/{dataset_id}.yaml")
            if isinstance(builtin_overlay, dict) and builtin_overlay:
                overlay = builtin_overlay
            else:
                logger.warning(
                    "Dataset preset '{}' was requested but no config file was found.",
                    dataset_id,
                )

    if not overlay:
        return cfg

    builtin_defaults = load_builtin_yaml(_builtins_default_relative_path(scenario_name))
    builtin_dataset_defaults = builtin_defaults.get("dataset") or {}
    explicit_dataset_cfg = _drop_none_values(dataset_cfg)
    if isinstance(explicit_dataset_cfg, dict) and isinstance(builtin_dataset_defaults, dict):
        explicit_dataset_cfg = {
            key: value for key, value in explicit_dataset_cfg.items() if builtin_dataset_defaults.get(key) != value
        }

    out = dict(cfg)
    out["dataset"] = deep_merge(overlay, explicit_dataset_cfg)
    return out


def load_scenario_config(
    scenario_name: str,
    *,
    config_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = load_builtin_yaml(_builtins_default_relative_path(scenario_name))
    cfg = cfg if isinstance(cfg, dict) else {}

    resolved_config_path: str | Path | None = config_path
    if resolved_config_path is None:
        default_path = _repo_default_config_path(scenario_name)
        if default_path.exists():
            resolved_config_path = default_path

    if resolved_config_path is not None:
        path = Path(resolved_config_path)
        if config_path is not None and not path.exists():
            raise FileNotFoundError(f"Config file does not exist: {path}")
        if path.exists():
            cfg = deep_merge(cfg, _load_yaml_overlay(path))

    if config:
        cfg = deep_merge(cfg, config)

    # Allow CLI/dotted dataset overrides (for example `dataset.id=...`) to affect
    # which dataset preset is loaded, then re-apply them below to preserve
    # explicit override precedence over the preset values.
    pre_overlay_cfg = deep_merge(cfg, overrides) if overrides else cfg

    cfg = _resolve_dataset_overlay(
        pre_overlay_cfg,
        scenario_name=scenario_name,
        scenario_config_path=resolved_config_path,
    )

    if overrides:
        cfg = deep_merge(cfg, overrides)

    return cfg
