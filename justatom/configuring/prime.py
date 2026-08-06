from __future__ import annotations

from copy import deepcopy
from typing import Any

import dotenv

from justatom.etc.pattern import singleton


class ConfigNode(dict):
    """Dict with recursive attribute access for legacy Config consumers."""

    def __init__(self, data: dict[str, Any] | None = None):
        super().__init__()
        for key, value in (data or {}).items():
            super().__setitem__(key, _wrap_config_value(value))

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, _wrap_config_value(value))


def _wrap_config_value(value: Any) -> Any:
    if isinstance(value, ConfigNode):
        return value
    if isinstance(value, dict):
        return ConfigNode(value)
    if isinstance(value, list):
        return [_wrap_config_value(item) for item in value]
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _default_config_data() -> dict[str, Any]:
    return {
        "loguru": {
            "LOG_FILE_NAME": "",
            "LOG_ROTATION": "10 MB",
            "LOG_RETENTION": "10 days",
        },
        "train": {
            "max_seq_len": 512,
            "index_name": "justatom",
            "dataset_path": ".data",
            "shuffle": True,
            "model": {
                "model_name_or_path": "intfloat/multilingual-e5-base",
                "props": {
                    "dropout": 0.1,
                },
            },
            "do_scale": False,
            "do_scale_unit": 1,
            "max_epochs": 5,
            "save_top_k": 2,
            "loss": "triplet",
            "loss_props": {
                "margin": 1.2,
            },
            "log_every_n_steps": 10,
            "val_check_interval": 30,
            "devices": "auto",
            "batch_size": 64,
            "early_stopping": {
                "metric": "TrainingLoss",
                "size": 20,
                "mode": "min",
            },
            "save_model_path": "weights",
            "log": {
                "log_batch_metrics": True,
                "log_epoch_metrics": True,
            },
        },
    }


def _normalize_legacy_config(data: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(data)
    train_cfg = cfg.get("train")
    if isinstance(train_cfg, dict):
        train_cfg.setdefault("index_name", "justatom")
        train_cfg.setdefault("shuffle", True)
        train_cfg.setdefault("do_scale_unit", 1)

    return cfg


def _build_config_tree(data: dict[str, Any] | None = None) -> ConfigNode:
    merged = _deep_merge(_default_config_data(), _normalize_legacy_config(data or {}))
    return ConfigNode(merged)


@singleton
class IConfig:
    loguru: ConfigNode
    train: ConfigNode

    def __init__(self):
        config = _build_config_tree()
        for k, v in config.items():
            if k.startswith("_"):
                continue
            setattr(self, k, v)
        dotenv.load_dotenv()


Config = IConfig()


__all__ = ["Config"]
