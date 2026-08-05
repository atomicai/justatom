from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from justatom.configuring.scenarios import (
    deep_merge,
    load_scenario_config,
    parse_unknown_overrides,
)
from justatom.training.config import TrainConfig, TrainingMethod, parse_train_config
from justatom.training.job import TrainingJob, TrainingResult


def load_train_config(
    *,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return load_scenario_config(
        "train",
        config=config,
        config_path=config_path,
        overrides=overrides,
    )


def resolve_train_config(
    *,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> TrainConfig:
    return parse_train_config(
        load_train_config(config=config, config_path=config_path, overrides=overrides)
    )


def run(
    *,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> TrainingResult:
    return TrainingJob(
        resolve_train_config(config=config, config_path=config_path, overrides=overrides)
    ).run()


def _parse_args(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        prog="justatom|train",
        description="Train a retriever with vanilla, atom_gate, or atomic.",
    )
    parser.add_argument("--config")
    parser.add_argument("--method", choices=[method.value for method in TrainingMethod])
    args, unknown = parser.parse_known_args(sys.argv[1:] if argv is None else argv)

    overrides = parse_unknown_overrides(unknown)
    if args.method is not None:
        overrides = deep_merge(overrides, {"method": args.method})
    return {
        "config_path": args.config,
        "overrides": overrides or None,
    }


def main(argv: list[str] | None = None) -> TrainingResult:
    return run(**_parse_args(argv))


if __name__ == "__main__":
    main()


__all__ = [
    "TrainingResult",
    "load_train_config",
    "resolve_train_config",
    "run",
    "main",
]
