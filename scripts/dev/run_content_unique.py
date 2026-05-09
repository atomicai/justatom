from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from justatom.api import train as train_api
from justatom.configuring.scenarios import parse_unknown_overrides
from justatom.running.trainer_jobs import create_training_job


def _resolve_report_path(resolved_kwargs: dict[str, Any], explicit_report_path: str | None) -> Path:
    if explicit_report_path is not None:
        return Path(explicit_report_path)

    save_dir = resolved_kwargs.get("save_dir")
    if save_dir is not None:
        return Path(save_dir) / "content_unique_batching.json"
    return REPO_ROOT / ".tmp_runs" / "content_unique_batching.json"


def _build_content_unique_sampler(*, batch_size: int, report_path: Path):
    def _sample_training_rows_content_unique(**kwargs):
        sampled_rows, lexical_text_by_content = train_api.sample_training_rows(**kwargs)
        duplicate_batches_before = train_api.count_batches_with_duplicate_content(sampled_rows, batch_size=batch_size)
        rebalanced_rows = train_api.rebalance_rows_by_content(sampled_rows, batch_size=batch_size)
        duplicate_batches_after = train_api.count_batches_with_duplicate_content(rebalanced_rows, batch_size=batch_size)

        summary = {
            "num_rows": len(sampled_rows),
            "batch_size": batch_size,
            "duplicate_batches_before": duplicate_batches_before,
            "duplicate_batches_after": duplicate_batches_after,
            "improved_batches": duplicate_batches_before - duplicate_batches_after,
            "unique_contents": len({str(row.get("content", "")) for row in sampled_rows}),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info(
            "Content-unique batching: duplicate batches before={} after={} rows={} report={}",
            duplicate_batches_before,
            duplicate_batches_after,
            len(sampled_rows),
            report_path,
        )
        return rebalanced_rows, lexical_text_by_content

    return _sample_training_rows_content_unique


def _parse_args(argv: list[str] | None = None) -> tuple[str | None, str | None, dict[str, Any] | None]:
    parser = argparse.ArgumentParser(
        prog="run_content_unique",
        description="Train with content-unique batch ordering while keeping the standard loss and trainer unchanged.",
    )
    parser.add_argument("--config")
    parser.add_argument("--report-path")
    args, unknown = parser.parse_known_args(argv)
    overrides = parse_unknown_overrides(unknown)
    return args.config, args.report_path, overrides or None


def main(argv: list[str] | None = None) -> str:
    config_path, explicit_report_path, overrides = _parse_args(argv)
    resolved_kwargs = train_api.resolve_train_kwargs(
        config_path=config_path,
        overrides=overrides,
    )

    dataset_name_or_path = resolved_kwargs.get("dataset_name_or_path")
    if dataset_name_or_path is None:
        raise ValueError("dataset_name_or_path must be provided for training")
    raw_dataset_name_or_path = str(dataset_name_or_path)
    if "://" not in raw_dataset_name_or_path and raw_dataset_name_or_path != "justatom":
        resolved_kwargs["dataset_name_or_path"] = Path(raw_dataset_name_or_path)

    report_path = _resolve_report_path(resolved_kwargs, explicit_report_path)
    batch_size = int(resolved_kwargs.get("batch_size", 1))

    job = create_training_job(
        prepare_training_data_fn=train_api.prepare_training_data,
        sample_training_rows_fn=_build_content_unique_sampler(
            batch_size=batch_size,
            report_path=report_path,
        ),
        roll_metrics_path_fn=train_api._roll_metrics_path_if_exists,
        **resolved_kwargs,
    )
    logger.info(f"Selected training mode: {job.training_mode}")
    return job.train()


if __name__ == "__main__":
    main(sys.argv[1:])
