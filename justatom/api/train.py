from __future__ import annotations

import argparse
import os
import random
import sys
from collections.abc import Generator, Iterable
from itertools import islice
from pathlib import Path
from typing import Any

import dotenv
import polars as pl
from loguru import logger

from justatom.configuring.scenarios import load_scenario_config, parse_unknown_overrides
from justatom.running.trainer_jobs import (
    BaseTrainingJob,
    EncoderGammaTrainingJob,
    EncoderOnlyTrainingJob,
    GammaOnlyTrainingJob,
    create_training_job as _create_training_job,
)
from justatom.storing.dataset import API as DatasetApi
from justatom.tooling.dataset import DatasetRecordAdapter

dotenv.load_dotenv()

logger.info(
    f"Enable MPS fallback = {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', -1)}"
)


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


def _cfg_to_train_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset = cfg.get("dataset") or {}
    model = cfg.get("model") or {}
    training = cfg.get("training") or {}
    output = cfg.get("output") or {}
    logging_cfg = cfg.get("logging") or {}
    filters_cfg = cfg.get("filters") or {}

    return {
        "dataset_name_or_path": dataset.get("name_or_path"),
        "model_name_or_path": model.get("name", "intfloat/multilingual-e5-small"),
        "loss": training.get("loss", "contrastive"),
        "num_samples": int(training.get("num_samples", 100)),
        "batch_size": int(training.get("batch_size", 4)),
        "max_seq_len": int(training.get("max_seq_len", 512)),
        "freeze_encoder": bool(training.get("freeze_encoder", True)),
        "include_semantic_gamma": bool(training.get("include_semantic_gamma", True)),
        "include_keywords_gamma": bool(training.get("include_keywords_gamma", True)),
        "activation_fn": training.get("activation_fn", "sigmoid"),
        "focal_gamma": float(training.get("focal_gamma", 2.0)),
        "log_backend": logging_cfg.get("backend", "csv"),
        "wandb_project": logging_cfg.get("wandb_project", "justatom-gamma"),
        "wandb_run_name": logging_cfg.get("wandb_run_name"),
        "n_epochs": int(training.get("n_epochs", 1)),
        "content_field": dataset.get("content_field", "content"),
        "labels_field": dataset.get("labels_field", "queries"),
        "split": dataset.get("split"),
        "limit": dataset.get("limit"),
        "chunk_id_col": dataset.get("chunk_id_col"),
        "keywords_or_phrases_field": dataset.get("keywords_col"),
        "keywords_nested_col": dataset.get("keywords_nested_col"),
        "explanation_nested_col": dataset.get("explanation_nested_col"),
        "filters": (
            {"fields": filters_cfg.get("fields")} if filters_cfg.get("fields") else None
        ),
        "lr_gamma": float(training.get("lr_gamma", 1e-2)),
        "lr_encoder": float(training.get("lr_encoder", 2e-5)),
        "weight_decay": float(training.get("weight_decay", 0.01)),
        "save_dir": output.get("save_dir"),
        "metrics_path": output.get("metrics_path"),
    }


def resolve_train_kwargs(
    *,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = load_train_config(
        config=config,
        config_path=config_path,
        overrides=overrides,
    )
    return _cfg_to_train_kwargs(cfg)


def _roll_metrics_path_if_exists(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}.{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def create_training_job(**kwargs) -> BaseTrainingJob:
    return _create_training_job(
        prepare_training_data_fn=prepare_training_data,
        sample_training_rows_fn=sample_training_rows,
        roll_metrics_path_fn=_roll_metrics_path_if_exists,
        **kwargs,
    )


def _row_passes_filters(row: dict[str, Any], filters: dict | None) -> bool:
    filter_fields = (filters or {}).get("fields") or []
    return not any(
        DatasetRecordAdapter._is_missing(row.get(field)) for field in filter_fields
    )


def _normalize_lexical_text(
    row: dict[str, Any],
    *,
    content_field: str,
    keywords_or_phrases_field: str | None,
    keywords_nested_col: str | None,
    explanation_nested_col: str | None,
) -> str | list[str]:
    normalized_keywords = DatasetRecordAdapter.normalize_keywords(
        row.get(keywords_or_phrases_field) if keywords_or_phrases_field else None,
        keywords_nested_col=keywords_nested_col,
        explanation_nested_col=explanation_nested_col,
    )
    if normalized_keywords:
        return [str(item["keyword_or_phrase"]) for item in normalized_keywords]
    return str(row.get(content_field, ""))


def _iterate_from_raw_samples(
    samples: Iterable[dict[str, Any]],
    *,
    content_field: str,
    labels_field: str,
    chunk_id_col: str | None,
    keywords_or_phrases_field: str | None,
    keywords_nested_col: str | None,
    explanation_nested_col: str | None,
    filters: dict | None,
) -> Generator[dict[str, Any], None, None]:
    for row in samples:
        if not isinstance(row, dict):
            continue
        if not _row_passes_filters(row, filters):
            continue
        if DatasetRecordAdapter._is_missing(row.get(content_field)):
            continue

        content = str(row.get(content_field))
        queries = DatasetRecordAdapter.normalize_queries(row.get(labels_field))
        if not queries:
            continue

        lexical_text = _normalize_lexical_text(
            row,
            content_field=content_field,
            keywords_or_phrases_field=keywords_or_phrases_field,
            keywords_nested_col=keywords_nested_col,
            explanation_nested_col=explanation_nested_col,
        )
        chunk_id = None if chunk_id_col is None else row.get(chunk_id_col)

        for query in queries:
            payload = {
                "queries": query,
                "content": content,
                "lexical_text": lexical_text,
            }
            if chunk_id is not None:
                payload["chunk_id"] = str(chunk_id)
            yield payload


def _iterate_from_frame_batches(
    frame_batches: Iterable[pl.DataFrame],
    *,
    content_field: str,
    labels_field: str,
    chunk_id_col: str | None,
    keywords_or_phrases_field: str | None,
    keywords_nested_col: str | None,
    explanation_nested_col: str | None,
    filters: dict | None,
) -> Generator[dict[str, Any], None, None]:
    for batch in frame_batches:
        # `batch` is one materialized DataFrame chunk, not the whole dataset.
        yield from _iterate_from_raw_samples(
            batch.iter_rows(named=True),
            content_field=content_field,
            labels_field=labels_field,
            chunk_id_col=chunk_id_col,
            keywords_or_phrases_field=keywords_or_phrases_field,
            keywords_nested_col=keywords_nested_col,
            explanation_nested_col=explanation_nested_col,
            filters=filters,
        )


def _frame_batches_from_source(
    *,
    dataset_name_or_path: str | Path,
    split: str | None,
) -> Iterable[pl.DataFrame] | None:
    source: Any = DatasetApi.named(str(dataset_name_or_path)).iterator(
        lazy=True, split=split
    )  # Might be a polars LazyFrame, a polars DataFrame, or HuggingFace datasets object
    if isinstance(source, pl.DataFrame):
        # Eager `.json` fallback: the backend already loaded the whole dataset.
        # We wrap it into a one-item iterable so downstream code can treat both
        # eager and lazy sources uniformly as `for batch in frame_batches`.
        return [source]
    if isinstance(source, pl.LazyFrame):
        # Real lazy path: polars yields the dataset as multiple DataFrame batches.
        return source.collect_batches(maintain_order=True)
    # Returns None when loading dataset from huggingface
    return None


def _iterate_adapter_rows(
    docs_adapter: DatasetRecordAdapter,
) -> Generator[dict[str, Any], None, None]:
    for doc in docs_adapter.iterator():
        yield {
            "content": doc.get("content"),
            "queries": doc.get("meta", {}).get("labels", []),
            "chunk_id": doc.get("id"),
            "keywords_or_phrases": doc.get("meta", {}).get("keywords_or_phrases", []),
        }


def iterate_training_rows(
    *,
    dataset_name_or_path: str | Path,
    content_field: str = "content",
    labels_field: str = "queries",
    split: str | None = None,
    limit: int | None = None,
    chunk_id_col: str | None = None,
    keywords_or_phrases_field: str | None = None,
    keywords_nested_col: str | None = None,
    explanation_nested_col: str | None = None,
    filters: dict | None = None,
) -> Iterable[dict[str, Any]]:
    if dataset_name_or_path is None:
        raise ValueError("dataset_name_or_path must be provided for training")

    source_kwargs = {
        "content_field": content_field,
        "labels_field": labels_field,
        "chunk_id_col": chunk_id_col,
        "keywords_or_phrases_field": keywords_or_phrases_field,
        "keywords_nested_col": keywords_nested_col,
        "explanation_nested_col": explanation_nested_col,
        "filters": filters,
    }

    frame_batches = _frame_batches_from_source(
        dataset_name_or_path=dataset_name_or_path,
        split=split,
    )
    if frame_batches is not None:
        # `rows_iter` is a lazy stream of training rows produced from one batch
        # at a time. Only the current batch is materialized in this branch.
        rows_iter = _iterate_from_frame_batches(frame_batches, **source_kwargs)
        return rows_iter if limit is None else islice(rows_iter, int(limit))

    docs_adapter = DatasetRecordAdapter.from_source(
        dataset_name_or_path=str(dataset_name_or_path),
        lazy=True,
        content_col=content_field,
        queries_col=labels_field,
        split=split,
        chunk_id_col=chunk_id_col,
        keywords_col=keywords_or_phrases_field,
        keywords_nested_col=keywords_nested_col,
        explanation_nested_col=explanation_nested_col,
        filter_fields=(filters or {}).get("fields", []),
    )
    rows_iter = _iterate_from_raw_samples(
        _iterate_adapter_rows(docs_adapter),
        content_field="content",
        labels_field="queries",
        chunk_id_col="chunk_id" if chunk_id_col is not None else None,
        keywords_or_phrases_field="keywords_or_phrases",
        keywords_nested_col="keyword_or_phrase",
        explanation_nested_col="explanation",
        filters=None,
    )
    # Fallback branch: still return a lazy row iterator, but the underlying
    # adapter may already sit on top of an eager in-memory dataset.
    return rows_iter if limit is None else islice(rows_iter, int(limit))


def _reservoir_sample_rows(
    rows: Iterable[dict[str, Any]], num_samples: int
) -> list[dict[str, Any]]:
    if num_samples <= 0:
        return []

    rng = random.Random(0)
    # `sample` is the in-memory training subset we keep for fitting; the full
    # incoming dataset is never stored here.
    sample: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if idx < num_samples:
            sample.append(row)
            continue
        replacement_idx = rng.randint(0, idx)
        if replacement_idx < num_samples:
            sample[replacement_idx] = row
    return sample


def sample_training_rows(
    *,
    dataset_name_or_path: str | Path,
    num_samples: int = 100,
    content_field: str = "content",
    labels_field: str = "queries",
    split: str | None = None,
    limit: int | None = None,
    chunk_id_col: str | None = None,
    keywords_or_phrases_field: str | None = None,
    keywords_nested_col: str | None = None,
    explanation_nested_col: str | None = None,
    filters: dict | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str | list[str]]]:
    # `rows_iter` is a lazy iterator over the full training source.
    rows_iter = iterate_training_rows(
        dataset_name_or_path=dataset_name_or_path,
        content_field=content_field,
        labels_field=labels_field,
        split=split,
        limit=limit,
        chunk_id_col=chunk_id_col,
        keywords_or_phrases_field=keywords_or_phrases_field,
        keywords_nested_col=keywords_nested_col,
        explanation_nested_col=explanation_nested_col,
        filters=filters,
    )
    # `sampled_rows` is the bounded in-memory subset used by training.
    sampled_rows = _reservoir_sample_rows(rows_iter, int(num_samples))
    lexical_text_by_content = {
        row["content"]: row["lexical_text"]
        for row in sampled_rows
        if row.get("content")
    }
    return sampled_rows, lexical_text_by_content


def prepare_training_data(
    *,
    dataset_name_or_path: str | Path,
    num_samples: int = 100,
    content_field: str = "content",
    labels_field: str = "queries",
    split: str | None = None,
    limit: int | None = None,
    chunk_id_col: str | None = None,
    keywords_or_phrases_field: str | None = None,
    keywords_nested_col: str | None = None,
    explanation_nested_col: str | None = None,
    filters: dict | None = None,
) -> tuple[pl.DataFrame, list[dict[str, Any]], dict[str, str | list[str]]]:
    # Public compatibility wrapper: first sample a bounded subset, then
    # materialize only that subset into `js_data` / `pl_data`.
    sampled_rows, lexical_text_by_content = sample_training_rows(
        dataset_name_or_path=dataset_name_or_path,
        num_samples=num_samples,
        content_field=content_field,
        labels_field=labels_field,
        split=split,
        limit=limit,
        chunk_id_col=chunk_id_col,
        keywords_or_phrases_field=keywords_or_phrases_field,
        keywords_nested_col=keywords_nested_col,
        explanation_nested_col=explanation_nested_col,
        filters=filters,
    )
    # `js_data` is not the whole dataset; it is only the sampled training rows.
    js_data = sampled_rows
    pl_data = (
        pl.from_dicts(js_data)
        if js_data
        else pl.DataFrame(
            schema={
                "queries": pl.Utf8,
                "content": pl.Utf8,
                "lexical_text": pl.Object,
            }
        )
    )
    return pl_data, js_data, lexical_text_by_content


def train_model(**kwargs) -> str:
    dataset_name_or_path = kwargs.get("dataset_name_or_path")
    if dataset_name_or_path is None:
        raise ValueError("dataset_name_or_path must be provided for training")

    resolved_kwargs = dict(kwargs)
    raw_dataset_name_or_path = str(dataset_name_or_path)
    if "://" not in raw_dataset_name_or_path and raw_dataset_name_or_path != "justatom":
        resolved_kwargs["dataset_name_or_path"] = Path(raw_dataset_name_or_path)

    job = create_training_job(**resolved_kwargs)
    logger.info(f"Selected training mode: {job.training_mode}")
    return job.train()


def run(
    *,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> str:
    resolved = resolve_train_kwargs(
        config=config,
        config_path=config_path,
        overrides=overrides,
    )
    return train_model(**resolved)


def _parse_args(argv: list[str] | None = None) -> dict[str, Any]:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="justatom|train",
        description="Train encoder and gamma retrievers from a scenario config",
    )
    parser.add_argument("--config")
    args, unknown = parser.parse_known_args(argv)

    overrides = parse_unknown_overrides(unknown)
    return {
        "config_path": args.config,
        "overrides": overrides or None,
    }


def main(argv: list[str] | None = None) -> str:
    parsed = _parse_args(argv)
    return run(**parsed)


if __name__ == "__main__":
    main()


__all__ = [
    "BaseTrainingJob",
    "GammaOnlyTrainingJob",
    "EncoderGammaTrainingJob",
    "EncoderOnlyTrainingJob",
    "DatasetApi",
    "DatasetRecordAdapter",
    "create_training_job",
    "load_train_config",
    "resolve_train_kwargs",
    "iterate_training_rows",
    "sample_training_rows",
    "prepare_training_data",
    "train_model",
    "run",
    "main",
]
