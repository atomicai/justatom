from __future__ import annotations

import random
from collections.abc import Generator, Iterable
from itertools import islice
from pathlib import Path
from typing import Any

import polars as pl

from justatom.storing.dataset import API as DatasetApi
from justatom.tooling.dataset import DatasetRecordAdapter
from justatom.training.config import TrainConfig


def _row_passes_filters(row: dict[str, Any], filters: dict | None) -> bool:
    filter_fields = (filters or {}).get("fields") or []
    return not any(DatasetRecordAdapter._is_missing(row.get(field)) for field in filter_fields)


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
        if not isinstance(row, dict) or not _row_passes_filters(row, filters):
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
            payload = {"queries": query, "content": content, "lexical_text": lexical_text}
            if chunk_id is not None:
                payload["chunk_id"] = str(chunk_id)
            yield payload


def _iterate_from_frame_batches(
    frame_batches: Iterable[pl.DataFrame],
    **kwargs: Any,
) -> Generator[dict[str, Any], None, None]:
    for batch in frame_batches:
        yield from _iterate_from_raw_samples(batch.iter_rows(named=True), **kwargs)


def _frame_batches_from_source(
    *,
    dataset_name_or_path: str | Path,
    split: str | None,
) -> Iterable[pl.DataFrame] | None:
    source: Any = DatasetApi.named(str(dataset_name_or_path)).iterator(lazy=True, split=split)
    if isinstance(source, pl.DataFrame):
        return [source]
    if isinstance(source, pl.LazyFrame):
        return source.collect_batches(maintain_order=True)
    return None


def _iterate_adapter_rows(docs_adapter: DatasetRecordAdapter) -> Generator[dict[str, Any], None, None]:
    for document in docs_adapter.iterator():
        yield {
            "content": document.get("content"),
            "queries": document.get("meta", {}).get("labels", []),
            "chunk_id": document.get("id"),
            "keywords_or_phrases": document.get("meta", {}).get("keywords_or_phrases", []),
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
    frame_batches = _frame_batches_from_source(dataset_name_or_path=dataset_name_or_path, split=split)
    if frame_batches is not None:
        rows = _iterate_from_frame_batches(frame_batches, **source_kwargs)
        return rows if limit is None else islice(rows, int(limit))

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
    rows = _iterate_from_raw_samples(
        _iterate_adapter_rows(docs_adapter),
        content_field="content",
        labels_field="queries",
        chunk_id_col="chunk_id" if chunk_id_col is not None else None,
        keywords_or_phrases_field="keywords_or_phrases",
        keywords_nested_col="keyword_or_phrase",
        explanation_nested_col="explanation",
        filters=None,
    )
    return rows if limit is None else islice(rows, int(limit))


def _reservoir_sample_rows(
    rows: Iterable[dict[str, Any]],
    num_samples: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    if num_samples == -1:
        return list(rows)
    if num_samples <= 0:
        return []
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index < num_samples:
            sample.append(row)
            continue
        replacement = rng.randint(0, index)
        if replacement < num_samples:
            sample[replacement] = row
    return sample


def rebalance_rows_by_content(
    rows: Iterable[dict[str, Any]],
    batch_size: int,
    *,
    content_key: str = "content",
) -> list[dict[str, Any]]:
    rows_list = list(rows)
    if batch_size <= 1 or len(rows_list) <= 1:
        return rows_list
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows_list:
        value = row.get(content_key)
        grouped.setdefault("" if value is None else str(value), []).append(row)
    if len(grouped) == len(rows_list):
        return rows_list
    result: list[dict[str, Any]] = []
    groups = list(grouped.values())
    for level in range(max(len(group) for group in groups)):
        for group in groups:
            if level < len(group):
                result.append(group[level])
    return result


def count_batches_with_duplicate_content(
    rows: Iterable[dict[str, Any]],
    batch_size: int,
    *,
    content_key: str = "content",
) -> int:
    if batch_size <= 1:
        return 0
    rows_list = list(rows)
    duplicate_batches = 0
    for start in range(0, len(rows_list), batch_size):
        batch = rows_list[start : start + batch_size]
        contents = ["" if row.get(content_key) is None else str(row.get(content_key)) for row in batch]
        duplicate_batches += int(len(contents) != len(set(contents)))
    return duplicate_batches


def sample_training_rows(
    *,
    dataset_name_or_path: str | Path,
    num_samples: int = 100,
    seed: int = 0,
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
    rows = iterate_training_rows(
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
    sampled = _reservoir_sample_rows(rows, int(num_samples), seed=seed)
    lexical_lookup = {row["content"]: row["lexical_text"] for row in sampled if row.get("content")}
    return sampled, lexical_lookup


def prepare_training_data(
    **kwargs: Any,
) -> tuple[pl.DataFrame, list[dict[str, Any]], dict[str, str | list[str]]]:
    sampled, lexical_lookup = sample_training_rows(**kwargs)
    frame = (
        pl.from_dicts(sampled)
        if sampled
        else pl.DataFrame(schema={"queries": pl.Utf8, "content": pl.Utf8, "lexical_text": pl.Object})
    )
    return frame, sampled, lexical_lookup


def prepare_training_data_from_config(
    config: TrainConfig,
) -> tuple[list[dict[str, Any]], dict[str, str | list[str]]]:
    if config.dataset.name_or_path is None:
        raise ValueError("dataset.name_or_path is required")
    rows, lexical_lookup = sample_training_rows(
        dataset_name_or_path=config.dataset.name_or_path,
        num_samples=config.optimization.num_samples,
        seed=config.experiment.seed,
        content_field=config.dataset.content_field,
        labels_field=config.dataset.labels_field,
        split=config.dataset.split,
        limit=config.dataset.limit,
        chunk_id_col=config.dataset.chunk_id_col,
        keywords_or_phrases_field=config.dataset.keywords_col,
        keywords_nested_col=config.dataset.keywords_nested_col,
        explanation_nested_col=config.dataset.explanation_nested_col,
        filters=None if config.filters.fields is None else {"fields": config.filters.fields},
    )
    rows = rebalance_rows_by_content(rows, config.optimization.batch_size)
    return rows, lexical_lookup
