import json
import inspect
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from json_repair import loads as json_repair_loads
from loguru import logger

from justatom.etc.schema import Document
from justatom.tooling.profiler import MemoryProfiler
from justatom.storing.dataset import API as DatasetApi


class DatasetRecordAdapter:
    def __init__(
        self,
        records: Iterable[dict[str, Any]],
        content_col: str | None = "content",
        queries_col: str | None = "queries",
        dataframe_col: str | None = None,
        chunk_id_col: str | None = None,
        keywords_col: str | None = None,
        keywords_nested_col: str | None = "keyword_or_phrase",
        explanation_nested_col: str | None = "explanation",
        filter_fields: list[str] | None = None,
        preserve_all_fields: bool = True,
    ):
        self.records = records
        self.content_col = content_col
        self.queries_col = queries_col
        self.dataframe_col = dataframe_col
        self.chunk_id_col = chunk_id_col
        self.keywords_col = keywords_col
        self.keywords_nested_col = keywords_nested_col
        self.explanation_nested_col = explanation_nested_col
        self.filter_fields = filter_fields or []
        self.preserve_all_fields = preserve_all_fields

    @classmethod
    def from_source(
        cls,
        dataset_name_or_path: str | Path,
        lazy: bool = False,
        **kwargs,
    ) -> "DatasetRecordAdapter":
        adapter_param_names = {
            name
            for name in inspect.signature(cls.__init__).parameters
            if name not in {"self", "records"}
        }
        adapter_kwargs = {
            key: value for key, value in kwargs.items() if key in adapter_param_names
        }
        dataset_kwargs = {
            key: value for key, value in kwargs.items() if key not in adapter_param_names
        }

        maybe_df_or_iter = DatasetApi.named(str(dataset_name_or_path)).iterator(
            lazy=lazy, **dataset_kwargs
        )
        records = cls._to_records(maybe_df_or_iter, lazy=lazy)

        return cls(records=records, **adapter_kwargs)

    @staticmethod
    def _to_records(
        maybe_df_or_iter: Any,
        lazy: bool,
    ) -> list[dict[str, Any]] | Generator[dict[str, Any], None, None]:
        if isinstance(maybe_df_or_iter, pl.DataFrame):
            return (
                maybe_df_or_iter.iter_rows(named=True)
                if lazy
                else maybe_df_or_iter.to_dicts()
            )
        if isinstance(maybe_df_or_iter, pl.LazyFrame):
            if not lazy:
                pl_view = maybe_df_or_iter.collect(streaming=True)
                return pl_view.to_dicts()

            def _iter_lazyframe_rows() -> Generator[dict[str, Any], None, None]:
                for batch in maybe_df_or_iter.collect_batches(maintain_order=True):
                    for row in batch.iter_rows(named=True):
                        yield row

            return _iter_lazyframe_rows()
        if not isinstance(maybe_df_or_iter, Iterable):
            msg = (
                "Dataset API returned unsupported value. Expected one of: "
                "polars.DataFrame, polars.LazyFrame, or iterable of dict-like rows."
            )
            logger.error(msg)
            raise TypeError(msg)

        coerced_records = DatasetRecordAdapter._coerce_iter(maybe_df_or_iter)
        return coerced_records if lazy else list(coerced_records)

    @staticmethod
    def _coerce_iter(raw_iter: Iterable[Any]) -> Generator[dict[str, Any], None, None]:
        for item in raw_iter:
            if isinstance(item, (bytes, bytearray)):
                try:
                    yield json.loads(item.decode("utf-8"))
                except Exception as ex:
                    msg = "Iterator yielded bytes that could not be parsed as UTF-8 JSON objects."
                    logger.error(msg)
                    raise ValueError(msg) from ex
            else:
                yield item

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        try:
            if isinstance(value, float) and np.isnan(value):
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def _maybe_parse_json_string(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        raw = value.strip()
        if raw == "":
            return value
        repaired_obj = json_repair_loads(raw)
        if repaired_obj == "" and raw not in {'""', "''"}:
            return value
        return repaired_obj

    @classmethod
    def normalize_labels(cls, raw: Any) -> list[str]:
        raw = cls._maybe_parse_json_string(raw)
        if cls._is_missing(raw):
            return []
        if isinstance(raw, str):
            return [raw] if raw.strip() else []
        if isinstance(raw, Iterable):
            return [str(x).strip() for x in raw if not cls._is_missing(x) and str(x).strip()]
        return [str(raw).strip() ] if str(raw).strip() else []

    @classmethod
    def normalize_queries(cls, raw: Any) -> list[str]:
        return cls.normalize_labels(raw)

    @classmethod
    def normalize_keywords(
        cls,
        raw: Any,
        keywords_nested_col: str | None,
        explanation_nested_col: str | None,
    ) -> list[dict[str, str]]:
        raw = cls._maybe_parse_json_string(raw)
        if cls._is_missing(raw):
            return []
        if isinstance(raw, str):
            if keywords_nested_col is None:
                return []
            return [{"keyword_or_phrase": raw}] if raw.strip() else []
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, Iterable):
            return []

        output: list[dict[str, str]] = []
        for item in raw:
            if cls._is_missing(item):
                continue
            if isinstance(item, str):
                if keywords_nested_col is None:
                    continue
                if item.strip():
                    output.append({"keyword_or_phrase": item})
                continue
            if not isinstance(item, dict):
                continue

            key_candidates = []
            if keywords_nested_col:
                key_candidates.extend(
                    [keywords_nested_col, "keyword_or_phrase", "keywords_or_phrase"]
                )
            exp_candidates = []
            if explanation_nested_col:
                exp_candidates.extend([explanation_nested_col, "explanation"])

            keyword_value = None
            for key_name in key_candidates:
                if key_name is not None and not cls._is_missing(item.get(key_name)):
                    keyword_value = item.get(key_name)
                    break
            if cls._is_missing(keyword_value):
                continue

            normalized = {"keyword_or_phrase": str(keyword_value)}
            explanation_value = None
            for key_name in exp_candidates:
                if key_name is not None and not cls._is_missing(item.get(key_name)):
                    explanation_value = item.get(key_name)
                    break
            if not cls._is_missing(explanation_value):
                normalized["explanation"] = str(explanation_value)
            output.append(normalized)

        return output

    def iterator(
        self,
        profiler: MemoryProfiler | None = None,
        as_json: bool = True,
        preserve_all_fields: bool | None = None,
    ) -> Generator[dict[str, Any] | Document, None, None]:
        active_profiler = profiler or MemoryProfiler(enabled=False)
        preserve_fields = self.preserve_all_fields if preserve_all_fields is None else preserve_all_fields
        seen_chunk_ids: set[str] = set()

        with active_profiler.span(
            "DatasetRecordAdapter.iterator",
            rows_source_type=type(self.records).__name__,
            preserve_all_fields=preserve_fields,
        ) as span:
            for row in self.records:
                if not isinstance(row, dict):
                    continue
                if any(self._is_missing(row.get(field)) for field in self.filter_fields):
                    continue
                if self._is_missing(row.get(self.content_col)):
                    continue

                content = str(row[self.content_col])
                raw_labels = row.get(self.queries_col) if self.queries_col else None
                if self._is_missing(raw_labels):
                    maybe_meta = row.get("meta")
                    if isinstance(maybe_meta, dict):
                        if self.queries_col:
                            raw_labels = maybe_meta.get(self.queries_col)
                        if self._is_missing(raw_labels):
                            raw_labels = maybe_meta.get("labels")
                labels = (
                    self.normalize_labels(raw_labels)
                    if self.queries_col is not None
                    else []
                )

                field_map: dict[str, str] = {}
                if self.content_col and self.content_col != "content":
                    field_map[self.content_col] = "content"
                if self.dataframe_col and self.dataframe_col != "dataframe":
                    field_map[self.dataframe_col] = "dataframe"
                if self.chunk_id_col and self.chunk_id_col != "id":
                    field_map[self.chunk_id_col] = "id"

                if preserve_fields:
                    source = row
                else:
                    source = {}
                    if self.content_col:
                        source[self.content_col] = row.get(self.content_col)
                    if self.dataframe_col:
                        source[self.dataframe_col] = row.get(self.dataframe_col)
                    if self.chunk_id_col:
                        source[self.chunk_id_col] = row.get(self.chunk_id_col)
                    maybe_meta = row.get("meta")
                    if isinstance(maybe_meta, dict):
                        source["meta"] = maybe_meta

                doc = Document.from_dict(
                    source,
                    field_map=field_map,
                    store_extra_fields_in_meta=preserve_fields,
                )
                doc.content = content
                doc.meta = doc.meta or {}
                doc.meta["labels"] = labels

                if self.chunk_id_col:
                    chunk_id = row.get(self.chunk_id_col)
                    if not self._is_missing(chunk_id):
                        normalized_chunk_id = str(chunk_id)
                        if normalized_chunk_id in seen_chunk_ids:
                            msg = f"Duplicate chunk_id detected: '{normalized_chunk_id}'"
                            raise ValueError(msg)
                        seen_chunk_ids.add(normalized_chunk_id)
                        doc.id = normalized_chunk_id
                        if not preserve_fields:
                            doc.meta["chunk_id"] = normalized_chunk_id

                if self.keywords_col is not None:
                    normalized_keywords = self.normalize_keywords(
                        row.get(self.keywords_col),
                        keywords_nested_col=self.keywords_nested_col,
                        explanation_nested_col=self.explanation_nested_col,
                    )
                    doc.meta["keywords_or_phrases"] = normalized_keywords

                span.tick()
                if as_json:
                    doc_json = doc.to_dict()
                    yield doc_json
                else:
                    yield doc

    def dataset(
        self,
        profiler: MemoryProfiler | None = None,
        as_json: bool = True,
        preserve_all_fields: bool | None = None,
    ) -> list[dict[str, Any] | Document]:
        logger.warning(
            "DatasetRecordAdapter.dataset() materializes the whole iterator in memory. "
            "Use iterator() for streaming large datasets."
        )
        active_profiler = profiler or MemoryProfiler(enabled=False)
        documents: list[dict[str, Any] | Document] = []
        with active_profiler.span("DatasetRecordAdapter.dataset") as span:
            for row in self.iterator(as_json=as_json, preserve_all_fields=preserve_all_fields):
                documents.append(row)
                span.tick()
        return documents

    @staticmethod
    def extract_labels(documents: Iterable[dict[str, Any] | Document]) -> list[str]:
        labels_out: list[str] = []
        for doc in documents:
            if isinstance(doc, Document):
                labels = doc.meta.get("labels", []) if doc.meta is not None else []
            else:
                labels = doc.get("meta", {}).get("labels", [])
            if isinstance(labels, str):
                labels = [labels]
            if isinstance(labels, Iterable):
                labels_out.extend([q for q in labels if isinstance(q, str) and q.strip()])
        return labels_out

    @staticmethod
    def extract_queries(documents: Iterable[dict[str, Any] | Document]) -> list[str]:
        return DatasetRecordAdapter.extract_labels(documents)
