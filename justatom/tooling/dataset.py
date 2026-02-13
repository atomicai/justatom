import json
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from json_repair import loads as json_repair_loads
from loguru import logger

from justatom.storing.dataset import API as DatasetApi


class DatasetRecordAdapter:
    def __init__(
        self,
        records: Iterable[dict[str, Any]],
        content_col: str,
        queries_col: str | None = "queries",
        chunk_id_col: str | None = None,
        keywords_col: str | None = None,
        keywords_nested_col: str | None = "keyword_or_phrase",
        explanation_nested_col: str | None = "explanation",
        filter_fields: list[str] | None = None,
    ):
        self.records = records
        self.content_col = content_col
        self.queries_col = queries_col
        self.chunk_id_col = chunk_id_col
        self.keywords_col = keywords_col
        self.keywords_nested_col = keywords_nested_col
        self.explanation_nested_col = explanation_nested_col
        self.filter_fields = filter_fields or []

    @classmethod
    def from_source(
        cls,
        dataset_name_or_path: str | Path,
        **kwargs,
    ) -> "DatasetRecordAdapter":
        maybe_df_or_iter = DatasetApi.named(str(dataset_name_or_path)).iterator(
            lazy=True
        )

        if isinstance(maybe_df_or_iter, pl.DataFrame):
            records = maybe_df_or_iter.iter_rows(named=True)
        elif isinstance(maybe_df_or_iter, pl.LazyFrame):
            records = maybe_df_or_iter.collect(streaming=True).iter_rows(named=True)
        elif isinstance(maybe_df_or_iter, (bytes, bytearray)):
            msg = "Dataset API returned raw bytes. Expected a path/DataFrame/iterator of dicts."
            logger.error(msg)
            raise ValueError(msg)
        else:
            records = cls._coerce_iter(maybe_df_or_iter)

        return cls(records=records, **kwargs)

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
    def normalize_queries(cls, raw: Any) -> list[str]:
        raw = cls._maybe_parse_json_string(raw)
        if cls._is_missing(raw):
            return []
        if isinstance(raw, str):
            return [raw] if raw.strip() else []
        if isinstance(raw, Iterable):
            return [str(x) for x in raw if not cls._is_missing(x) and str(x).strip()]
        return [str(raw)]

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

    def iter_documents(self) -> Generator[dict[str, Any], None, None]:
        seen_chunk_ids = set()

        for row in self.records:
            if not isinstance(row, dict):
                continue
            if any(self._is_missing(row.get(field)) for field in self.filter_fields):
                continue
            if self._is_missing(row.get(self.content_col)):
                continue

            content = str(row[self.content_col])
            queries = (
                self.normalize_queries(row.get(self.queries_col))
                if self.queries_col
                else []
            )

            out: dict[str, Any] = {"content": content, "meta": {"labels": queries}}

            if self.chunk_id_col:
                chunk_id = row.get(self.chunk_id_col)
                if not self._is_missing(chunk_id):
                    normalized_chunk_id = str(chunk_id)
                    if normalized_chunk_id in seen_chunk_ids:
                        msg = f"chunk_id must be unique. Duplicate found: {normalized_chunk_id}"
                        logger.error(msg)
                        raise ValueError(msg)
                    seen_chunk_ids.add(normalized_chunk_id)
                    out["id"] = normalized_chunk_id
                    out["meta"]["chunk_id"] = normalized_chunk_id

            if self.keywords_col is not None:
                normalized_keywords = self.normalize_keywords(
                    row.get(self.keywords_col),
                    keywords_nested_col=self.keywords_nested_col,
                    explanation_nested_col=self.explanation_nested_col,
                )
                if normalized_keywords:
                    out["keywords_or_phrases"] = normalized_keywords

            yield out

    def iter_labels(self) -> Generator[str, None, None]:
        for row in self.records:
            if not isinstance(row, dict):
                continue
            if any(self._is_missing(row.get(field)) for field in self.filter_fields):
                continue
            if self._is_missing(row.get(self.content_col)):
                continue
            if self.queries_col is None:
                continue

            labels = self.normalize_queries(row.get(self.queries_col))
            for label in labels:
                if isinstance(label, str) and label.strip():
                    yield label

    @staticmethod
    def extract_queries(documents: Iterable[dict[str, Any]]) -> list[str]:
        queries: list[str] = []
        for doc in documents:
            labels = doc.get("meta", {}).get("labels", [])
            if isinstance(labels, str):
                labels = [labels]
            if isinstance(labels, Iterable):
                queries.extend([q for q in labels if isinstance(q, str) and q.strip()])
        return queries
