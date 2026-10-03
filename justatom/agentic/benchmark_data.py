from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

_METADATA_FIELDS = ("domain", "language", "universe", "depth")


@dataclass(frozen=True)
class BenchmarkCase:
    query_id: str
    query: str
    chunk_ids: tuple[str, ...]
    metadata: dict[str, Any]


def _invalid(location: str, detail: str) -> ValueError:
    return ValueError(f"invalid benchmark row at {location}: {detail}")


def _required_string(row: dict[str, Any], field: str, location: str) -> str:
    if field not in row:
        raise _invalid(location, f"missing {field}")
    value = row[field]
    if not isinstance(value, str) or not value.strip():
        raise _invalid(location, f"{field} must be a non-empty string")
    return value


def _metadata_from_row(row: dict[str, Any], location: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for field in _METADATA_FIELDS:
        if field not in row:
            continue
        value = row[field]
        if value is None:
            metadata[field] = None
            continue
        if field == "depth":
            if isinstance(value, bool) or not isinstance(value, int):
                raise _invalid(location, "depth must be an integer or null")
        elif not isinstance(value, str):
            raise _invalid(location, f"{field} must be a string or null")
        metadata[field] = value
    return metadata


def _case_from_row(row: object, *, location: str) -> BenchmarkCase:
    if not isinstance(row, dict):
        raise _invalid(location, "row must be a JSON object")

    query_id = _required_string(row, "query_id", location)
    query = _required_string(row, "query", location)
    if "chunk_ids" not in row:
        raise _invalid(location, "missing chunk_ids")
    raw_chunk_ids = row["chunk_ids"]
    if not isinstance(raw_chunk_ids, list) or not raw_chunk_ids:
        raise _invalid(location, "chunk_ids must be a non-empty list of unique strings")
    if any(not isinstance(chunk_id, str) or not chunk_id.strip() for chunk_id in raw_chunk_ids):
        raise _invalid(location, "chunk_ids must contain only non-empty strings")
    if len(set(raw_chunk_ids)) != len(raw_chunk_ids):
        raise _invalid(location, "chunk_ids must contain unique strings")

    return BenchmarkCase(
        query_id=query_id,
        query=query,
        chunk_ids=tuple(raw_chunk_ids),
        metadata=_metadata_from_row(row, location),
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _jsonl_rows(path: Path) -> Iterator[tuple[object, str]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                row = json.loads(line, parse_constant=_reject_json_constant)
            except (TypeError, ValueError) as error:
                raise _invalid(location, "malformed JSON") from error
            yield row, location


def _parquet_rows(path: Path) -> Iterator[tuple[object, str]]:
    try:
        from pyarrow import parquet as pq
    except ImportError as error:
        raise ImportError("Parquet benchmark loading requires pyarrow; install it with `pip install pyarrow`.") from error

    for row_number, row in enumerate(pq.read_table(path).to_pylist(), start=1):
        yield row, f"{path} row {row_number}"


def _validate_rows(rows: Iterable[tuple[object, str]]) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    seen_query_ids: set[str] = set()
    for row, location in rows:
        case = _case_from_row(row, location=location)
        if case.query_id in seen_query_ids:
            raise _invalid(location, f"duplicate query_id {case.query_id!r}")
        seen_query_ids.add(case.query_id)
        cases.append(case)
    return cases


def load_cases(path: str | Path, *, limit: int | None = None) -> list[BenchmarkCase]:
    """Load and validate local benchmark cases in their physical row order."""

    if isinstance(limit, bool) or (limit is not None and not isinstance(limit, int)):
        raise TypeError("limit must be a positive integer or None")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer or None")
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or pathlib.Path")
    if isinstance(path, str) and "://" in path:
        raise ValueError("benchmark path must refer to a local file")

    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".jsonl":
        rows = _jsonl_rows(resolved)
    elif suffix == ".parquet":
        rows = _parquet_rows(resolved)
    else:
        raise ValueError(f"unsupported benchmark file extension {resolved.suffix!r}; use .jsonl or .parquet")

    cases = _validate_rows(rows)
    if not cases:
        raise ValueError(f"benchmark dataset is empty: {resolved}")
    return cases if limit is None else cases[:limit]


__all__ = ["BenchmarkCase", "load_cases"]
