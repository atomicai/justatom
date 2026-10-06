from __future__ import annotations

import builtins
import copy
import json
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from justatom.agentic import benchmark_data
from justatom.agentic.benchmark_data import BenchmarkCase, load_cases


def _write_jsonl(path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _valid_row(query_id: str = "q-1") -> dict[str, object]:
    return {
        "query_id": query_id,
        "query": f"Question for {query_id}",
        "chunk_ids": [f"chunk-{query_id}"],
        "answer": "gold answer",
        "language": "en",
        "domain": "science",
        "universe": "demo",
        "depth": 1,
    }


def test_load_cases_preserves_identity_order_and_exposes_only_safe_metadata(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    _write_jsonl(
        path,
        [
            {
                "query_id": "  custom-id  ",
                "query": "  What is alpha?  ",
                "chunk_ids": [" chunk-2 ", "chunk-1"],
                "answer": "private gold answer",
                "language": "en",
                "domain": "science",
                "universe": "demo",
                "depth": 2,
                "secret": "must not escape",
            },
            {
                "query_id": "second",
                "query": "Second question",
                "chunk_ids": ["chunk-3"],
            },
        ],
    )

    cases = load_cases(path)

    assert cases == [
        BenchmarkCase(
            query_id="  custom-id  ",
            query="  What is alpha?  ",
            chunk_ids=(" chunk-2 ", "chunk-1"),
            metadata={
                "language": "en",
                "domain": "science",
                "universe": "demo",
                "depth": 2,
            },
        ),
        BenchmarkCase(
            query_id="second",
            query="Second question",
            chunk_ids=("chunk-3",),
            metadata={},
        ),
    ]
    assert [field.name for field in fields(BenchmarkCase)] == [
        "query_id",
        "query",
        "chunk_ids",
        "metadata",
    ]
    with pytest.raises(FrozenInstanceError):
        cases[0].query = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("replacement", "field"),
    [
        ({"query_id": " ", "query": "Question", "chunk_ids": ["chunk"]}, "query_id"),
        ({"query_id": "q", "query": "\t", "chunk_ids": ["chunk"]}, "query"),
        ({"query_id": "q", "query": "Question", "chunk_ids": []}, "chunk_ids"),
        ({"query_id": "q", "query": "Question", "chunk_ids": ["chunk", "chunk"]}, "chunk_ids"),
        ({"query_id": "q", "query": "Question", "chunk_ids": ["chunk", 7]}, "chunk_ids"),
        ({"query_id": "q", "query": "Question", "chunk_ids": [" "]}, "chunk_ids"),
        ({"query_id": "q", "query": "Question"}, "chunk_ids"),
    ],
)
def test_jsonl_schema_errors_name_the_physical_line_and_invalid_field(tmp_path, replacement, field):
    path = tmp_path / "benchmark.jsonl"
    _write_jsonl(path, [_valid_row("good"), replacement])

    with pytest.raises(ValueError, match=rf"benchmark\.jsonl:2.*{field}"):
        load_cases(path)


def test_jsonl_parse_errors_name_the_physical_line(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    path.write_text(json.dumps(_valid_row()) + "\n{not-json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"benchmark\.jsonl:2"):
        load_cases(path)


def test_jsonl_reports_the_first_invalid_physical_row(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    path.write_text('{"query_id": "q", "query": " "}\n{not-json}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"benchmark\.jsonl:1.*query"):
        load_cases(path)


def test_non_object_jsonl_row_is_rejected_with_its_line(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    path.write_text(json.dumps(_valid_row()) + "\n[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"benchmark\.jsonl:2.*object"):
        load_cases(path)


def test_duplicate_query_id_reports_the_later_line(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    _write_jsonl(path, [_valid_row("duplicate"), _valid_row("duplicate")])

    with pytest.raises(ValueError, match=r"benchmark\.jsonl:2.*duplicate query_id"):
        load_cases(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("language", 7),
        ("domain", []),
        ("universe", False),
        ("depth", True),
        ("depth", "2"),
        ("depth", 1.5),
    ],
)
def test_optional_metadata_types_are_validated(tmp_path, field, value):
    path = tmp_path / "benchmark.jsonl"
    row = _valid_row()
    row[field] = value
    _write_jsonl(path, [row])

    with pytest.raises(ValueError, match=rf"benchmark\.jsonl:1.*{field}"):
        load_cases(path)


def test_optional_metadata_accepts_and_preserves_nulls(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    row = _valid_row()
    row.update(language=None, domain=None, universe=None, depth=None)
    _write_jsonl(path, [row])

    assert load_cases(path)[0].metadata == {
        "domain": None,
        "language": None,
        "universe": None,
        "depth": None,
    }


def test_optional_metadata_validation_does_not_invent_value_constraints(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    row = _valid_row()
    row.update(language="", domain=" ", universe="custom", depth=0)
    _write_jsonl(path, [row])

    assert load_cases(path)[0].metadata == {
        "domain": " ",
        "language": "",
        "universe": "custom",
        "depth": 0,
    }


@pytest.mark.parametrize("limit", [True, False, 0, -1, 1.5, "1"])
def test_limit_must_be_a_positive_integer(tmp_path, limit):
    path = tmp_path / "benchmark.jsonl"
    _write_jsonl(path, [_valid_row()])

    with pytest.raises((TypeError, ValueError), match="limit"):
        load_cases(path, limit=limit)


def test_limit_preserves_order_but_does_not_hide_malformed_gold(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    bad_row = _valid_row("q-3")
    bad_row["query"] = " "
    _write_jsonl(path, [_valid_row("q-1"), _valid_row("q-2"), bad_row])

    with pytest.raises(ValueError, match=r"benchmark\.jsonl:3.*query"):
        load_cases(path, limit=1)

    bad_row["query"] = "Question for q-3"
    _write_jsonl(path, [_valid_row("q-1"), _valid_row("q-2"), bad_row])
    assert [case.query_id for case in load_cases(path, limit=2)] == ["q-1", "q-2"]


def test_row_conversion_does_not_mutate_input_mapping():
    row = _valid_row()
    original = copy.deepcopy(row)

    benchmark_data._case_from_row(row, location="fixture row 1")

    assert row == original


@pytest.mark.parametrize("contents", ["", "\n  \n"])
def test_empty_jsonl_dataset_is_rejected(tmp_path, contents):
    path = tmp_path / "benchmark.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_cases(path)


def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "benchmark.csv"
    path.write_text("query_id,query,chunk_ids\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"unsupported.*\.csv"):
        load_cases(path)


@pytest.mark.parametrize("url", ["https://example.test/benchmark.jsonl", "hf://private/benchmark.parquet"])
def test_remote_sources_are_rejected_without_access(url):
    with pytest.raises(ValueError, match="local"):
        load_cases(url)


def test_jsonl_loading_does_not_import_optional_pyarrow(tmp_path, monkeypatch):
    path = tmp_path / "benchmark.jsonl"
    _write_jsonl(path, [_valid_row()])
    real_import = builtins.__import__

    def reject_pyarrow(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise AssertionError("JSONL loading attempted to import pyarrow")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_pyarrow)

    assert load_cases(path)[0].query_id == "q-1"


def test_missing_pyarrow_has_actionable_parquet_error(tmp_path, monkeypatch):
    path = tmp_path / "benchmark.parquet"
    path.write_bytes(b"not read because dependency is unavailable")
    real_import = builtins.__import__

    def hide_pyarrow(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ModuleNotFoundError("No module named 'pyarrow'", name="pyarrow")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", hide_pyarrow)

    with pytest.raises(ImportError, match=r"pyarrow.*pip install pyarrow"):
        load_cases(path)


def test_parquet_matches_jsonl_and_preserves_physical_row_order(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = [_valid_row("q-2"), _valid_row("q-1")]
    jsonl_path = tmp_path / "benchmark.jsonl"
    parquet_path = tmp_path / "benchmark.parquet"
    _write_jsonl(jsonl_path, rows)
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)

    assert load_cases(parquet_path) == load_cases(jsonl_path)
    assert [case.query_id for case in load_cases(parquet_path)] == ["q-2", "q-1"]


def test_parquet_schema_error_names_physical_row(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    bad_row = _valid_row("bad")
    bad_row["query"] = " "
    path = Path(tmp_path) / "benchmark.parquet"
    pq.write_table(pa.Table.from_pylist([_valid_row("good"), bad_row]), path)

    with pytest.raises(ValueError, match=r"benchmark\.parquet row 2.*query"):
        load_cases(path)
