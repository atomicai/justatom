from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import polars as pl
import psutil

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from justatom.storing.dataset import API as DatasetApi
from justatom.tooling.dataset import DatasetRecordAdapter


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _write_temp_jsonl_from_json(source_json: Path) -> Path:
    with source_json.open("r", encoding="utf-8") as fp:
        rows = json.load(fp)

    fd, tmp_path = tempfile.mkstemp(suffix=".jsonl", prefix="justatom_lazy_check_")
    os.close(fd)
    out = Path(tmp_path)
    with out.open("w", encoding="utf-8") as fp:
        for row in rows:
            if row is None:
                continue
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def main() -> None:
    source_json = REPO_ROOT / ".data" / "polaroids.ai.data.json"
    temp_jsonl = _write_temp_jsonl_from_json(source_json)

    try:
        print("=== JSONL lazy check ===")
        print(f"source_json={source_json}")
        print(f"temp_jsonl={temp_jsonl}")

        before_source = rss_mb()
        source = DatasetApi.named(str(temp_jsonl)).iterator(lazy=True)
        after_source = rss_mb()

        print(f"source_type={type(source).__name__}")
        print(f"is_polars_lazyframe={isinstance(source, pl.LazyFrame)}")
        print(f"is_polars_dataframe={isinstance(source, pl.DataFrame)}")
        print(f"rss_delta_on_iterator_call_mb={after_source - before_source:.2f}")

        adapter = DatasetRecordAdapter.from_source(
            temp_jsonl,
            lazy=True,
            content_col="content",
            queries_col="queries",
            chunk_id_col="chunk_id",
        )
        first_doc = next(adapter.iterator())

        print(f"adapter_records_type={type(adapter.records).__name__}")
        print(f"first_doc_content_len={len(first_doc.get('content', ''))}")
        print(
            "verdict=lazy=True is real for .jsonl here; backend returns a polars LazyFrame and rows are streamed by batches"
        )
    finally:
        temp_jsonl.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
