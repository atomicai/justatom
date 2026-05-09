from __future__ import annotations

import json
import os
import sys
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


def main() -> None:
    dataset_path = REPO_ROOT / ".data" / "polaroids.ai.data.json"

    print("=== JSON lazy check ===")
    print(f"dataset_ref=justatom")
    print(f"dataset_path={dataset_path}")

    before_source = rss_mb()
    source = DatasetApi.named("justatom").iterator(lazy=True)
    after_source = rss_mb()

    print(f"source_type={type(source).__name__}")
    print(f"is_polars_lazyframe={isinstance(source, pl.LazyFrame)}")
    print(f"is_polars_dataframe={isinstance(source, pl.DataFrame)}")
    print(f"rss_delta_on_iterator_call_mb={after_source - before_source:.2f}")

    adapter = DatasetRecordAdapter.from_source(
        "justatom",
        lazy=True,
        content_col="content",
        queries_col="queries",
        chunk_id_col="chunk_id",
    )
    first_doc = next(adapter.iterator())

    print(f"adapter_records_type={type(adapter.records).__name__}")
    print(f"first_doc_content_len={len(first_doc.get('content', ''))}")
    print(
        "verdict=lazy=True does not save memory for .json here; backend already returned an eager DataFrame"
    )


if __name__ == "__main__":
    main()
