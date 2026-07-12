from __future__ import annotations

from pathlib import Path

import polars as pl

from justatom.api.ir_dataset import inspect_stage, load_ir_dataset_config, parse_cli
from justatom.tooling.ir_dataset.dense import DenseSearchHit
from justatom.tooling.ir_dataset.neighbors import merge_neighbors
from justatom.tooling.ir_dataset.sparse import SearchHit


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "datasets" / "habr-ir.yaml"


def test_checked_in_config_resolves_local_defaults():
    config = load_ir_dataset_config(CONFIG_PATH)

    assert config.source.repo_id == "justatom/habr-ds"
    assert config.chunking.accepted_max_tokens == 504
    assert config.preparation.max_passages == 100_000
    assert config.retrieval.bm25_k == 20
    assert config.retrieval.dense_k == 20


def test_dotted_cli_overrides_are_typed(tmp_path):
    parsed = parse_cli(
        [
            "--config",
            str(CONFIG_PATH),
            "--preparation.max_articles",
            "100",
            "--retrieval.query_passages=25",
            "--output.root",
            str(tmp_path),
            "run",
        ]
    )

    assert parsed.stage == "run"
    assert parsed.config.preparation.max_articles == 100
    assert parsed.config.retrieval.query_passages == 25
    assert parsed.config.output.root == tmp_path


def test_rrf_union_excludes_self_and_retains_source_ranks():
    bm25 = [
        SearchHit("p1", 3.0, 1),
        SearchHit("p2", 2.0, 2),
        SearchHit("p3", 1.0, 3),
    ]
    dense = [
        DenseSearchHit("p1", 1.0, 1),
        DenseSearchHit("p3", 0.9, 2),
        DenseSearchHit("p4", 0.8, 3),
    ]

    rows = merge_neighbors("p1", bm25, dense, rrf_k=60, limit=3)

    assert [row.candidate_id for row in rows] == ["p3", "p2", "p4"]
    assert all(row.candidate_id != "p1" for row in rows)
    assert rows[0].bm25_rank == 3
    assert rows[0].dense_rank == 2
    assert rows[0].bm25_score == 1.0
    assert rows[0].dense_score == 0.9


def test_rrf_ties_are_stable_by_candidate_id():
    rows = merge_neighbors(
        "query",
        [SearchHit("b", 1.0, 1)],
        [DenseSearchHit("a", 1.0, 1)],
        rrf_k=60,
        limit=2,
    )

    assert [row.candidate_id for row in rows] == ["a", "b"]


def test_inspect_can_select_one_query_passage(tmp_path):
    config = load_ir_dataset_config(
        CONFIG_PATH,
        overrides={"output": {"root": str(tmp_path)}},
    )
    pl.DataFrame(
        [
            {"query_id": "q1", "candidate_id": "a", "rrf_score": 0.1},
            {"query_id": "q2", "candidate_id": "b", "rrf_score": 0.2},
            {"query_id": "q2", "candidate_id": "c", "rrf_score": 0.1},
        ]
    ).write_parquet(tmp_path / "neighbors.parquet")

    rows = inspect_stage(config, sample=10, passage_id="q2")

    assert [row["candidate_id"] for row in rows] == ["b", "c"]
