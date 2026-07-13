from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from justatom.api.ir_dataset import _embed_fingerprint, embed_stage, inspect_stage, load_ir_dataset_config, parse_cli
from justatom.tooling.ir_dataset.artifacts import PrepareSummary
from justatom.tooling.ir_dataset.dense import DenseIndex, DenseSearchHit
from justatom.tooling.ir_dataset.neighbors import include_structural_neighbors, merge_neighbors
from justatom.tooling.ir_dataset.sparse import BM25Index, SearchHit


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "datasets" / "habr-ir.yaml"


class CLIEncoder:
    dimension = 2
    model_name = "test/encoder"
    model_revision = "test-commit"
    device = "cpu"

    def encode(self, texts, batch_size):
        return np.asarray(
            [[1.0, 0.0] if "docker" in str(text).casefold() else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


def prepared_fixture(tmp_path: Path) -> PrepareSummary:
    path = tmp_path / "passages.parquet"
    pl.DataFrame(
        [
            {
                "corpus_rank": 0,
                "passage_id": "p1",
                "article_id": "a1",
                "title": "Docker",
                "section": "Порты",
                "content": "Docker открывает порт после настройки.",
                "serialized_passage": "passage: Docker\nПорты\n\nDocker открывает порт после настройки.",
                "flows": ["develop"],
                "hubs": ["containers"],
            },
            {
                "corpus_rank": 1,
                "passage_id": "p2",
                "article_id": "a2",
                "title": "PostgreSQL",
                "section": "WAL",
                "content": "PostgreSQL хранит журнал WAL.",
                "serialized_passage": "passage: PostgreSQL\nWAL\n\nPostgreSQL хранит журнал WAL.",
                "flows": ["develop"],
                "hubs": ["databases"],
            },
        ]
    ).write_parquet(path)
    return PrepareSummary(
        passages_path=path,
        manifest_path=tmp_path / "manifest.json",
        article_count=2,
        passage_count=2,
        fingerprint="prepare-v1",
        reused=False,
    )


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


def test_embed_fingerprint_changes_with_model_revision():
    config = load_ir_dataset_config(CONFIG_PATH)
    changed = replace(config, retrieval=replace(config.retrieval, model_revision="different-commit"))

    assert _embed_fingerprint(config, "prepare-v1") != _embed_fingerprint(changed, "prepare-v1")


def test_embed_stage_rebuilds_when_dense_index_digest_changes(tmp_path):
    config = load_ir_dataset_config(
        CONFIG_PATH,
        overrides={
            "output": {"root": str(tmp_path)},
            "retrieval": {"model_name": "test/encoder", "model_revision": "test-commit", "device": "cpu"},
        },
    )
    prepared = prepared_fixture(tmp_path)

    assert embed_stage(config, prepared, encoder=CLIEncoder()) is False
    assert embed_stage(config, prepared, encoder=CLIEncoder()) is True
    with (tmp_path / "dense" / "embeddings.f32").open("r+b") as stream:
        stream.seek(0)
        stream.write(b"broken")

    assert embed_stage(config, prepared, encoder=CLIEncoder()) is False


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


def test_structural_neighbors_are_guaranteed_beyond_rrf_limit():
    ranked = merge_neighbors(
        "query",
        [SearchHit("retrieved", 1.0, 1)],
        [],
        rrf_k=60,
        limit=1,
    )

    rows = include_structural_neighbors(
        ranked,
        [("sibling", True), ("retrieved", False), ("far-sibling", False)],
    )

    assert [row.candidate_id for row in rows] == ["retrieved", "sibling", "far-sibling"]
    assert rows[0].structural_rank == 2
    assert rows[1].structural_rank == 1
    assert rows[1].adjacent is True
    assert len({row.candidate_id for row in rows}) == 3


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


def test_inspect_free_query_searches_bm25_and_dense_indices(tmp_path):
    config = load_ir_dataset_config(
        CONFIG_PATH,
        overrides={"output": {"root": str(tmp_path)}, "retrieval": {"device": "cpu"}},
    )
    prepared = prepared_fixture(tmp_path)
    frame = pl.read_parquet(prepared.passages_path)
    rows = list(zip(frame["passage_id"], frame["serialized_passage"], strict=True))
    bm25 = BM25Index.build(rows, tmp_path / "bm25")
    dense = DenseIndex.build(rows, tmp_path / "dense", CLIEncoder())

    result = inspect_stage(
        config,
        query="почему docker не открывает порт",
        bm25_index=bm25,
        dense_index=dense,
    )

    assert result[0]["candidate_id"] == "p1"
    assert result[0]["bm25_rank"] is not None or result[0]["dense_rank"] is not None
