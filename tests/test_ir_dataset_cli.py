from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from justatom.api import ir_dataset as ir_dataset_module
from justatom.api.ir_dataset import (
    _embed_fingerprint,
    embed_stage,
    inspect_stage,
    load_ir_dataset_config,
    parse_cli,
    prepare_generation_stage,
)
from justatom.tooling.ir_dataset.batch import REQUIRED_SOURCE_CORPUS_FINGERPRINT
from justatom.tooling.ir_dataset.artifacts import PrepareSummary
from justatom.tooling.ir_dataset.chunking import CHUNKER_VERSION
from justatom.tooling.ir_dataset.dense import DenseIndex, DenseSearchHit
from justatom.tooling.ir_dataset.neighbors import include_structural_neighbors, merge_neighbors, select_query_passages
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
    assert config.chunking.tokenizer_revision == "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    assert config.chunking.accepted_max_tokens == 504
    assert config.preparation.max_passages == 100_000
    assert config.retrieval.bm25_k == 20
    assert config.retrieval.dense_k == 20
    assert config.generation.max_shard_bytes == 100_000_000
    assert config.generation.scale_authorized is False
    assert config.generation.max_batch_attempts == 2
    assert config.target_selection.article_count == 50
    assert config.output.generation_root == Path(".tmp_runs/datasets/habr-ir/generation-v1")
    assert config.output.pilot_generation_root == config.output.generation_root


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


def test_cli_accepts_resumable_generation_stages(tmp_path):
    stages = (
        "select-targets",
        "prepare-generation",
        "submit-generation",
        "generation-status",
        "retry-generation",
        "collect-generation",
    )

    parsed = [parse_cli(["--config", str(CONFIG_PATH), "--output.root", str(tmp_path), stage]) for stage in stages]

    assert [item.stage for item in parsed] == list(stages)
    assert all(item.config.generation.model == "gpt-5.6-terra" for item in parsed)


def test_source_corpus_authorization_checks_the_actual_passage_sha256(tmp_path):
    passages_path = tmp_path / "passages.parquet"
    passages_path.write_bytes(b"actual corpus bytes")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "fingerprint": REQUIRED_SOURCE_CORPUS_FINGERPRINT,
                "chunker_version": CHUNKER_VERSION,
                "passages_sha256": hashlib.sha256(b"different corpus bytes").hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="passages.parquet SHA-256"):
        ir_dataset_module._source_corpus_manifest_fingerprint(tmp_path)


def test_generation_rejects_a_source_corpus_from_an_old_chunker_contract(tmp_path):
    passages_path = tmp_path / "passages.parquet"
    passages_path.write_bytes(b"old chunker corpus")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "fingerprint": REQUIRED_SOURCE_CORPUS_FINGERPRINT,
                "chunker_version": CHUNKER_VERSION - 1,
                "passages_sha256": hashlib.sha256(b"old chunker corpus").hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="chunker contract"):
        ir_dataset_module._source_corpus_manifest_fingerprint(tmp_path)


def test_prepare_generation_reads_existing_generation_workspace_artifacts(tmp_path, monkeypatch):
    source_root = tmp_path / "local-100k"
    generation_root = tmp_path / "generation-v1"
    source_root.mkdir()
    generation_root.mkdir()
    pl.DataFrame({"passage_id": ["source-passage"]}).write_parquet(source_root / "passages.parquet")
    passages_sha256 = ir_dataset_module.sha256_file(source_root / "passages.parquet")
    (source_root / "manifest.json").write_text(
        json.dumps(
            {
                "fingerprint": REQUIRED_SOURCE_CORPUS_FINGERPRINT,
                "chunker_version": CHUNKER_VERSION,
                "passages_sha256": passages_sha256,
            }
        ),
        encoding="utf-8",
    )
    ir_dataset_module.write_bound_parquet_artifact(
        pl.DataFrame({"passage_id": ["pilot-target"]}),
        generation_root / "targets.parquet",
        generation_root / "targets_state.json",
        artifact_kind="targets",
        source_corpus_fingerprint=REQUIRED_SOURCE_CORPUS_FINGERPRINT,
        passages_sha256=passages_sha256,
        config={"article_count": 50, "seed": 42, "output_dir": None, "max_flow_share": 0.30},
        upstream_sha256=ir_dataset_module.sha256_file(source_root / "manifest.json"),
    )
    ir_dataset_module.write_bound_parquet_artifact(
        pl.DataFrame({"target_passage_id": ["pilot-target"], "context_index": [0]}),
        generation_root / "generation_context.parquet",
        generation_root / "generation_context_state.json",
        artifact_kind="generation_context",
        source_corpus_fingerprint=REQUIRED_SOURCE_CORPUS_FINGERPRINT,
        passages_sha256=passages_sha256,
        config={
            "bm25_k": 20,
            "dense_k": 20,
            "union_k": 30,
            "rrf_k": 60,
            "dense_block_size": 65536,
            "device": "mps",
            "output_dir": None,
        },
        upstream_sha256=ir_dataset_module.sha256_file(generation_root / "targets.parquet"),
    )
    captured = {}

    def prepare(targets, generation_context, config, output_dir, *, source_corpus_fingerprint, **bindings):
        captured.update(
            targets=targets.to_dicts(),
            context=generation_context.to_dicts(),
            output_dir=output_dir,
            source_corpus_fingerprint=source_corpus_fingerprint,
            bindings=bindings,
        )
        return {"prepared": 1}

    monkeypatch.setattr(ir_dataset_module, "prepare_generation_batches", prepare)
    config = load_ir_dataset_config(
        CONFIG_PATH,
        overrides={"output": {"root": str(source_root), "generation_root": str(generation_root)}},
    )

    assert prepare_generation_stage(config) == {"prepared": 1}
    assert captured["targets"] == [{"passage_id": "pilot-target"}]
    assert captured["context"] == [{"target_passage_id": "pilot-target", "context_index": 0}]
    assert captured["output_dir"] == generation_root
    assert captured["source_corpus_fingerprint"] == REQUIRED_SOURCE_CORPUS_FINGERPRINT


def test_embed_fingerprint_changes_with_model_revision():
    config = load_ir_dataset_config(CONFIG_PATH)
    changed = replace(config, retrieval=replace(config.retrieval, model_revision="different-commit"))

    assert _embed_fingerprint(config, "prepare-v1") != _embed_fingerprint(changed, "prepare-v1")


def test_embed_fingerprint_includes_bm25_tokenizer_contract(monkeypatch):
    config = load_ir_dataset_config(CONFIG_PATH)
    original = _embed_fingerprint(config, "prepare-v1")
    monkeypatch.setattr(ir_dataset_module, "TECHNICAL_TOKEN_PATTERN", "changed-pattern")

    assert _embed_fingerprint(config, "prepare-v1") != original


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


def test_query_sampler_requires_an_adjacent_corpus_sibling():
    frame = pl.DataFrame(
        [
            {"corpus_rank": 0, "passage_id": "single", "article_id": "a", "start_unit": 0, "end_unit": 1},
            {"corpus_rank": 1, "passage_id": "far-1", "article_id": "b", "start_unit": 0, "end_unit": 1},
            {"corpus_rank": 2, "passage_id": "far-2", "article_id": "b", "start_unit": 10, "end_unit": 11},
            {"corpus_rank": 3, "passage_id": "first", "article_id": "c", "start_unit": 3, "end_unit": 5},
            {"corpus_rank": 4, "passage_id": "second", "article_id": "c", "start_unit": 6, "end_unit": 8},
        ]
    )

    selected = select_query_passages(frame, count=2)

    assert selected["passage_id"].to_list() == ["first", "second"]


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
