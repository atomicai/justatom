from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from justatom.tooling.ir_dataset.dense import DenseIndex
from justatom.tooling.ir_dataset.generation_context import GenerationContextConfig, build_generation_context
from justatom.tooling.ir_dataset.sparse import BM25Index
from justatom.tooling.ir_dataset.targets import TargetSelectionConfig, select_target_slots


class ContextEncoder:
    dimension = 2
    model_name = "test/context-encoder"
    model_revision = "test-commit"
    device = "cpu"

    def encode(self, texts, batch_size):
        return np.asarray(
            [
                [1.0, 0.0] if "anchor" in str(text).casefold() or "competitor" in str(text).casefold() else [0.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )


def test_context_excludes_target_and_prioritizes_adjacent_sibling(tmp_path: Path):
    passages = pl.DataFrame(
        [
            {
                "passage_id": "target",
                "article_id": "article-a",
                "content": "anchor explanation",
                "serialized_passage": "passage: Anchor\n\nanchor explanation",
                "start_unit": 3,
                "end_unit": 5,
            },
            {
                "passage_id": "sibling",
                "article_id": "article-a",
                "content": "adjacent implementation detail",
                "serialized_passage": "passage: Anchor\n\nadjacent implementation detail",
                "start_unit": 6,
                "end_unit": 8,
            },
            {
                "passage_id": "competitor",
                "article_id": "article-b",
                "content": "competitor anchor explanation",
                "serialized_passage": "passage: Competitor\n\ncompetitor anchor explanation",
                "start_unit": 0,
                "end_unit": 1,
            },
            {
                "passage_id": "remaining",
                "article_id": "article-c",
                "content": "secondary competitor",
                "serialized_passage": "passage: Secondary\n\nsecondary competitor",
                "start_unit": 0,
                "end_unit": 1,
            },
        ]
    )
    targets = passages.filter(pl.col("passage_id") == "target").with_columns(
        pl.lit("train").alias("split"),
        pl.lit("how_to").alias("requested_intent"),
        pl.lit(0).alias("slot_index"),
    )
    rows = list(zip(passages["passage_id"], passages["serialized_passage"], strict=True))
    bm25 = BM25Index.build(rows, tmp_path / "bm25")
    dense = DenseIndex.build(rows, tmp_path / "dense", ContextEncoder())

    context = build_generation_context(
        targets,
        passages,
        bm25,
        dense,
        GenerationContextConfig(output_dir=tmp_path, dense_block_size=2),
    )

    assert context["candidate_passage_id"].to_list() == ["sibling", "competitor", "remaining"]
    assert context.height == 3
    assert context["target_passage_id"].unique().to_list() == ["target"]
    assert context["context_index"].to_list() == [0, 1, 2]
    assert context["candidate_passage_id"].n_unique() == 3
    assert "target" not in context["candidate_passage_id"].to_list()
    assert context[0, "same_article"]
    assert context[0, "adjacent"]
    assert context[0, "selection_source"] == "adjacent_sibling"
    assert context[1, "selection_source"] == "strongest_non_sibling"
    assert context[0, "structural_rank"] == 1
    assert context[1, "bm25_rank"] is not None
    assert context[1, "dense_rank"] is not None
    assert set(context[1, "source_labels"]) == {"bm25", "dense"}
    assert (tmp_path / "generation_context.parquet").exists()
    assert pl.read_parquet(tmp_path / "generation_context.parquet").equals(context)


def test_context_processes_target_subsets_without_excluding_other_target_slots(tmp_path: Path):
    passages = pl.DataFrame(
        [
            {
                "passage_id": "target-a",
                "article_id": "article-a",
                "content": "anchor one",
                "serialized_passage": "passage: Anchor\n\nanchor one",
                "start_unit": 3,
                "end_unit": 5,
            },
            {
                "passage_id": "target-b",
                "article_id": "article-a",
                "content": "anchor two",
                "serialized_passage": "passage: Anchor\n\nanchor two",
                "start_unit": 6,
                "end_unit": 8,
            },
            {
                "passage_id": "competitor",
                "article_id": "article-b",
                "content": "competitor anchor",
                "serialized_passage": "passage: Competitor\n\ncompetitor anchor",
                "start_unit": 0,
                "end_unit": 1,
            },
            {
                "passage_id": "remaining",
                "article_id": "article-c",
                "content": "remaining explanation",
                "serialized_passage": "passage: Remaining\n\nremaining explanation",
                "start_unit": 0,
                "end_unit": 1,
            },
        ]
    )
    targets = passages.filter(pl.col("passage_id").is_in(["target-a", "target-b"]))
    rows = list(zip(passages["passage_id"], passages["serialized_passage"], strict=True))
    bm25 = BM25Index.build(rows, tmp_path / "bm25")
    dense = DenseIndex.build(rows, tmp_path / "dense", ContextEncoder())

    context = build_generation_context(
        targets,
        passages,
        bm25,
        dense,
        GenerationContextConfig(dense_block_size=2),
    )

    assert context.filter(pl.col("target_passage_id") == "target-a")["candidate_passage_id"].to_list()[0] == "target-b"
    assert context.filter(pl.col("target_passage_id") == "target-b")["candidate_passage_id"].to_list()[0] == "target-a"


def test_context_fails_closed_when_a_duplicate_content_target_reaches_it(tmp_path: Path):
    passages = pl.DataFrame(
        [
            {
                "passage_id": "target",
                "article_id": "article-a",
                "content": "duplicate target content",
                "serialized_passage": "passage: Target\n\nduplicate target content",
                "start_unit": 0,
                "end_unit": 0,
            },
            {
                "passage_id": "collision",
                "article_id": "article-b",
                "content": "  duplicate   target content  ",
                "serialized_passage": "passage: Collision\n\nduplicate target content",
                "start_unit": 0,
                "end_unit": 0,
            },
        ]
    )
    targets = passages.filter(pl.col("passage_id") == "target")
    rows = list(zip(passages["passage_id"], passages["serialized_passage"], strict=True))
    bm25 = BM25Index.build(rows, tmp_path / "bm25")
    dense = DenseIndex.build(rows, tmp_path / "dense", ContextEncoder())

    with pytest.raises(ValueError, match="duplicate normalized content"):
        build_generation_context(targets, passages, bm25, dense)


def test_context_accepts_slots_selected_with_noneligible_adjacent_siblings(tmp_path: Path):
    passage_rows = []
    for article_id, flow in (("article-a", "develop"), ("article-b", "admin")):
        passage_rows.extend(
            [
                {
                    "passage_id": f"{article_id}-target-1",
                    "article_id": article_id,
                    "content": f"anchor explanation for {article_id} first target " * 20,
                    "serialized_passage": f"passage: {article_id}\n\nanchor explanation first target",
                    "token_count": 120,
                    "flows": [flow],
                    "hubs": [f"{flow}-hub"],
                    "start_unit": 0,
                    "end_unit": 0,
                },
                {
                    "passage_id": f"{article_id}-sibling-1",
                    "article_id": article_id,
                    "content": "x = 1",
                    "serialized_passage": f"passage: {article_id}\n\nx = 1",
                    "token_count": 3,
                    "flows": [flow],
                    "hubs": [f"{flow}-hub"],
                    "start_unit": 1,
                    "end_unit": 1,
                },
                {
                    "passage_id": f"{article_id}-sibling-2",
                    "article_id": article_id,
                    "content": "x = 2",
                    "serialized_passage": f"passage: {article_id}\n\nx = 2",
                    "token_count": 3,
                    "flows": [flow],
                    "hubs": [f"{flow}-hub"],
                    "start_unit": 3,
                    "end_unit": 3,
                },
                {
                    "passage_id": f"{article_id}-target-2",
                    "article_id": article_id,
                    "content": f"anchor explanation for {article_id} second target " * 20,
                    "serialized_passage": f"passage: {article_id}\n\nanchor explanation second target",
                    "token_count": 120,
                    "flows": [flow],
                    "hubs": [f"{flow}-hub"],
                    "start_unit": 4,
                    "end_unit": 4,
                },
            ]
        )
    passages = pl.DataFrame(passage_rows)
    targets = select_target_slots(passages, TargetSelectionConfig(article_count=2))
    rows = list(zip(passages["passage_id"], passages["serialized_passage"], strict=True))
    bm25 = BM25Index.build(rows, tmp_path / "bm25")
    dense = DenseIndex.build(rows, tmp_path / "dense", ContextEncoder())

    context = build_generation_context(
        targets,
        passages,
        bm25,
        dense,
        GenerationContextConfig(dense_block_size=2),
    )

    assert targets.height == 4
    assert context.height == 12
    assert context.group_by("target_passage_id").len()["len"].min() == 3
    assert context.filter(pl.col("context_index") == 0)["same_article"].all()
    assert context.filter(pl.col("context_index") == 0)["adjacent"].all()
