from __future__ import annotations

import polars as pl

from justatom.tooling.ir_dataset.release import (
    GenerationBinding,
    materialize_release_frames,
    stable_pair_id,
    stable_query_id,
)


def corpus() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "corpus_rank": 0,
                "passage_id": "p1",
                "article_id": "a1",
                "title": "Redis",
                "section": "Retry",
                "content": "Redis повторяет запрос после сетевой ошибки.",
                "serialized_passage": "passage: Redis\nRetry\n\nRedis повторяет запрос после сетевой ошибки.",
                "url": "https://example.test/redis",
                "flows": ["develop"],
                "hubs": ["redis"],
                "tags": ["redis", "retry"],
                "char_count": 43,
                "token_count": 8,
                "source_hash": "source-1",
            },
            {
                "corpus_rank": 1,
                "passage_id": "p2",
                "article_id": "a2",
                "title": "PostgreSQL",
                "section": "WAL",
                "content": "PostgreSQL хранит журнал WAL.",
                "serialized_passage": "passage: PostgreSQL\nWAL\n\nPostgreSQL хранит журнал WAL.",
                "url": "https://example.test/postgresql",
                "flows": ["develop"],
                "hubs": ["postgresql"],
                "tags": ["wal"],
                "char_count": 31,
                "token_count": 6,
                "source_hash": "source-2",
            },
        ]
    )


def targets() -> pl.DataFrame:
    return corpus().with_columns(
        pl.Series("split", ["train", "test"]),
        pl.Series("requested_intent", ["how_to", "concept"]),
    )


def records() -> list[dict[str, object]]:
    return [
        {
            "custom_id": "gen-accepted",
            "status": "accepted",
            "reason_codes": [],
            "output": {
                "query": "Как Redis обрабатывает запрос после сетевой ошибки?",
                "answer": "Redis повторяет запрос.",
                "evidence": "Redis повторяет запрос после сетевой ошибки.",
                "requested_intent": "how_to",
                "actual_intent": "how_to",
            },
        },
        {
            "custom_id": "gen-rejected",
            "status": "rejected",
            "reason_codes": ["usable_false"],
            "output": {
                "query": "",
                "answer": "",
                "evidence": "",
                "requested_intent": "concept",
                "actual_intent": "concept",
            },
        },
    ]


def bindings() -> dict[str, GenerationBinding]:
    return {
        "gen-accepted": GenerationBinding(
            custom_id="gen-accepted",
            passage_id="p1",
            article_id="a1",
            source_hash="source-1",
            prompt_hash="prompt-1",
            generation_attempt=1,
            batch_id="batch-1",
        ),
        "gen-rejected": GenerationBinding(
            custom_id="gen-rejected",
            passage_id="p2",
            article_id="a2",
            source_hash="source-2",
            prompt_hash="prompt-2",
            generation_attempt=1,
            batch_id="batch-1",
        ),
    }


def test_release_ids_are_domain_separated_and_stable():
    query_id = stable_query_id("gen-accepted")
    pair_id = stable_pair_id(query_id, "p1")

    assert query_id.startswith("q-") and len(query_id) == 66
    assert pair_id.startswith("pair-") and len(pair_id) == 69
    assert query_id == stable_query_id("gen-accepted")
    assert pair_id == stable_pair_id(query_id, "p1")


def test_materialization_keeps_only_accepted_rows_and_full_positive():
    result = materialize_release_frames(
        records=records(),
        targets=targets(),
        corpus=corpus(),
        bindings=bindings(),
        generator_model="test-model",
    )

    assert result.pairs.height == 1
    assert result.qrels.height == 1
    assert result.corpus.height == 2
    pair = result.pairs.row(0, named=True)
    assert pair["positive_passage_id"] == "p1"
    assert pair["positive_passage"] == corpus().row(0, named=True)["serialized_passage"]
    assert pair["evidence"] in pair["positive_passage"]
    assert result.qrels.row(0, named=True) == {
        "query_id": pair["query_id"],
        "passage_id": "p1",
        "relevance": 1,
    }
    assert result.corpus.filter(pl.col("passage_id") == "p1")["is_positive"].item() is True
    assert result.corpus.filter(pl.col("passage_id") == "p2")["is_positive"].item() is False
