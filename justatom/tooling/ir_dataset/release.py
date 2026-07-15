from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl


PAIR_SCHEMA = {
    "pair_id": pl.String,
    "query_id": pl.String,
    "article_id": pl.String,
    "positive_passage_id": pl.String,
    "split": pl.String,
    "query": pl.String,
    "answer": pl.String,
    "evidence": pl.String,
    "positive_passage": pl.String,
    "requested_intent": pl.String,
    "actual_intent": pl.String,
    "title": pl.String,
    "section": pl.String,
    "url": pl.String,
    "topic_flows": pl.List(pl.String),
    "topic_hubs": pl.List(pl.String),
    "tags": pl.List(pl.String),
    "generator_model": pl.String,
    "generator_prompt_hash": pl.String,
    "generation_attempt": pl.Int64,
    "generation_custom_id": pl.String,
    "generation_batch_id": pl.String,
}

QREL_SCHEMA = {
    "query_id": pl.String,
    "passage_id": pl.String,
    "relevance": pl.Int64,
}

CORPUS_COLUMNS = (
    "passage_id",
    "article_id",
    "title",
    "section",
    "content",
    "serialized_passage",
    "url",
    "flows",
    "hubs",
    "tags",
    "char_count",
    "token_count",
    "corpus_rank",
    "source_hash",
)

SPLIT_NAMES = {"train": "train", "dev": "validation", "test": "test"}


@dataclass(frozen=True, slots=True)
class GenerationBinding:
    custom_id: str
    passage_id: str
    article_id: str
    source_hash: str
    prompt_hash: str
    generation_attempt: int
    batch_id: str


@dataclass(frozen=True, slots=True)
class ReleaseFrames:
    pairs: pl.DataFrame
    corpus: pl.DataFrame
    qrels: pl.DataFrame


def stable_query_id(custom_id: str) -> str:
    if not isinstance(custom_id, str) or not custom_id:
        raise ValueError("generation custom_id must be a non-empty string")
    digest = hashlib.sha256(f"habr-ir-query-v1\0{custom_id}".encode("utf-8")).hexdigest()
    return f"q-{digest}"


def stable_pair_id(query_id: str, positive_passage_id: str) -> str:
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("query_id must be a non-empty string")
    if not isinstance(positive_passage_id, str) or not positive_passage_id:
        raise ValueError("positive_passage_id must be a non-empty string")
    payload = f"habr-ir-pair-v1\0{query_id}\0{positive_passage_id}"
    return f"pair-{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _required_columns(frame: pl.DataFrame, required: Sequence[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _unique_rows(frame: pl.DataFrame, key: str, label: str) -> dict[str, dict[str, Any]]:
    _required_columns(frame, (key,), label)
    rows: dict[str, dict[str, Any]] = {}
    for row in frame.iter_rows(named=True):
        value = str(row[key])
        if not value:
            raise ValueError(f"{label} contains an empty {key}")
        if value in rows:
            raise ValueError(f"{label} contains duplicate {key}: {value}")
        rows[value] = row
    return rows


def _normalized_query(value: str) -> str:
    return " ".join(value.split()).casefold()


def _required_output_string(output: Mapping[str, Any], key: str, custom_id: str) -> str:
    value = output.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"accepted generation {custom_id} has empty {key}")
    return value


def _validate_evidence(target: Mapping[str, Any], evidence: str, custom_id: str) -> None:
    content = str(target.get("content", ""))
    positive = str(target.get("serialized_passage", ""))
    if evidence not in content or evidence not in positive:
        raise ValueError(f"accepted generation {custom_id} evidence is not an exact positive substring")
    overlap_chars = int(target.get("overlap_prefix_chars") or 0)
    if overlap_chars > 0 and evidence in content[:overlap_chars] and evidence not in content[overlap_chars:]:
        raise ValueError(f"accepted generation {custom_id} evidence exists only in the overlap prefix")


def materialize_release_frames(
    *,
    records: Sequence[Mapping[str, Any]],
    targets: pl.DataFrame,
    corpus: pl.DataFrame,
    bindings: Mapping[str, GenerationBinding],
    generator_model: str,
) -> ReleaseFrames:
    """Validate immutable bindings and build deterministic release frames."""
    if not isinstance(generator_model, str) or not generator_model:
        raise ValueError("generator_model must be a non-empty string")
    _required_columns(
        targets,
        ("passage_id", "article_id", "source_hash", "serialized_passage", "content", "split", "requested_intent"),
        "targets",
    )
    _required_columns(corpus, CORPUS_COLUMNS, "corpus")
    target_rows = _unique_rows(targets, "passage_id", "targets")
    corpus_rows = _unique_rows(corpus, "passage_id", "corpus")

    record_rows: dict[str, Mapping[str, Any]] = {}
    for record in records:
        custom_id = record.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise ValueError("collected record is missing a non-empty custom_id")
        if custom_id in record_rows:
            raise ValueError(f"collected records contain duplicate custom_id: {custom_id}")
        if record.get("status") not in {"accepted", "rejected"}:
            raise ValueError(f"collected record {custom_id} has invalid status")
        record_rows[custom_id] = record
    if set(record_rows) != set(bindings):
        raise ValueError("collected record IDs do not exactly match immutable generation bindings")

    pair_rows: list[dict[str, Any]] = []
    qrel_rows: list[dict[str, Any]] = []
    normalized_queries: set[str] = set()
    article_splits: dict[str, str] = {}
    positive_ids: set[str] = set()

    for custom_id in sorted(record_rows):
        record = record_rows[custom_id]
        binding = bindings[custom_id]
        if binding.custom_id != custom_id:
            raise ValueError(f"generation binding key mismatch for {custom_id}")
        if binding.generation_attempt <= 0 or not binding.batch_id:
            raise ValueError(f"generation binding {custom_id} has invalid attempt or batch ID")
        target = target_rows.get(binding.passage_id)
        source = corpus_rows.get(binding.passage_id)
        if target is None or source is None:
            raise ValueError(f"generation binding {custom_id} references a missing positive passage")
        for label, row in (("target", target), ("corpus", source)):
            if str(row.get("article_id")) != binding.article_id:
                raise ValueError(f"generation binding {custom_id} {label} article identity mismatch")
            if str(row.get("source_hash")) != binding.source_hash:
                raise ValueError(f"generation binding {custom_id} {label} source identity mismatch")
        if str(target.get("serialized_passage")) != str(source.get("serialized_passage")):
            raise ValueError(f"generation binding {custom_id} target/corpus positive mismatch")
        if record["status"] == "rejected":
            continue

        output = record.get("output")
        if not isinstance(output, Mapping):
            raise ValueError(f"accepted generation {custom_id} is missing structured output")
        query = _required_output_string(output, "query", custom_id)
        answer = _required_output_string(output, "answer", custom_id)
        evidence = _required_output_string(output, "evidence", custom_id)
        requested_intent = _required_output_string(output, "requested_intent", custom_id)
        actual_intent = _required_output_string(output, "actual_intent", custom_id)
        if requested_intent != str(target["requested_intent"]):
            raise ValueError(f"accepted generation {custom_id} requested intent mismatch")
        normalized_query = _normalized_query(query)
        if normalized_query in normalized_queries:
            raise ValueError(f"accepted generation {custom_id} duplicates a normalized query")
        normalized_queries.add(normalized_query)
        _validate_evidence(target, evidence, custom_id)

        target_split = str(target.get("split"))
        if target_split not in SPLIT_NAMES:
            raise ValueError(f"accepted generation {custom_id} has invalid target split: {target_split}")
        split = SPLIT_NAMES[target_split]
        article_id = binding.article_id
        previous_split = article_splits.setdefault(article_id, split)
        if previous_split != split:
            raise ValueError(f"article {article_id} appears in multiple release splits")

        query_id = stable_query_id(custom_id)
        pair_id = stable_pair_id(query_id, binding.passage_id)
        positive_ids.add(binding.passage_id)
        pair_rows.append(
            {
                "pair_id": pair_id,
                "query_id": query_id,
                "article_id": article_id,
                "positive_passage_id": binding.passage_id,
                "split": split,
                "query": query,
                "answer": answer,
                "evidence": evidence,
                "positive_passage": str(source["serialized_passage"]),
                "requested_intent": requested_intent,
                "actual_intent": actual_intent,
                "title": str(source.get("title") or ""),
                "section": str(source.get("section") or ""),
                "url": str(source.get("url") or ""),
                "topic_flows": list(source.get("flows") or []),
                "topic_hubs": list(source.get("hubs") or []),
                "tags": list(source.get("tags") or []),
                "generator_model": generator_model,
                "generator_prompt_hash": binding.prompt_hash,
                "generation_attempt": int(binding.generation_attempt),
                "generation_custom_id": custom_id,
                "generation_batch_id": binding.batch_id,
            }
        )
        qrel_rows.append({"query_id": query_id, "passage_id": binding.passage_id, "relevance": 1})

    pairs = pl.DataFrame(pair_rows, schema=PAIR_SCHEMA)
    if pairs.height:
        pairs = pairs.sort(["split", "query_id"])
    qrels = pl.DataFrame(qrel_rows, schema=QREL_SCHEMA)
    if qrels.height:
        qrels = qrels.sort("query_id")
    release_corpus = corpus.select(CORPUS_COLUMNS).with_columns(
        pl.col("passage_id").is_in(sorted(positive_ids)).alias("is_positive")
    )
    release_corpus = release_corpus.sort("corpus_rank")
    return ReleaseFrames(pairs=pairs, corpus=release_corpus, qrels=qrels)


__all__ = [
    "GenerationBinding",
    "ReleaseFrames",
    "materialize_release_frames",
    "stable_pair_id",
    "stable_query_id",
]
