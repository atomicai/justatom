from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from justatom.tooling.ir_dataset.artifacts import sha256_file, validate_bound_parquet_artifact
from justatom.tooling.ir_dataset.batch import _generation_fingerprint, _target_context_fingerprint
from justatom.tooling.ir_dataset.chunking import CHUNKER_VERSION
from justatom.tooling.ir_dataset.generation import GeneratorConfig


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

RELEASE_ARTIFACT_PATHS = frozenset(
    {
        "README.md",
        "audit/pilot-review.csv",
        "data/corpus-100k/corpus.parquet",
        *(f"data/pairs/{split}.parquet" for split in ("train", "validation", "test")),
        *(f"data/qrels/{split}.parquet" for split in ("train", "validation", "test")),
    }
)

AUDIT_HUMAN_COLUMNS = (
    "human_target_answers_query",
    "human_evidence_supports_answer",
    "human_self_contained",
    "human_single_interpretation",
    "human_no_competitor_answers",
    "human_natural_not_copied",
    "human_correct_intent",
    "human_accept",
    "reviewer",
    "notes",
)

_SPREADSHEET_FORMULA_PREFIXES = ("=", "+", "-", "@")


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


@dataclass(frozen=True, slots=True)
class ReleaseSummary:
    root: Path
    manifest_path: Path
    pair_count: int
    corpus_count: int
    qrel_count: int
    review_count: int
    fingerprint: str
    reused: bool


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
    binding_passage_ids = [binding.passage_id for binding in bindings.values()]
    if len(binding_passage_ids) != len(set(binding_passage_ids)) or set(binding_passage_ids) != set(target_rows):
        raise ValueError("generation bindings must have a one-to-one mapping to targets")

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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}: expected an object")
    return value


def _bound_path(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path is missing")
    root_resolved = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        raise ValueError(f"{label} path escapes its artifact root") from None
    return candidate


def _require_checksum(path: Path, expected: Any, label: str) -> None:
    if not path.exists() or not isinstance(expected, str) or sha256_file(path) != expected:
        raise ValueError(f"{label} checksum mismatch")


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} does not exist: {path}") from None
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid {label} JSON on line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"invalid {label} row on line {line_number}")
        rows.append(row)
    return rows


def _validate_source(source_root: Path) -> tuple[dict[str, Any], Path, str]:
    manifest_path = source_root / "manifest.json"
    manifest = _read_json(manifest_path, "source corpus manifest")
    fingerprint = manifest.get("fingerprint")
    passages_sha256 = manifest.get("passages_sha256")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("source corpus manifest fingerprint is missing")
    if manifest.get("chunker_version") != CHUNKER_VERSION:
        raise ValueError("source corpus manifest does not match the current chunker contract")
    passages_path = source_root / "passages.parquet"
    _require_checksum(passages_path, passages_sha256, "source passages artifact")
    return manifest, passages_path, sha256_file(manifest_path)


def _validate_generation_artifact(
    generation_root: Path,
    descriptor: Any,
    *,
    artifact_kind: str,
    source_fingerprint: str,
    passages_sha256: str,
    upstream_sha256: str,
) -> Path:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"generation state is missing {artifact_kind} descriptor")
    artifact_path = _bound_path(generation_root, descriptor.get("path"), artifact_kind)
    state_path = _bound_path(generation_root, descriptor.get("state_path"), f"{artifact_kind} state")
    _require_checksum(artifact_path, descriptor.get("sha256"), f"{artifact_kind} artifact")
    _require_checksum(state_path, descriptor.get("state_sha256"), f"{artifact_kind} state artifact")
    bound_state = validate_bound_parquet_artifact(artifact_path, state_path, artifact_kind=artifact_kind)
    if bound_state.get("source_corpus_fingerprint") != source_fingerprint:
        raise ValueError(f"{artifact_kind} source corpus fingerprint mismatch")
    if bound_state.get("passages_sha256") != passages_sha256:
        raise ValueError(f"{artifact_kind} source passages checksum mismatch")
    if bound_state.get("upstream_sha256") != upstream_sha256:
        raise ValueError(f"{artifact_kind} upstream checksum mismatch")
    return artifact_path


def _request_bindings(
    state: Mapping[str, Any], generation_root: Path, generator_model: str
) -> tuple[list[str], dict[str, GenerationBinding], list[dict[str, Any]]]:
    shards = state.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("generation state has no request shards")
    ordered_ids: list[str] = []
    bindings: dict[str, GenerationBinding] = {}
    ordered_requests: list[dict[str, Any]] = []
    for shard_index, shard in enumerate(shards):
        if not isinstance(shard, Mapping):
            raise ValueError(f"invalid generation shard at index {shard_index}")
        request_path = _bound_path(generation_root, shard.get("request_path"), "request shard")
        _require_checksum(request_path, shard.get("request_sha256"), "request shard")
        expected_ids = shard.get("custom_ids")
        if not isinstance(expected_ids, list) or not all(isinstance(item, str) and item for item in expected_ids):
            raise ValueError(f"request shard {shard_index} has invalid custom IDs")
        request_rows = _read_jsonl(request_path, "request shard")
        if len(request_rows) != int(shard.get("request_count", -1)) or len(request_rows) != len(expected_ids):
            raise ValueError(f"request shard {shard_index} row count mismatch")
        actual_ids: list[str] = []
        for request in request_rows:
            custom_id = request.get("custom_id")
            body = request.get("body")
            metadata = body.get("metadata") if isinstance(body, Mapping) else None
            if not isinstance(custom_id, str) or not isinstance(body, Mapping) or not isinstance(metadata, Mapping):
                raise ValueError(f"request shard {shard_index} contains an invalid request binding")
            if body.get("model") != generator_model:
                raise ValueError(f"request {custom_id} generator model mismatch")
            if custom_id in bindings:
                raise ValueError(f"duplicate generation request custom_id: {custom_id}")
            try:
                attempt = int(metadata["generation_attempt"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"request {custom_id} has invalid generation attempt") from exc
            if attempt != int(shard.get("attempt", -1)):
                raise ValueError(f"request {custom_id} generation attempt mismatch")
            values = {key: metadata.get(key) for key in ("passage_id", "article_id", "source_hash", "prompt_hash")}
            if not all(isinstance(value, str) and value for value in values.values()):
                raise ValueError(f"request {custom_id} has incomplete immutable metadata")
            batch_id = shard.get("batch_id")
            if not isinstance(batch_id, str) or not batch_id:
                raise ValueError(f"request {custom_id} has no completed batch binding")
            bindings[custom_id] = GenerationBinding(
                custom_id=custom_id,
                passage_id=values["passage_id"],
                article_id=values["article_id"],
                source_hash=values["source_hash"],
                prompt_hash=values["prompt_hash"],
                generation_attempt=attempt,
                batch_id=batch_id,
            )
            actual_ids.append(custom_id)
            ordered_ids.append(custom_id)
            ordered_requests.append(request)
        if actual_ids != expected_ids:
            raise ValueError(f"request shard {shard_index} custom ID order mismatch")
    return ordered_ids, bindings, ordered_requests


def _validate_generation_inputs(
    source_manifest: Mapping[str, Any], source_manifest_sha256: str, generation_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], pl.DataFrame, pl.DataFrame, list[str], dict[str, GenerationBinding]]:
    state = _read_json(generation_root / "generation_state.json", "generation state")
    source_fingerprint = str(source_manifest["fingerprint"])
    passages_sha256 = str(source_manifest["passages_sha256"])
    if state.get("version") != 3:
        raise ValueError("release finalization requires generation state version 3")
    if state.get("source_corpus_fingerprint") != source_fingerprint:
        raise ValueError("generation source corpus fingerprint mismatch")
    if state.get("source_passages_sha256") != passages_sha256:
        raise ValueError("generation source passages checksum mismatch")
    targets_path = _validate_generation_artifact(
        generation_root,
        state.get("targets"),
        artifact_kind="targets",
        source_fingerprint=source_fingerprint,
        passages_sha256=passages_sha256,
        upstream_sha256=source_manifest_sha256,
    )
    context_path = _validate_generation_artifact(
        generation_root,
        state.get("generation_context"),
        artifact_kind="generation_context",
        source_fingerprint=source_fingerprint,
        passages_sha256=passages_sha256,
        upstream_sha256=str(state["targets"]["sha256"]),
    )
    for path_key, sha_key, label in (
        ("collected_path", "collected_sha256", "collected artifact"),
        ("diagnostics_path", "diagnostics_sha256", "diagnostics artifact"),
        ("pilot_metrics_path", "pilot_metrics_sha256", "pilot metrics artifact"),
    ):
        artifact_path = _bound_path(generation_root, state.get(path_key), label)
        _require_checksum(artifact_path, state.get(sha_key), label)
    generation_config = state.get("generation_config")
    generator_model = generation_config.get("model") if isinstance(generation_config, Mapping) else None
    if not isinstance(generator_model, str) or not generator_model:
        raise ValueError("generation state has no generator model")
    ordered_ids, bindings, requests = _request_bindings(state, generation_root, generator_model)
    collected_path = _bound_path(generation_root, state["collected_path"], "collected artifact")
    records = _read_jsonl(collected_path, "collected artifact")
    counts = state.get("counts")
    if not isinstance(counts, Mapping) or int(counts.get("prepared", -1)) != len(records):
        raise ValueError("generation collected count does not match state")
    accepted = sum(record.get("status") == "accepted" for record in records)
    rejected = sum(record.get("status") == "rejected" for record in records)
    if accepted != int(counts.get("accepted", -1)) or rejected != int(counts.get("rejected", -1)):
        raise ValueError("generation terminal status counts do not match state")
    targets = pl.read_parquet(targets_path)
    context = pl.read_parquet(context_path)
    target_context_fingerprint = _target_context_fingerprint(targets.to_dicts(), context.to_dicts())
    if state.get("target_context_fingerprint") != target_context_fingerprint:
        raise ValueError("generation target context fingerprint mismatch")
    try:
        generator = GeneratorConfig(**dict(generation_config))
    except (TypeError, ValueError) as exc:
        raise ValueError("generation state has an invalid generator config") from exc
    generation_fingerprint = _generation_fingerprint(source_fingerprint, target_context_fingerprint, generator, requests)
    if state.get("generation_fingerprint") != generation_fingerprint:
        raise ValueError("generation fingerprint mismatch")
    return state, records, targets, context, ordered_ids, bindings


def _audit_frame(
    records: Sequence[Mapping[str, Any]],
    targets: pl.DataFrame,
    context: pl.DataFrame,
    ordered_ids: Sequence[str],
    bindings: Mapping[str, GenerationBinding],
) -> pl.DataFrame:
    target_rows = _unique_rows(targets, "passage_id", "targets")
    record_rows = {str(record["custom_id"]): record for record in records}
    contexts: dict[str, list[dict[str, Any]]] = {}
    _required_columns(
        context,
        ("target_passage_id", "candidate_passage_id", "candidate_serialized_passage", "context_index"),
        "generation context",
    )
    for row in context.sort(["target_passage_id", "context_index"]).iter_rows(named=True):
        contexts.setdefault(str(row["target_passage_id"]), []).append(row)
    rows: list[dict[str, Any]] = []
    for custom_id in ordered_ids:
        binding = bindings[custom_id]
        target = target_rows[binding.passage_id]
        record = record_rows[custom_id]
        output = record.get("output") if isinstance(record.get("output"), Mapping) else {}
        candidates = contexts.get(binding.passage_id, [])
        if len(candidates) < 3:
            raise ValueError(f"audit target {binding.passage_id} has fewer than three competitors")
        row = {
            "generation_custom_id": custom_id,
            "automatic_status": str(record.get("status") or ""),
            "automatic_reason_codes": _canonical_json(record.get("reason_codes") or []),
            "requested_intent": str(output.get("requested_intent") or target.get("requested_intent") or ""),
            "actual_intent": str(output.get("actual_intent") or ""),
            "query": str(output.get("query") or ""),
            "answer": str(output.get("answer") or ""),
            "evidence": str(output.get("evidence") or ""),
            "positive_passage_id": binding.passage_id,
            "article_id": binding.article_id,
            "split": SPLIT_NAMES[str(target["split"])],
            "title": str(target.get("title") or ""),
            "section": str(target.get("section") or ""),
            "url": str(target.get("url") or ""),
            "positive_passage": str(target["serialized_passage"]),
        }
        for index, candidate in enumerate(candidates[:3], start=1):
            row[f"competitor_{index}_passage_id"] = str(candidate["candidate_passage_id"])
            row[f"competitor_{index}_passage"] = str(candidate["candidate_serialized_passage"])
        row.update({column: "" for column in AUDIT_HUMAN_COLUMNS})
        rows.append(row)
    return pl.DataFrame(rows)


def _csv_safe_cell(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    stripped = value.lstrip()
    if value[0] in "\t\r\n" or stripped.startswith(_SPREADSHEET_FORMULA_PREFIXES):
        return f"'{value}"
    return value


def _csv_safe_frame(frame: pl.DataFrame) -> pl.DataFrame:
    rows = [{key: _csv_safe_cell(value) for key, value in row.items()} for row in frame.iter_rows(named=True)]
    return pl.DataFrame(rows, schema=frame.schema)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _hf_arrow_type(dtype: pa.DataType) -> pa.DataType:
    if pa.types.is_large_string(dtype):
        return pa.string()
    if pa.types.is_large_binary(dtype):
        return pa.binary()
    if pa.types.is_large_list(dtype) or pa.types.is_list(dtype):
        return pa.list_(_hf_arrow_type(dtype.value_type))
    if pa.types.is_struct(dtype):
        return pa.struct(
            [pa.field(field.name, _hf_arrow_type(field.type), nullable=field.nullable, metadata=field.metadata) for field in dtype]
        )
    return dtype


def _write_hf_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = frame.to_arrow()
    schema = pa.schema(
        [
            pa.field(field.name, _hf_arrow_type(field.type), nullable=field.nullable, metadata=field.metadata)
            for field in table.schema
        ],
        metadata=table.schema.metadata,
    )
    pq.write_table(
        table.cast(schema),
        path,
        compression="zstd",
        row_group_size=50_000,
        write_page_index=True,
    )
    _fsync_file(path)


def _dataset_card() -> str:
    return """---
configs:
- config_name: pairs
  data_files:
  - split: train
    path: data/pairs/train.parquet
  - split: validation
    path: data/pairs/validation.parquet
  - split: test
    path: data/pairs/test.parquet
- config_name: corpus-100k
  data_files:
  - split: train
    path: data/corpus-100k/corpus.parquet
- config_name: qrels
  data_files:
  - split: train
    path: data/qrels/train.parquet
  - split: validation
    path: data/qrels/validation.parquet
  - split: test
    path: data/qrels/test.parquet
---

# Habr IR

Local pilot release. See the repository research record for provenance and audit status.
"""


def _release_fingerprint(source_manifest: Mapping[str, Any], state: Mapping[str, Any], *, git_sha: str, git_dirty: bool) -> str:
    return _canonical_hash(
        {
            "version": 1,
            "source_corpus_fingerprint": source_manifest["fingerprint"],
            "source_passages_sha256": source_manifest["passages_sha256"],
            "generation_fingerprint": state.get("generation_fingerprint"),
            "targets_sha256": state["targets"]["sha256"],
            "generation_context_sha256": state["generation_context"]["sha256"],
            "collected_sha256": state["collected_sha256"],
            "git": {"sha": git_sha, "dirty": bool(git_dirty)},
            "contract": "habr-ir-release-v1",
        }
    )


def _summary_from_manifest(root: Path, manifest: Mapping[str, Any], *, reused: bool) -> ReleaseSummary:
    counts = manifest["counts"]
    return ReleaseSummary(
        root=root,
        manifest_path=root / "data/manifests/release-manifest.json",
        pair_count=int(counts["pairs"]),
        corpus_count=int(counts["corpus"]),
        qrel_count=int(counts["qrels"]),
        review_count=int(counts["audit_rows"]),
        fingerprint=str(manifest["fingerprint"]),
        reused=reused,
    )


def _validate_existing_release(root: Path, fingerprint: str) -> ReleaseSummary:
    manifest_path = root / "data/manifests/release-manifest.json"
    manifest = _read_json(manifest_path, "release manifest")
    if manifest.get("fingerprint") != fingerprint:
        raise ValueError("refusing to overwrite a release with a different fingerprint")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("release manifest artifacts are invalid")
    artifact_paths = [item.get("path") for item in artifacts if isinstance(item, Mapping)]
    if len(artifact_paths) != len(artifacts) or len(artifact_paths) != len(set(artifact_paths)):
        raise ValueError("release required artifact set mismatch")
    if set(artifact_paths) != RELEASE_ARTIFACT_PATHS:
        raise ValueError("release required artifact set mismatch")
    for item in artifacts:
        path = _bound_path(root, item.get("path"), "release artifact")
        _require_checksum(path, item.get("sha256"), "release artifact")
        if item.get("bytes") != path.stat().st_size:
            raise ValueError("release artifact byte count mismatch")

    pair_splits = {
        split: pq.ParquetFile(root / f"data/pairs/{split}.parquet").metadata.num_rows for split in ("train", "validation", "test")
    }
    qrel_count = sum(
        pq.ParquetFile(root / f"data/qrels/{split}.parquet").metadata.num_rows for split in ("train", "validation", "test")
    )
    actual_counts = {
        "pairs": sum(pair_splits.values()),
        "pair_splits": pair_splits,
        "corpus": pq.ParquetFile(root / "data/corpus-100k/corpus.parquet").metadata.num_rows,
        "qrels": qrel_count,
        "audit_rows": pl.read_csv(root / "audit/pilot-review.csv").height,
    }
    if manifest.get("counts") != actual_counts:
        raise ValueError("release counts mismatch")
    return _summary_from_manifest(root, manifest, reused=True)


def finalize_release(
    source_root: str | Path,
    generation_root: str | Path,
    release_root: str | Path,
    *,
    git_sha: str,
    git_dirty: bool,
) -> ReleaseSummary:
    """Validate collected generation state and atomically write a local release."""
    source = Path(source_root)
    generation = Path(generation_root)
    release = Path(release_root)
    if not isinstance(git_sha, str) or not git_sha:
        raise ValueError("git_sha must be a non-empty string")
    source_manifest, passages_path, source_manifest_sha256 = _validate_source(source)
    state, records, targets, context, ordered_ids, bindings = _validate_generation_inputs(
        source_manifest,
        source_manifest_sha256,
        generation,
    )
    fingerprint = _release_fingerprint(source_manifest, state, git_sha=git_sha, git_dirty=git_dirty)
    if release.exists():
        return _validate_existing_release(release, fingerprint)

    generation_config = state["generation_config"]
    frames = materialize_release_frames(
        records=records,
        targets=targets,
        corpus=pl.read_parquet(passages_path),
        bindings=bindings,
        generator_model=str(generation_config["model"]),
    )
    review = _audit_frame(records, targets, context, ordered_ids, bindings)
    temporary = release.with_name(f".{release.name}.{uuid.uuid4().hex}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        for split in ("train", "validation", "test"):
            pair_path = temporary / "data/pairs" / f"{split}.parquet"
            _write_hf_parquet(frames.pairs.filter(pl.col("split") == split), pair_path)
            split_qrels = (
                frames.qrels.join(
                    frames.pairs.filter(pl.col("split") == split).select("query_id"),
                    on="query_id",
                    how="semi",
                )
                .select(list(QREL_SCHEMA))
                .sort("query_id")
            )
            qrel_path = temporary / "data/qrels" / f"{split}.parquet"
            _write_hf_parquet(split_qrels, qrel_path)
        corpus_path = temporary / "data/corpus-100k/corpus.parquet"
        _write_hf_parquet(frames.corpus, corpus_path)
        review_path = temporary / "audit/pilot-review.csv"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        _csv_safe_frame(review).write_csv(review_path, include_bom=True)
        _fsync_file(review_path)
        _write_text(temporary / "README.md", _dataset_card())

        artifacts: list[dict[str, Any]] = []
        for path in sorted(item for item in temporary.rglob("*") if item.is_file()):
            artifacts.append(
                {
                    "path": path.relative_to(temporary).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
        split_counts = {split: frames.pairs.filter(pl.col("split") == split).height for split in ("train", "validation", "test")}
        manifest = {
            "version": 1,
            "fingerprint": fingerprint,
            "source": {
                "corpus_fingerprint": source_manifest["fingerprint"],
                "passages_sha256": source_manifest["passages_sha256"],
                "chunker_version": source_manifest["chunker_version"],
            },
            "generation": {
                "fingerprint": state.get("generation_fingerprint"),
                "model": generation_config["model"],
                "collected_sha256": state["collected_sha256"],
            },
            "git": {"sha": git_sha, "dirty": bool(git_dirty)},
            "counts": {
                "pairs": frames.pairs.height,
                "pair_splits": split_counts,
                "corpus": frames.corpus.height,
                "qrels": frames.qrels.height,
                "audit_rows": review.height,
            },
            "schemas": {
                "pairs": {name: str(dtype) for name, dtype in frames.pairs.schema.items()},
                "corpus": {name: str(dtype) for name, dtype in frames.corpus.schema.items()},
                "qrels": {name: str(dtype) for name, dtype in frames.qrels.schema.items()},
            },
            "artifacts": artifacts,
        }
        manifest_path = temporary / "data/manifests/release-manifest.json"
        _write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        release.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, release)
        return _summary_from_manifest(release, manifest, reused=False)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


__all__ = [
    "GenerationBinding",
    "ReleaseFrames",
    "ReleaseSummary",
    "finalize_release",
    "materialize_release_frames",
    "stable_pair_id",
    "stable_query_id",
]
