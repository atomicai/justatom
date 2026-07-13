from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from justatom.tooling.ir_dataset.generation import GENERATOR_SCHEMA, GeneratorConfig, build_generator_request


DEFAULT_MAX_REQUESTS = 1_000
DEFAULT_MAX_BYTES = 100_000_000
STATE_FILE_NAME = "generation_state.json"
COLLECTED_FILE_NAME = "generation_collected.jsonl"


@dataclass(frozen=True, slots=True)
class BatchShard:
    path: Path
    request_count: int
    byte_count: int
    sha256: str
    custom_ids: tuple[str, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_bytes_atomic(path, (_canonical_json(payload) + "\n").encode("utf-8"))


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"batch artifact path is outside output root: {path}") from exc


def _read_rows(value: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]] | Any, label: str) -> list[dict[str, Any]]:
    rows = value.to_dicts() if hasattr(value, "to_dicts") else list(value)
    if not all(isinstance(row, Mapping) for row in rows):
        raise TypeError(f"{label} rows must be mappings")
    return [dict(row) for row in rows]


def _validate_request(request: Mapping[str, Any]) -> tuple[str, bytes]:
    custom_id = request.get("custom_id")
    if not isinstance(custom_id, str) or not custom_id:
        raise ValueError("batch request is missing a non-empty custom_id")
    return custom_id, (_canonical_json(dict(request)) + "\n").encode("utf-8")


def write_batch_shards(
    requests: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[BatchShard]:
    """Write immutable OpenAI Batch JSONL shards with the documented caps."""
    if not 1 <= int(max_requests) <= DEFAULT_MAX_REQUESTS:
        raise ValueError("max_requests must be within [1, 1000]")
    if not 1 <= int(max_bytes) <= DEFAULT_MAX_BYTES:
        raise ValueError("max_bytes must be within [1, 100000000]")
    root = Path(output_dir)
    seen_ids: set[str] = set()
    encoded: list[tuple[str, bytes]] = []
    for request in requests:
        if not isinstance(request, Mapping):
            raise TypeError("batch requests must be mappings")
        custom_id, row = _validate_request(request)
        if custom_id in seen_ids:
            raise ValueError(f"duplicate custom_id in batch requests: {custom_id}")
        seen_ids.add(custom_id)
        if len(row) > max_bytes:
            raise ValueError(f"request {custom_id} exceeds max_bytes")
        encoded.append((custom_id, row))
    if not encoded:
        raise ValueError("batch requests must not be empty")

    batches: list[list[tuple[str, bytes]]] = []
    current: list[tuple[str, bytes]] = []
    current_bytes = 0
    for row in encoded:
        if current and (len(current) >= max_requests or current_bytes + len(row[1]) > max_bytes):
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(row)
        current_bytes += len(row[1])
    if current:
        batches.append(current)

    shards: list[BatchShard] = []
    for index, rows in enumerate(batches):
        content = b"".join(row for _, row in rows)
        path = root / f"generation-{index:05d}.jsonl"
        checksum = _sha256_bytes(content)
        if path.exists():
            if _sha256_file(path) != checksum:
                raise ValueError(f"refusing to reuse checksum-mismatched shard: {path}")
        else:
            _write_bytes_atomic(path, content)
        shards.append(
            BatchShard(
                path=path,
                request_count=len(rows),
                byte_count=len(content),
                sha256=checksum,
                custom_ids=tuple(custom_id for custom_id, _ in rows),
            )
        )
    return shards


def _corpus_fingerprint(targets: list[dict[str, Any]], context: list[dict[str, Any]]) -> str:
    return _sha256_bytes(_canonical_json({"targets": targets, "context": context}).encode("utf-8"))


def _generation_fingerprint(corpus_fingerprint: str, config: GeneratorConfig, requests: list[dict[str, Any]]) -> str:
    payload = {
        "corpus_fingerprint": corpus_fingerprint,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "attempt": config.attempt,
        "schema": GENERATOR_SCHEMA,
        "requests": requests,
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _state_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / STATE_FILE_NAME


def _load_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid batch state: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid batch state: {path}")
    return value


def _validate_shard_checksum(shard: Mapping[str, Any], output_dir: str | Path) -> Path:
    request_path = Path(output_dir) / str(shard["request_path"])
    expected = str(shard["request_sha256"])
    if not request_path.exists() or _sha256_file(request_path) != expected:
        raise ValueError(f"request shard checksum mismatch: {request_path}")
    return request_path


def _validate_state_requests(state: Mapping[str, Any], output_dir: str | Path) -> None:
    seen_ids: set[str] = set()
    for shard in state.get("shards", []):
        if not isinstance(shard, Mapping):
            raise ValueError("invalid batch state shard")
        _validate_shard_checksum(shard, output_dir)
        for custom_id in shard.get("custom_ids", []):
            if custom_id in seen_ids:
                raise ValueError(f"duplicate custom_id in batch state: {custom_id}")
            seen_ids.add(custom_id)


def prepare_generation_batches(
    targets: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]] | Any,
    generation_context: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]] | Any,
    config: GeneratorConfig | Mapping[str, Any] | None,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build a corpus-fingerprinted, resumable OpenAI Batch state manifest."""
    active_config = config if isinstance(config, GeneratorConfig) else GeneratorConfig(**dict(config or {}))
    target_rows = _read_rows(targets, "target")
    context_rows = _read_rows(generation_context, "generation context")
    if not target_rows:
        raise ValueError("targets must not be empty")
    target_ids = [str(row.get("passage_id", "")) for row in target_rows]
    if not all(target_ids) or len(set(target_ids)) != len(target_ids):
        raise ValueError("targets must have unique non-empty passage_id values")
    context_by_target: dict[str, list[dict[str, Any]]] = {}
    for row in context_rows:
        target_id = str(row.get("target_passage_id", ""))
        if not target_id:
            raise ValueError("generation context is missing target_passage_id")
        context_by_target.setdefault(target_id, []).append(row)
    requests = [
        build_generator_request(row, context_by_target.get(str(row["passage_id"]), ()), active_config) for row in target_rows
    ]
    corpus_fingerprint = _corpus_fingerprint(target_rows, context_rows)
    generation_fingerprint = _generation_fingerprint(corpus_fingerprint, active_config, requests)
    root = Path(output_dir)
    state_path = _state_path(root)
    if state_path.exists():
        existing = _load_state(state_path)
        if (
            existing.get("corpus_fingerprint") != corpus_fingerprint
            or existing.get("generation_fingerprint") != generation_fingerprint
        ):
            raise ValueError("refusing to reuse batch state with a different corpus or generation fingerprint")
        _validate_state_requests(existing, root)
        return existing

    shards = write_batch_shards(
        requests,
        root / "generation_requests",
        max_requests=active_config.max_requests_per_shard,
        max_bytes=min(active_config.max_shard_bytes, DEFAULT_MAX_BYTES),
    )
    state: dict[str, Any] = {
        "version": 1,
        "corpus_fingerprint": corpus_fingerprint,
        "generation_fingerprint": generation_fingerprint,
        "shards": [
            {
                "request_path": _relative_to(shard.path, root),
                "request_count": shard.request_count,
                "request_bytes": shard.byte_count,
                "request_sha256": shard.sha256,
                "custom_ids": list(shard.custom_ids),
                "input_file_id": None,
                "batch_id": None,
                "status": "prepared",
                "output_file_id": None,
                "error_file_id": None,
            }
            for shard in shards
        ],
        "counts": {"prepared": len(requests)},
    }
    _write_json_atomic(state_path, state)
    return state


def _persist_state(state: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    result = dict(state)
    _write_json_atomic(_state_path(output_dir), result)
    return result


def _object_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def submit_pending_shards(state: Mapping[str, Any], client: Any, output_dir: str | Path) -> dict[str, Any]:
    """Upload and create each absent batch exactly once, persisting after every remote call."""
    result = copy.deepcopy(dict(state))
    _validate_state_requests(result, output_dir)
    changed = False
    for shard in result.get("shards", []):
        request_path = _validate_shard_checksum(shard, output_dir)
        if not shard.get("input_file_id"):
            with request_path.open("rb") as stream:
                uploaded = client.files.create(file=stream, purpose="batch")
            input_file_id = _object_value(uploaded, "id")
            if not input_file_id:
                raise ValueError("OpenAI file upload did not return an id")
            shard["input_file_id"] = str(input_file_id)
            shard["status"] = "uploaded"
            result = _persist_state(result, output_dir)
            changed = True
        if not shard.get("batch_id"):
            batch = client.batches.create(
                input_file_id=str(shard["input_file_id"]), endpoint="/v1/responses", completion_window="24h"
            )
            batch_id = _object_value(batch, "id")
            if not batch_id:
                raise ValueError("OpenAI batch creation did not return an id")
            shard["batch_id"] = str(batch_id)
            shard["status"] = str(_object_value(batch, "status", "submitted"))
            result = _persist_state(result, output_dir)
            changed = True
    if not changed:
        return dict(state)
    result["counts"] = {"submitted": sum(1 for shard in result["shards"] if shard.get("batch_id"))}
    return _persist_state(result, output_dir)


def refresh_batch_status(state: Mapping[str, Any], client: Any, output_dir: str | Path) -> dict[str, Any]:
    """Fetch and persist batch status and remotely assigned output/error file IDs."""
    result = copy.deepcopy(dict(state))
    _validate_state_requests(result, output_dir)
    for shard in result.get("shards", []):
        batch_id = shard.get("batch_id")
        if not batch_id:
            continue
        batch = client.batches.retrieve(str(batch_id))
        shard["status"] = str(_object_value(batch, "status", shard.get("status", "submitted")))
        for field in ("output_file_id", "error_file_id"):
            value = _object_value(batch, field)
            if value is not None:
                shard[field] = str(value)
    result["counts"] = {
        "submitted": sum(1 for shard in result["shards"] if shard.get("batch_id")),
        "completed": sum(1 for shard in result["shards"] if shard.get("status") == "completed"),
    }
    return _persist_state(result, output_dir)


def _content_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    for field in ("content", "text"):
        nested = _object_value(value, field)
        if isinstance(nested, bytes):
            return nested
        if isinstance(nested, str):
            return nested.encode("utf-8")
    read = getattr(value, "read", None)
    if callable(read):
        result = read()
        return result if isinstance(result, bytes) else str(result).encode("utf-8")
    raise TypeError("OpenAI file content must be bytes, text, or expose read()")


def _download_file(shard: dict[str, Any], field: str, client: Any, output_dir: Path) -> bytes | None:
    file_id = shard.get(field)
    if not file_id:
        return None
    artifact_field = field.replace("_file_id", "_path")
    checksum_field = field.replace("_file_id", "_sha256")
    existing_path = shard.get(artifact_field)
    existing_checksum = shard.get(checksum_field)
    if existing_path or existing_checksum:
        if not existing_path or not existing_checksum:
            raise ValueError(f"incomplete persisted {field} artifact metadata")
        path = output_dir / str(existing_path)
        if not path.exists() or _sha256_file(path) != str(existing_checksum):
            raise ValueError(f"downloaded {field} checksum mismatch: {path}")
        return path.read_bytes()
    content = _content_bytes(client.files.content(str(file_id)))
    path = output_dir / "generation_outputs" / f"{file_id}.jsonl"
    _write_bytes_atomic(path, content)
    shard[artifact_field] = _relative_to(path, output_dir)
    shard[checksum_field] = _sha256_bytes(content)
    return content


def _output_text_from_body(body: Any) -> str | None:
    direct = _object_value(body, "output_text")
    if isinstance(direct, str):
        return direct
    for output in _object_value(body, "output", []) or []:
        for content in _object_value(output, "content", []) or []:
            if _object_value(content, "type") == "output_text" and isinstance(_object_value(content, "text"), str):
                return str(_object_value(content, "text"))
    return None


def _strict_output(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != set(GENERATOR_SCHEMA["required"]):
        return None
    if not isinstance(value.get("usable"), bool):
        return None
    if not all(
        isinstance(value.get(key), str) for key in ("reason", "query", "answer", "evidence", "requested_intent", "actual_intent")
    ):
        return None
    if value["reason"] not in GENERATOR_SCHEMA["properties"]["reason"]["enum"]:
        return None
    intents = GENERATOR_SCHEMA["properties"]["requested_intent"]["enum"]
    if value["requested_intent"] not in intents or value["actual_intent"] not in intents:
        return None
    if not isinstance(value.get("disambiguators"), list) or not all(isinstance(item, str) for item in value["disambiguators"]):
        return None
    return value


def _parse_batch_rows(content: bytes, known_ids: set[str], seen_ids: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_line in content.decode("utf-8", errors="replace").splitlines():
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError:
            records.append({"status": "rejected", "reason": "malformed_batch_row", "raw": raw_line})
            continue
        custom_id = _object_value(row, "custom_id")
        if not isinstance(custom_id, str) or custom_id not in known_ids:
            records.append({"status": "rejected", "reason": "unknown_custom_id", "custom_id": custom_id})
            continue
        if custom_id in seen_ids:
            records.append({"status": "rejected", "reason": "duplicate_custom_id", "custom_id": custom_id})
            continue
        seen_ids.add(custom_id)
        response = _object_value(row, "response")
        if _object_value(response, "status_code") != 200:
            records.append(
                {
                    "status": "rejected",
                    "reason": f"response_status_{_object_value(response, 'status_code', 'missing')}",
                    "custom_id": custom_id,
                }
            )
            continue
        output_text = _output_text_from_body(_object_value(response, "body"))
        try:
            parsed = json.loads(output_text) if output_text is not None else None
        except json.JSONDecodeError:
            parsed = None
        output = _strict_output(parsed)
        if output is None:
            records.append({"status": "rejected", "reason": "structured_output_invalid", "custom_id": custom_id})
        else:
            records.append({"status": "accepted", "custom_id": custom_id, "output": output})
    return records


def collect_completed_shards(state: Mapping[str, Any], client: Any, output_dir: str | Path) -> dict[str, Any]:
    """Download completed batch files and persist strict accepted/rejected records once."""
    root = Path(output_dir)
    result = copy.deepcopy(dict(state))
    _validate_state_requests(result, root)
    collected_path = root / COLLECTED_FILE_NAME
    existing_checksum = result.get("collected_sha256")
    if existing_checksum:
        if not collected_path.exists() or _sha256_file(collected_path) != existing_checksum:
            raise ValueError(f"collected artifact checksum mismatch: {collected_path}")
        return dict(state)

    refreshed = refresh_batch_status(result, client, root)
    known_ids = {str(custom_id) for shard in refreshed["shards"] for custom_id in shard["custom_ids"]}
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for shard in refreshed["shards"]:
        if shard.get("status") != "completed":
            continue
        for field in ("output_file_id", "error_file_id"):
            content = _download_file(shard, field, client, root)
            if content is not None:
                records.extend(_parse_batch_rows(content, known_ids, seen_ids))
    encoded = b"".join((_canonical_json(record) + "\n").encode("utf-8") for record in records)
    _write_bytes_atomic(collected_path, encoded)
    refreshed["collected_path"] = COLLECTED_FILE_NAME
    refreshed["collected_sha256"] = _sha256_bytes(encoded)
    refreshed["counts"] = {
        "accepted": sum(record["status"] == "accepted" for record in records),
        "rejected": sum(record["status"] == "rejected" for record in records),
    }
    return _persist_state(refreshed, root)


__all__ = [
    "BatchShard",
    "COLLECTED_FILE_NAME",
    "STATE_FILE_NAME",
    "collect_completed_shards",
    "prepare_generation_batches",
    "refresh_batch_status",
    "submit_pending_shards",
    "write_batch_shards",
]
