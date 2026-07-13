from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from justatom.tooling.ir_dataset.generation import GENERATOR_SCHEMA, GeneratorConfig, build_generator_request


REQUIRED_SOURCE_CORPUS_FINGERPRINT = "bb6ad903b82c337a61cce2b1cd5bf5dd7e3303b6b3263258979372c00e40c3c9"
DEFAULT_MAX_REQUESTS = 1_000
DEFAULT_MAX_BYTES = 100_000_000
STATE_FILE_NAME = "generation_state.json"
COLLECTED_FILE_NAME = "generation_collected.jsonl"
DIAGNOSTICS_FILE_NAME = "generation_diagnostics.jsonl"


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
    parent_created = not path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        if parent_created:
            _fsync_directory(path.parent.parent)
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

    groups: list[list[tuple[str, bytes]]] = []
    current: list[tuple[str, bytes]] = []
    current_bytes = 0
    for row in encoded:
        if current and (len(current) >= max_requests or current_bytes + len(row[1]) > max_bytes):
            groups.append(current)
            current, current_bytes = [], 0
        current.append(row)
        current_bytes += len(row[1])
    if current:
        groups.append(current)

    root = Path(output_dir)
    shards: list[BatchShard] = []
    for index, rows in enumerate(groups):
        content = b"".join(row for _, row in rows)
        checksum = _sha256_bytes(content)
        path = root / f"generation-{index:05d}-{checksum[:16]}.jsonl"
        if path.exists():
            if _sha256_file(path) != checksum:
                raise ValueError(f"refusing to reuse checksum-mismatched shard: {path}")
        else:
            _write_bytes_atomic(path, content)
        shards.append(BatchShard(path, len(rows), len(content), checksum, tuple(custom_id for custom_id, _ in rows)))
    return shards


def _target_context_fingerprint(targets: list[dict[str, Any]], context: list[dict[str, Any]]) -> str:
    return _sha256_bytes(_canonical_json({"targets": targets, "context": context}).encode("utf-8"))


def _generation_fingerprint(
    source_corpus_fingerprint: str,
    target_context_fingerprint: str,
    config: GeneratorConfig,
    requests: list[dict[str, Any]],
) -> str:
    payload = {
        "source_corpus_fingerprint": source_corpus_fingerprint,
        "target_context_fingerprint": target_context_fingerprint,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "attempt": config.attempt,
        "max_requests_per_shard": config.max_requests_per_shard,
        "max_shard_bytes": config.max_shard_bytes,
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
            if not isinstance(custom_id, str) or custom_id in seen_ids:
                raise ValueError(f"duplicate custom_id in batch state: {custom_id}")
            seen_ids.add(custom_id)


def prepare_generation_batches(
    targets: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]] | Any,
    generation_context: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]] | Any,
    config: GeneratorConfig | Mapping[str, Any] | None,
    output_dir: str | Path,
    *,
    source_corpus_fingerprint: str,
) -> dict[str, Any]:
    """Build a corpus-bound, resumable OpenAI Batch state manifest."""
    if source_corpus_fingerprint != REQUIRED_SOURCE_CORPUS_FINGERPRINT:
        raise ValueError("source corpus manifest fingerprint does not match the required Habr corpus")
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
    context_fingerprint = _target_context_fingerprint(target_rows, context_rows)
    generation_fingerprint = _generation_fingerprint(source_corpus_fingerprint, context_fingerprint, active_config, requests)
    root = Path(output_dir)
    state_path = _state_path(root)
    if state_path.exists():
        existing = _load_state(state_path)
        if (
            existing.get("source_corpus_fingerprint") != source_corpus_fingerprint
            or existing.get("target_context_fingerprint") != context_fingerprint
            or existing.get("generation_fingerprint") != generation_fingerprint
        ):
            raise ValueError("refusing to reuse batch state with a different corpus, target/context, or generation fingerprint")
        _validate_state_requests(existing, root)
        return existing

    shards = write_batch_shards(
        requests,
        root / "generation_requests",
        max_requests=active_config.max_requests_per_shard,
        max_bytes=active_config.max_shard_bytes,
    )
    state: dict[str, Any] = {
        "version": 2,
        "source_corpus_fingerprint": source_corpus_fingerprint,
        "target_context_fingerprint": context_fingerprint,
        "generation_fingerprint": generation_fingerprint,
        "limits": {
            "model": active_config.model,
            "max_requests_per_shard": active_config.max_requests_per_shard,
            "max_shard_bytes": active_config.max_shard_bytes,
        },
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


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _json_value(dump())
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def _remote_items(service: Any) -> list[Any]:
    try:
        page = service.list(limit=100)
    except TypeError:
        page = service.list()
    items = list(_object_value(page, "data", page) or [])
    while callable(getattr(page, "has_next_page", None)) and page.has_next_page():
        page = page.get_next_page()
        items.extend(_object_value(page, "data", page) or [])
    return items


def _file_operation(shard: Mapping[str, Any], request_path: Path) -> dict[str, Any]:
    return {"filename": request_path.name, "bytes": int(shard["request_bytes"]), "sha256": str(shard["request_sha256"])}


def _find_remote_file(client: Any, operation: Mapping[str, Any]) -> str | None:
    for item in _remote_items(client.files):
        if _object_value(item, "purpose") != "batch":
            continue
        if _object_value(item, "filename") == operation["filename"] and int(_object_value(item, "bytes", -1)) == operation["bytes"]:
            item_id = _object_value(item, "id")
            if item_id:
                return str(item_id)
    return None


def _batch_operation(state: Mapping[str, Any], shard: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {
        "justatom_generation_fingerprint": str(state["generation_fingerprint"]),
        "justatom_shard_sha256": str(shard["request_sha256"]),
    }
    return {"input_file_id": str(shard["input_file_id"]), "metadata": metadata}


def _find_remote_batch(client: Any, operation: Mapping[str, Any]) -> str | None:
    for item in _remote_items(client.batches):
        if _object_value(item, "input_file_id") != operation["input_file_id"]:
            continue
        metadata = dict(_object_value(item, "metadata", {}) or {})
        if any(metadata.get(key) != value for key, value in operation["metadata"].items()):
            continue
        item_id = _object_value(item, "id")
        if item_id:
            return str(item_id)
    return None


@contextmanager
def _submit_lock(output_dir: str | Path):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".generation_submit.lock"
    try:
        stream = path.open("a+")
    except OSError as exc:
        raise RuntimeError(f"unable to open local generation submission lock: {path}") from exc
    with stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            raise RuntimeError(f"unable to acquire local generation submission lock: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _submit_pending_shards_unlocked(state: Mapping[str, Any], client: Any, output_dir: str | Path) -> dict[str, Any]:
    """Submit each shard with durable operation identities and remote reconciliation."""
    result = copy.deepcopy(dict(state))
    _validate_state_requests(result, output_dir)
    for shard in result.get("shards", []):
        request_path = _validate_shard_checksum(shard, output_dir)
        if not shard.get("input_file_id"):
            shard.setdefault("file_operation", _file_operation(shard, request_path))
            _persist_state(result, output_dir)
            input_file_id = _find_remote_file(client, shard["file_operation"])
            if input_file_id is None:
                with request_path.open("rb") as stream:
                    uploaded = client.files.create(file=stream, purpose="batch")
                input_file_id = _object_value(uploaded, "id")
            if not input_file_id:
                raise ValueError("OpenAI file upload did not return an id")
            shard["input_file_id"] = str(input_file_id)
            shard["status"] = "uploaded"
            _persist_state(result, output_dir)
        if not shard.get("batch_id"):
            shard.setdefault("batch_operation", _batch_operation(result, shard))
            _persist_state(result, output_dir)
            batch_id = _find_remote_batch(client, shard["batch_operation"])
            if batch_id is None:
                batch = client.batches.create(
                    input_file_id=str(shard["input_file_id"]),
                    endpoint="/v1/responses",
                    completion_window="24h",
                    metadata=dict(shard["batch_operation"]["metadata"]),
                )
                batch_id = _object_value(batch, "id")
                status = _object_value(batch, "status", "submitted")
            else:
                status = "submitted"
            if not batch_id:
                raise ValueError("OpenAI batch creation did not return an id")
            shard["batch_id"] = str(batch_id)
            shard["status"] = str(status)
            _persist_state(result, output_dir)
    result["counts"] = {"submitted": sum(1 for shard in result["shards"] if shard.get("batch_id"))}
    return _persist_state(result, output_dir)


def submit_pending_shards(state: Mapping[str, Any], client: Any, output_dir: str | Path) -> dict[str, Any]:
    """Serialize local submit/reconcile transitions; this does not provide cross-machine locking."""
    with _submit_lock(output_dir):
        return _submit_pending_shards_unlocked(state, client, output_dir)


def refresh_batch_status(state: Mapping[str, Any], client: Any, output_dir: str | Path) -> dict[str, Any]:
    """Fetch and persist status, artifact IDs, and top-level Batch errors."""
    result = copy.deepcopy(dict(state))
    _validate_state_requests(result, output_dir)
    for shard in result.get("shards", []):
        if not shard.get("batch_id"):
            continue
        batch = client.batches.retrieve(str(shard["batch_id"]))
        shard["status"] = str(_object_value(batch, "status", shard.get("status", "submitted")))
        for field in ("output_file_id", "error_file_id"):
            value = _object_value(batch, field)
            if value is not None:
                shard[field] = str(value)
        errors = _object_value(batch, "errors")
        if errors is not None:
            shard["batch_errors"] = _json_value(errors)
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
        content = read()
        return content if isinstance(content, bytes) else str(content).encode("utf-8")
    raise TypeError("OpenAI file content must be bytes, text, or expose read()")


def _download_file(shard: dict[str, Any], field: str, client: Any, output_dir: Path) -> bytes:
    file_id = shard.get(field)
    if not file_id:
        raise ValueError(f"missing completed shard {field}")
    artifact_field = field.replace("_file_id", "_path")
    checksum_field = field.replace("_file_id", "_sha256")
    existing_path, existing_checksum = shard.get(artifact_field), shard.get(checksum_field)
    if existing_path or existing_checksum:
        if not existing_path or not existing_checksum:
            raise ValueError(f"incomplete persisted {field} artifact metadata")
        path = output_dir / str(existing_path)
        if not path.exists() or _sha256_file(path) != str(existing_checksum):
            raise ValueError(f"downloaded {field} checksum mismatch: {path}")
        return path.read_bytes()
    path = output_dir / "generation_outputs" / f"{file_id}.jsonl"
    if path.exists():
        content = path.read_bytes()
    else:
        content = _content_bytes(client.files.content(str(file_id)))
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


def _parse_artifact_rows(
    content: bytes,
    *,
    shard_index: int,
    expected_ids: set[str],
    all_ids: set[str],
    events: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for raw_line in content.decode("utf-8", errors="replace").splitlines():
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError:
            diagnostics.append({"reason": "malformed_batch_row", "shard_index": shard_index, "raw": raw_line})
            continue
        if not isinstance(row, Mapping):
            diagnostics.append({"reason": "malformed_batch_row", "shard_index": shard_index, "raw": row})
            continue
        custom_id = row.get("custom_id")
        response = row.get("response")
        if not isinstance(custom_id, str) or custom_id not in all_ids:
            diagnostics.append(
                {"reason": "unknown_custom_id", "shard_index": shard_index, "custom_id": custom_id, "raw": _json_value(row)}
            )
            continue
        if custom_id not in expected_ids:
            diagnostics.append(
                {"reason": "cross_shard_custom_id", "shard_index": shard_index, "custom_id": custom_id, "raw": _json_value(row)}
            )
            continue
        events.setdefault(custom_id, []).append({"kind": "response" if response is not None else "error", "row": dict(row)})
    return diagnostics


def _response_record(custom_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
    response = row.get("response")
    status_code = _object_value(response, "status_code")
    body = _object_value(response, "body")
    if status_code != 200:
        return {
            "status": "rejected",
            "reason": f"response_status_{status_code if status_code is not None else 'missing'}",
            "custom_id": custom_id,
            "response_body": _json_value(body),
            "error": _json_value(_object_value(response, "error")),
        }
    output_text = _output_text_from_body(body)
    try:
        parsed = json.loads(output_text) if output_text is not None else None
    except json.JSONDecodeError:
        parsed = None
    output = _strict_output(parsed)
    if output is None:
        return {"status": "rejected", "reason": "structured_output_invalid", "custom_id": custom_id}
    return {"status": "accepted", "custom_id": custom_id, "output": output}


def _terminal_record(custom_id: str, events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not events:
        return {"status": "rejected", "reason": "missing_custom_id", "custom_id": custom_id}
    if len(events) > 1:
        return {"status": "rejected", "reason": "duplicate_custom_id", "custom_id": custom_id}
    event = events[0]
    row = event["row"]
    if event["kind"] == "error":
        return {
            "status": "rejected",
            "reason": "batch_error_row",
            "custom_id": custom_id,
            "error": _json_value(row.get("error")),
            "raw": _json_value(row),
        }
    return _response_record(custom_id, row)


def _ensure_collection_ready(state: Mapping[str, Any]) -> None:
    incomplete = [
        str(index)
        for index, shard in enumerate(state.get("shards", []))
        if shard.get("status") != "completed" or not (shard.get("output_file_id") or shard.get("error_file_id"))
    ]
    if incomplete:
        raise ValueError(f"collection is not ready for shards: {', '.join(incomplete)}")


def collect_completed_shards(state: Mapping[str, Any], client: Any, output_dir: str | Path) -> dict[str, Any]:
    """Reconcile every completed shard, then persist a complete strict collection once."""
    root = Path(output_dir)
    result = copy.deepcopy(dict(state))
    _validate_state_requests(result, root)
    finalized_counts = result.get("counts") if result.get("collected_sha256") else None
    refreshed = refresh_batch_status(result, client, root)
    _ensure_collection_ready(refreshed)
    collected_path = root / COLLECTED_FILE_NAME
    diagnostics_path = root / DIAGNOSTICS_FILE_NAME
    if refreshed.get("collected_sha256"):
        if not collected_path.exists() or _sha256_file(collected_path) != refreshed["collected_sha256"]:
            raise ValueError(f"collected artifact checksum mismatch: {collected_path}")
        if not diagnostics_path.exists() or _sha256_file(diagnostics_path) != refreshed.get("diagnostics_sha256"):
            raise ValueError(f"diagnostics artifact checksum mismatch: {diagnostics_path}")
        if finalized_counts is not None:
            refreshed["counts"] = finalized_counts
        return _persist_state(refreshed, root)

    all_ids = {str(custom_id) for shard in refreshed["shards"] for custom_id in shard["custom_ids"]}
    terminal_records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for shard_index, shard in enumerate(refreshed["shards"]):
        expected_ids = {str(custom_id) for custom_id in shard["custom_ids"]}
        events: dict[str, list[dict[str, Any]]] = {}
        for field in ("output_file_id", "error_file_id"):
            if not shard.get(field):
                continue
            content = _download_file(shard, field, client, root)
            refreshed = _persist_state(refreshed, root)
            diagnostics.extend(
                _parse_artifact_rows(
                    content,
                    shard_index=shard_index,
                    expected_ids=expected_ids,
                    all_ids=all_ids,
                    events=events,
                )
            )
        for error in _object_value(shard.get("batch_errors"), "data", shard.get("batch_errors", [])) or []:
            diagnostics.append(
                {
                    "reason": "top_level_batch_error",
                    "shard_index": shard_index,
                    "error": _json_value(error),
                }
            )
        for custom_id in sorted(expected_ids):
            terminal_records.append(_terminal_record(custom_id, events.get(custom_id, [])))
    encoded = b"".join((_canonical_json(record) + "\n").encode("utf-8") for record in terminal_records)
    diagnostics_encoded = b"".join((_canonical_json(record) + "\n").encode("utf-8") for record in diagnostics)
    _write_bytes_atomic(diagnostics_path, diagnostics_encoded)
    _write_bytes_atomic(collected_path, encoded)
    refreshed["collected_path"] = COLLECTED_FILE_NAME
    refreshed["collected_sha256"] = _sha256_bytes(encoded)
    refreshed["diagnostics_path"] = DIAGNOSTICS_FILE_NAME
    refreshed["diagnostics_sha256"] = _sha256_bytes(diagnostics_encoded)
    refreshed["counts"] = {
        "prepared": len(terminal_records),
        "accepted": sum(record["status"] == "accepted" for record in terminal_records),
        "rejected": sum(record["status"] == "rejected" for record in terminal_records),
        "diagnostics": len(diagnostics),
    }
    return _persist_state(refreshed, root)


__all__ = [
    "BatchShard",
    "COLLECTED_FILE_NAME",
    "DIAGNOSTICS_FILE_NAME",
    "REQUIRED_SOURCE_CORPUS_FINGERPRINT",
    "STATE_FILE_NAME",
    "collect_completed_shards",
    "prepare_generation_batches",
    "refresh_batch_status",
    "submit_pending_shards",
    "write_batch_shards",
]
