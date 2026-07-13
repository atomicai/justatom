from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from justatom.tooling.ir_dataset.generation import (
    GENERATOR_SCHEMA,
    GeneratorConfig,
    build_generator_request,
    normalize_query,
    validate_generator_result,
)


REQUIRED_SOURCE_CORPUS_FINGERPRINT = "bb6ad903b82c337a61cce2b1cd5bf5dd7e3303b6b3263258979372c00e40c3c9"
DEFAULT_MAX_REQUESTS = 1_000
DEFAULT_MAX_BYTES = 100_000_000
STATE_FILE_NAME = "generation_state.json"
COLLECTED_FILE_NAME = "generation_collected.jsonl"
DIAGNOSTICS_FILE_NAME = "generation_diagnostics.jsonl"
PILOT_METRICS_FILE_NAME = "pilot_metrics.json"
PILOT_MAX_REQUESTS = 100
PILOT_MIN_USABLE_RATE = 0.70
PILOT_MIN_GATE_PASS_RATE = 0.60
VALIDATOR_VERSION = 1
FINALIZER_VERSION = 1
RETRYABLE_BATCH_STATUSES = frozenset({"failed", "expired", "cancelled"})


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


def _write_parquet_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        pl.DataFrame(rows).write_parquet(temporary, compression="zstd")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


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
        "accepted_max_tokens": config.accepted_max_tokens,
        "validator_version": VALIDATOR_VERSION,
        "finalizer_version": FINALIZER_VERSION,
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
    _validate_generation_inputs(state, output_dir)
    seen_ids: set[str] = set()
    for shard in state.get("shards", []):
        if not isinstance(shard, Mapping):
            raise ValueError("invalid batch state shard")
        _validate_shard_checksum(shard, output_dir)
        for custom_id in shard.get("custom_ids", []):
            if not isinstance(custom_id, str) or custom_id in seen_ids:
                raise ValueError(f"duplicate custom_id in batch state: {custom_id}")
            seen_ids.add(custom_id)


def _input_artifact_path(state: Mapping[str, Any], output_dir: str | Path, field: str, default: str) -> Path:
    metadata = state.get(field)
    relative = metadata.get("path", default) if isinstance(metadata, Mapping) else default
    return Path(output_dir) / str(relative)


def _validate_generation_inputs(state: Mapping[str, Any], output_dir: str | Path) -> None:
    root = Path(output_dir)
    for field, default, label in (
        ("targets", "targets.parquet", "targets"),
        ("generation_context", "generation_context.parquet", "generation context"),
    ):
        metadata = state.get(field)
        if not isinstance(metadata, Mapping):
            continue
        path = root / str(metadata.get("path", default))
        expected = metadata.get("sha256")
        if not expected or not path.exists() or _sha256_file(path) != str(expected):
            raise ValueError(f"{label} artifact checksum mismatch: {path}")
        state_path = metadata.get("state_path")
        state_sha256 = metadata.get("state_sha256")
        if state_path or state_sha256:
            if not state_path or not state_sha256:
                raise ValueError(f"incomplete {label} artifact state binding")
            bound_state_path = root / str(state_path)
            if not bound_state_path.exists() or _sha256_file(bound_state_path) != str(state_sha256):
                raise ValueError(f"{label} artifact state checksum mismatch: {bound_state_path}")


def _ensure_input_artifact(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    expected_sha256: str | None,
    label: str,
) -> str:
    if path.exists():
        actual_sha256 = _sha256_file(path)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError(f"{label} artifact checksum mismatch: {path}")
        try:
            persisted_rows = pl.read_parquet(path).to_dicts()
        except Exception as exc:
            raise ValueError(f"invalid {label} artifact: {path}") from exc
        if _canonical_json(persisted_rows) != _canonical_json(rows):
            raise ValueError(f"refusing to reuse {label} artifact with different rows: {path}")
        return actual_sha256
    if expected_sha256 is not None:
        raise FileNotFoundError(f"checksummed {label} artifact does not exist: {path}")
    _write_parquet_atomic(path, rows)
    return _sha256_file(path)


def prepare_generation_batches(
    targets: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]] | Any,
    generation_context: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]] | Any,
    config: GeneratorConfig | Mapping[str, Any] | None,
    output_dir: str | Path,
    *,
    source_corpus_fingerprint: str,
    source_passages_sha256: str | None = None,
    targets_sha256: str | None = None,
    generation_context_sha256: str | None = None,
    targets_state_sha256: str | None = None,
    generation_context_state_sha256: str | None = None,
    pilot_generation_root: str | Path | None = None,
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
    root = Path(output_dir)
    if len(requests) > PILOT_MAX_REQUESTS and pilot_generation_root is not None:
        if root.resolve() == Path(pilot_generation_root).resolve():
            raise ValueError("preparing more than 100 requests requires a separate generation root")
    targets_path = root / "targets.parquet"
    context_path = root / "generation_context.parquet"
    target_artifact_sha256 = _ensure_input_artifact(
        target_rows,
        targets_path,
        expected_sha256=targets_sha256,
        label="targets",
    )
    context_artifact_sha256 = _ensure_input_artifact(
        context_rows,
        context_path,
        expected_sha256=generation_context_sha256,
        label="generation context",
    )
    context_fingerprint = _target_context_fingerprint(target_rows, context_rows)
    generation_fingerprint = _generation_fingerprint(source_corpus_fingerprint, context_fingerprint, active_config, requests)
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
        "version": 3,
        "source_corpus_fingerprint": source_corpus_fingerprint,
        "source_passages_sha256": source_passages_sha256,
        "targets": {
            "path": _relative_to(targets_path, root),
            "sha256": target_artifact_sha256,
            "state_path": "targets_state.json" if targets_state_sha256 else None,
            "state_sha256": targets_state_sha256,
        },
        "generation_context": {
            "path": _relative_to(context_path, root),
            "sha256": context_artifact_sha256,
            "state_path": "generation_context_state.json" if generation_context_state_sha256 else None,
            "state_sha256": generation_context_state_sha256,
        },
        "target_context_fingerprint": context_fingerprint,
        "generation_fingerprint": generation_fingerprint,
        "generation_config": asdict(active_config),
        "limits": {
            "model": active_config.model,
            "max_requests_per_shard": active_config.max_requests_per_shard,
            "max_shard_bytes": active_config.max_shard_bytes,
            "max_batch_attempts": active_config.max_batch_attempts,
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
                "attempt": 1,
                "history": [],
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
        "justatom_batch_attempt": str(shard.get("attempt", 1)),
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


def _request_count(state: Mapping[str, Any]) -> int:
    return sum(int(shard.get("request_count", len(shard.get("custom_ids", ())))) for shard in state.get("shards", []))


def _requires_scale_gate(state: Mapping[str, Any]) -> bool:
    return _request_count(state) > PILOT_MAX_REQUESTS or len(state.get("shards", ())) > 1


def _validate_scale_gate(
    state: Mapping[str, Any],
    output_dir: str | Path,
    *,
    scale_authorized: bool,
    pilot_generation_root: str | Path | None,
) -> None:
    if not _requires_scale_gate(state):
        return
    if not scale_authorized:
        raise ValueError("multi-shard or >100-request submission requires explicit scale authorization")
    if pilot_generation_root is None:
        raise ValueError("scale submission requires a pilot generation root")
    root = Path(output_dir).resolve()
    pilot_root = Path(pilot_generation_root).resolve()
    if root == pilot_root:
        raise ValueError("scale submission requires a generation root separate from the pilot workspace")
    metrics_path = pilot_root / PILOT_METRICS_FILE_NAME
    pilot_state_path = pilot_root / STATE_FILE_NAME
    if not metrics_path.exists() or not pilot_state_path.exists():
        raise ValueError("scale submission requires checksummed pilot metrics and pilot state artifacts")
    pilot_state = _load_state(pilot_state_path)
    expected_checksum = pilot_state.get("pilot_metrics_sha256")
    if not expected_checksum or _sha256_file(metrics_path) != str(expected_checksum):
        raise ValueError("pilot metrics checksum mismatch")
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("invalid pilot metrics artifact") from exc
    if not isinstance(metrics, Mapping) or metrics.get("generation_fingerprint") != pilot_state.get("generation_fingerprint"):
        raise ValueError("pilot metrics are not bound to the pilot generation fingerprint")
    if pilot_state.get("validator_version") != VALIDATOR_VERSION or pilot_state.get("finalizer_version") != FINALIZER_VERSION:
        raise ValueError("pilot metrics were not produced by the current deterministic finalizer")
    if pilot_state.get("source_corpus_fingerprint") != state.get("source_corpus_fingerprint"):
        raise ValueError("pilot metrics source corpus does not match the scale generation")
    pilot_passages_sha256 = pilot_state.get("source_passages_sha256")
    if pilot_passages_sha256 and pilot_passages_sha256 != state.get("source_passages_sha256"):
        raise ValueError("pilot metrics source passages do not match the scale generation")
    if int(metrics.get("request_count", PILOT_MAX_REQUESTS + 1)) > PILOT_MAX_REQUESTS:
        raise ValueError("pilot metrics must describe at most 100 requests")
    if float(metrics.get("usable_rate", -1.0)) < PILOT_MIN_USABLE_RATE:
        raise ValueError(f"pilot usable rate must be >= {PILOT_MIN_USABLE_RATE:.2f}")
    if float(metrics.get("deterministic_gate_pass_rate", -1.0)) < PILOT_MIN_GATE_PASS_RATE:
        raise ValueError(f"pilot deterministic gate pass rate must be >= {PILOT_MIN_GATE_PASS_RATE:.2f}")


def submit_pending_shards(
    state: Mapping[str, Any],
    client: Any,
    output_dir: str | Path,
    *,
    scale_authorized: bool = False,
    pilot_generation_root: str | Path | None = None,
) -> dict[str, Any]:
    """Serialize local submit/reconcile transitions; this does not provide cross-machine locking."""
    _validate_state_requests(state, output_dir)
    _validate_scale_gate(
        state,
        output_dir,
        scale_authorized=scale_authorized,
        pilot_generation_root=pilot_generation_root,
    )
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


def _attempt_history_entry(shard: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "attempt",
        "batch_id",
        "status",
        "output_file_id",
        "error_file_id",
        "output_path",
        "output_sha256",
        "error_path",
        "error_sha256",
        "batch_errors",
        "batch_operation",
    )
    return {field: copy.deepcopy(shard[field]) for field in fields if field in shard}


def retry_failed_shards(
    state: Mapping[str, Any],
    client: Any,
    output_dir: str | Path,
    *,
    scale_authorized: bool = False,
    pilot_generation_root: str | Path | None = None,
) -> dict[str, Any]:
    """Retry failed terminal shards explicitly while retaining every prior remote attempt."""
    _validate_state_requests(state, output_dir)
    _validate_scale_gate(
        state,
        output_dir,
        scale_authorized=scale_authorized,
        pilot_generation_root=pilot_generation_root,
    )
    with _submit_lock(output_dir):
        refreshed = refresh_batch_status(state, client, output_dir)
        eligible = [shard for shard in refreshed.get("shards", []) if shard.get("status") in RETRYABLE_BATCH_STATUSES]
        if not eligible:
            raise ValueError("no failed, expired, or cancelled shards are eligible for retry")
        max_attempts = int(
            refreshed.get("generation_config", {}).get(
                "max_batch_attempts", refreshed.get("limits", {}).get("max_batch_attempts", 2)
            )
        )
        capped = [shard for shard in eligible if int(shard.get("attempt", 1)) >= max_attempts]
        if capped:
            raise ValueError(f"batch attempt cap of {max_attempts} reached for retryable shard")
        for shard in eligible:
            shard.setdefault("history", []).append(_attempt_history_entry(shard))
            shard["attempt"] = int(shard.get("attempt", 1)) + 1
            for field in (
                "batch_id",
                "batch_operation",
                "output_file_id",
                "error_file_id",
                "output_path",
                "output_sha256",
                "error_path",
                "error_sha256",
                "batch_errors",
            ):
                shard.pop(field, None)
            shard["status"] = "uploaded" if shard.get("input_file_id") else "prepared"
        refreshed.pop("collected_path", None)
        refreshed.pop("collected_sha256", None)
        refreshed.pop("diagnostics_path", None)
        refreshed.pop("diagnostics_sha256", None)
        refreshed.pop("pilot_metrics_path", None)
        refreshed.pop("pilot_metrics_sha256", None)
        refreshed = _persist_state(refreshed, output_dir)
        return _submit_pending_shards_unlocked(refreshed, client, output_dir)


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
    return {
        "status": "parsed",
        "custom_id": custom_id,
        "output": output,
        "usage": _json_value(_object_value(body, "usage", {})),
    }


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


def _generator_config_from_state(state: Mapping[str, Any]) -> GeneratorConfig:
    persisted = state.get("generation_config")
    if isinstance(persisted, Mapping):
        return GeneratorConfig(**dict(persisted))
    limits = state.get("limits") if isinstance(state.get("limits"), Mapping) else {}
    supported = {
        key: limits[key] for key in ("model", "max_requests_per_shard", "max_shard_bytes", "max_batch_attempts") if key in limits
    }
    return GeneratorConfig(**supported)


def _load_request_bindings(
    state: Mapping[str, Any], output_dir: str | Path
) -> tuple[list[str], dict[str, tuple[dict[str, Any], list[dict[str, Any]]]], GeneratorConfig]:
    root = Path(output_dir)
    targets_path = _input_artifact_path(state, root, "targets", "targets.parquet")
    context_path = _input_artifact_path(state, root, "generation_context", "generation_context.parquet")
    if not targets_path.exists() or not context_path.exists():
        raise FileNotFoundError("generation target/context artifacts are required for deterministic collection")
    try:
        target_rows = pl.read_parquet(targets_path).to_dicts()
        context_rows = pl.read_parquet(context_path).to_dicts()
    except Exception as exc:
        raise ValueError("invalid generation target/context artifact") from exc
    legacy_fingerprint = state.get("target_context_fingerprint")
    if legacy_fingerprint and _target_context_fingerprint(target_rows, context_rows) != legacy_fingerprint:
        raise ValueError("generation target/context fingerprint mismatch")
    context_by_target: dict[str, list[dict[str, Any]]] = {}
    for row in context_rows:
        context_by_target.setdefault(str(row.get("target_passage_id", "")), []).append(row)
    config = _generator_config_from_state(state)
    state_ids = [str(custom_id) for shard in state.get("shards", []) for custom_id in shard.get("custom_ids", [])]
    state_id_set = set(state_ids)
    bindings: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    ordered_ids: list[str] = []
    for target in target_rows:
        target_id = str(target.get("passage_id", ""))
        target_context = context_by_target.get(target_id, [])
        request = build_generator_request(target, target_context, config)
        custom_id = str(request["custom_id"])
        if custom_id not in state_id_set:
            raise ValueError(f"batch state custom IDs do not match immutable generation inputs: {target_id}")
        bindings[custom_id] = (target, target_context)
        ordered_ids.append(custom_id)
    if len(bindings) != len(state_ids) or set(bindings) != state_id_set:
        raise ValueError("batch state request mapping is not complete for immutable generation inputs")
    return ordered_ids, bindings, config


def _finalize_record(
    custom_id: str,
    events: Sequence[Mapping[str, Any]],
    slot: Mapping[str, Any],
    generation_context: Sequence[Mapping[str, Any]],
    accepted_normalized_queries: set[str],
    config: GeneratorConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = _terminal_record(custom_id, events)
    output = parsed.get("output") if isinstance(parsed.get("output"), Mapping) else {}
    validation = validate_generator_result(
        output,
        slot,
        generation_context,
        accepted_normalized_queries=accepted_normalized_queries,
        config=config,
    )
    reason_codes: list[str] = []
    if parsed["status"] == "rejected":
        reason_codes.append(str(parsed["reason"]))
    reason_codes.extend(validation.reason_codes)
    reason_codes = list(dict.fromkeys(reason_codes))
    accepted = parsed["status"] == "parsed" and validation.accepted
    record = {key: value for key, value in parsed.items() if key not in {"usage", "status"}}
    record["status"] = "accepted" if accepted else "rejected"
    record["reason_codes"] = reason_codes
    if not accepted and "reason" not in record:
        record["reason"] = reason_codes[0] if reason_codes else "deterministic_gate_failed"
    if accepted:
        accepted_normalized_queries.add(normalize_query(output["query"]))
    evidence_invalid = {"empty_evidence", "evidence_not_substring", "evidence_overlap_only"}
    observation = {
        "schema_success": parsed["status"] == "parsed",
        "usable": bool(output.get("usable")) if output else False,
        "gate_pass": accepted,
        "intent_agreement": bool(output)
        and output.get("requested_intent") == slot.get("requested_intent")
        and output.get("actual_intent") == output.get("requested_intent"),
        "evidence_valid": bool(output) and not evidence_invalid.intersection(reason_codes),
        "duplicate": "duplicate_normalized_query" in reason_codes,
        "usage": parsed.get("usage") if isinstance(parsed.get("usage"), Mapping) else {},
    }
    return record, observation


def _pilot_metrics(state: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(observations)
    denominator = max(count, 1)
    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for observation in observations:
        usage = observation.get("usage") if isinstance(observation.get("usage"), Mapping) else {}
        for key in token_usage:
            value = usage.get(key, 0)
            if isinstance(value, int) and not isinstance(value, bool):
                token_usage[key] += value
    return {
        "version": 1,
        "generation_fingerprint": state["generation_fingerprint"],
        "request_count": count,
        "schema_success_count": sum(bool(item["schema_success"]) for item in observations),
        "schema_success_rate": sum(bool(item["schema_success"]) for item in observations) / denominator,
        "usable_count": sum(bool(item["usable"]) for item in observations),
        "usable_rate": sum(bool(item["usable"]) for item in observations) / denominator,
        "deterministic_gate_pass_count": sum(bool(item["gate_pass"]) for item in observations),
        "deterministic_gate_pass_rate": sum(bool(item["gate_pass"]) for item in observations) / denominator,
        "intent_agreement_count": sum(bool(item["intent_agreement"]) for item in observations),
        "intent_agreement_rate": sum(bool(item["intent_agreement"]) for item in observations) / denominator,
        "evidence_valid_count": sum(bool(item["evidence_valid"]) for item in observations),
        "evidence_validity_rate": sum(bool(item["evidence_valid"]) for item in observations) / denominator,
        "duplicate_count": sum(bool(item["duplicate"]) for item in observations),
        "token_usage": token_usage,
    }


def _prior_attempt_diagnostics(shard: dict[str, Any], client: Any, root: Path, shard_index: int) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for history in shard.get("history", []):
        if not isinstance(history, dict):
            continue
        for field in ("output_file_id", "error_file_id"):
            if not history.get(field):
                continue
            content = _download_file(history, field, client, root)
            for raw_line in content.decode("utf-8", errors="replace").splitlines():
                try:
                    raw: Any = json.loads(raw_line)
                except json.JSONDecodeError:
                    raw = raw_line
                diagnostics.append(
                    {
                        "reason": "prior_attempt_artifact",
                        "shard_index": shard_index,
                        "attempt": history.get("attempt", 1),
                        "status": history.get("status"),
                        "artifact_field": field,
                        "raw": _json_value(raw),
                    }
                )
    return diagnostics


def collect_completed_shards(state: Mapping[str, Any], client: Any, output_dir: str | Path) -> dict[str, Any]:
    """Reconcile raw artifacts and deterministically finalize one row per immutable request binding."""
    root = Path(output_dir)
    result = copy.deepcopy(dict(state))
    _validate_state_requests(result, root)
    finalized_counts = (
        copy.deepcopy(result.get("counts"))
        if result.get("collected_sha256")
        and result.get("validator_version") == VALIDATOR_VERSION
        and result.get("finalizer_version") == FINALIZER_VERSION
        else None
    )
    refreshed = refresh_batch_status(result, client, root)
    _ensure_collection_ready(refreshed)
    collected_path = root / COLLECTED_FILE_NAME
    diagnostics_path = root / DIAGNOSTICS_FILE_NAME
    if (
        refreshed.get("collected_sha256")
        and refreshed.get("validator_version") == VALIDATOR_VERSION
        and refreshed.get("finalizer_version") == FINALIZER_VERSION
    ):
        if not collected_path.exists() or _sha256_file(collected_path) != refreshed["collected_sha256"]:
            raise ValueError(f"collected artifact checksum mismatch: {collected_path}")
        if not diagnostics_path.exists() or _sha256_file(diagnostics_path) != refreshed.get("diagnostics_sha256"):
            raise ValueError(f"diagnostics artifact checksum mismatch: {diagnostics_path}")
        if _request_count(refreshed) <= PILOT_MAX_REQUESTS:
            metrics_path = root / PILOT_METRICS_FILE_NAME
            if not metrics_path.exists() or _sha256_file(metrics_path) != refreshed.get("pilot_metrics_sha256"):
                raise ValueError(f"pilot metrics artifact checksum mismatch: {metrics_path}")
        if finalized_counts is not None:
            refreshed["counts"] = finalized_counts
        return _persist_state(refreshed, root)

    all_ids = {str(custom_id) for shard in refreshed["shards"] for custom_id in shard["custom_ids"]}
    ordered_ids, bindings, generation_config = _load_request_bindings(refreshed, root)
    events_by_id: dict[str, list[dict[str, Any]]] = {}
    diagnostics: list[dict[str, Any]] = []
    for shard_index, shard in enumerate(refreshed["shards"]):
        expected_ids = {str(custom_id) for custom_id in shard["custom_ids"]}
        events: dict[str, list[dict[str, Any]]] = {}
        diagnostics.extend(_prior_attempt_diagnostics(shard, client, root, shard_index))
        refreshed = _persist_state(refreshed, root)
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
        for custom_id in expected_ids:
            events_by_id[custom_id] = events.get(custom_id, [])
    accepted_normalized_queries: set[str] = set()
    terminal_records: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for custom_id in ordered_ids:
        slot, generation_context = bindings[custom_id]
        record, observation = _finalize_record(
            custom_id,
            events_by_id.get(custom_id, []),
            slot,
            generation_context,
            accepted_normalized_queries,
            generation_config,
        )
        terminal_records.append(record)
        observations.append(observation)
    encoded = b"".join((_canonical_json(record) + "\n").encode("utf-8") for record in terminal_records)
    diagnostics_encoded = b"".join((_canonical_json(record) + "\n").encode("utf-8") for record in diagnostics)
    _write_bytes_atomic(diagnostics_path, diagnostics_encoded)
    _write_bytes_atomic(collected_path, encoded)
    refreshed["collected_path"] = COLLECTED_FILE_NAME
    refreshed["collected_sha256"] = _sha256_bytes(encoded)
    refreshed["diagnostics_path"] = DIAGNOSTICS_FILE_NAME
    refreshed["diagnostics_sha256"] = _sha256_bytes(diagnostics_encoded)
    refreshed["validator_version"] = VALIDATOR_VERSION
    refreshed["finalizer_version"] = FINALIZER_VERSION
    refreshed["version"] = max(int(refreshed.get("version", 2)), 3)
    refreshed["counts"] = {
        "prepared": len(terminal_records),
        "accepted": sum(record["status"] == "accepted" for record in terminal_records),
        "rejected": sum(record["status"] == "rejected" for record in terminal_records),
        "diagnostics": len(diagnostics),
    }
    if len(terminal_records) <= PILOT_MAX_REQUESTS:
        metrics_path = root / PILOT_METRICS_FILE_NAME
        metrics = _pilot_metrics(refreshed, observations)
        _write_json_atomic(metrics_path, metrics)
        refreshed["pilot_metrics_path"] = PILOT_METRICS_FILE_NAME
        refreshed["pilot_metrics_sha256"] = _sha256_file(metrics_path)
    return _persist_state(refreshed, root)


__all__ = [
    "BatchShard",
    "COLLECTED_FILE_NAME",
    "DIAGNOSTICS_FILE_NAME",
    "PILOT_METRICS_FILE_NAME",
    "REQUIRED_SOURCE_CORPUS_FINGERPRINT",
    "STATE_FILE_NAME",
    "collect_completed_shards",
    "prepare_generation_batches",
    "refresh_batch_status",
    "retry_failed_shards",
    "submit_pending_shards",
    "write_batch_shards",
]
