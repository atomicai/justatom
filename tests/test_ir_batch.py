from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

from justatom.tooling.ir_dataset import batch as batch_module
from justatom.tooling.ir_dataset.artifacts import sha256_file, write_bound_parquet_artifact
from justatom.tooling.ir_dataset.batch import (
    collect_completed_shards,
    prepare_generation_batches as prepare_generation_batches_core,
    submit_pending_shards,
    write_batch_shards,
)
from justatom.tooling.ir_dataset.generation import GeneratorConfig


SOURCE_CORPUS_FINGERPRINT = "bb6ad903b82c337a61cce2b1cd5bf5dd7e3303b6b3263258979372c00e40c3c9"
SOURCE_PASSAGES_SHA256 = "a" * 64


def request(index: int) -> dict[str, object]:
    return {
        "custom_id": f"request-{index}",
        "method": "POST",
        "url": "/v1/responses",
        "body": {"model": "test-model", "metadata": {"source_hash": f"source-{index}"}},
    }


def requests(count: int) -> list[dict[str, object]]:
    return [request(index) for index in range(count)]


def target(index: int = 0) -> dict[str, object]:
    return {
        "article_id": f"article-{index}",
        "passage_id": f"passage-{index}",
        "source_hash": f"source-{index}",
        "content": "Redis повторяет запросы после сетевой ошибки в production режиме.",
        "serialized_passage": "passage: Redis\n\nRedis повторяет запросы после сетевой ошибки в production режиме.",
        "token_count": 12,
        "requested_intent": "how_to",
    }


def context(row: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "target_article_id": row["article_id"],
            "target_passage_id": row["passage_id"],
            "target_source_hash": row["source_hash"],
            "candidate_passage_id": f"neighbor-{index}",
            "candidate_serialized_passage": f"passage: Neighbor {index}\n\nКонтекст {index}.",
            "context_index": index,
        }
        for index in range(3)
    ]


@dataclass
class FakeObject:
    id: str
    status: str = "completed"
    output_file_id: str | None = None
    error_file_id: str | None = None
    filename: str | None = None
    bytes: int | None = None
    input_file_id: str | None = None
    metadata: dict[str, str] | None = None
    errors: object | None = None
    purpose: str | None = None


class FakeFiles:
    def __init__(self, owner: "FakeOpenAI") -> None:
        self.owner = owner

    def create(self, *, file, purpose: str):
        assert purpose == "batch"
        name, content = Path(file.name).name, file.read()
        self.owner.uploads.append((name, content))
        item = FakeObject(id=f"file-{len(self.owner.uploads)}", filename=name, bytes=len(content), purpose=purpose)
        self.owner.file_objects.append(item)
        return item

    def list(self, **_kwargs):
        self.owner.file_lists += 1
        return self.owner.file_objects

    def content(self, file_id: str):
        self.owner.downloads.append(file_id)
        return self.owner.file_contents[file_id]


class FakeBatches:
    def __init__(self, owner: "FakeOpenAI") -> None:
        self.owner = owner

    def create(self, *, input_file_id: str, endpoint: str, completion_window: str, metadata: dict[str, str]):
        assert endpoint == "/v1/responses"
        assert completion_window == "24h"
        self.owner.creates.append(input_file_id)
        item = FakeObject(
            id=f"batch-{len(self.owner.creates)}",
            input_file_id=input_file_id,
            metadata=metadata,
        )
        self.owner.batch_objects[item.id] = item
        return item

    def list(self, **_kwargs):
        self.owner.batch_lists += 1
        return list(self.owner.batch_objects.values())

    def retrieve(self, batch_id: str):
        self.owner.retrieves.append(batch_id)
        return self.owner.batch_objects[batch_id]


class FakeOpenAI:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, bytes]] = []
        self.file_lists = 0
        self.file_objects: list[FakeObject] = []
        self.creates: list[str] = []
        self.batch_lists = 0
        self.retrieves: list[str] = []
        self.downloads: list[str] = []
        self.file_contents: dict[str, object] = {}
        self.batch_objects: dict[str, FakeObject] = {}
        self.files = FakeFiles(self)
        self.batches = FakeBatches(self)


def accepted_output(query: str = "Как Redis повторяет запросы после сетевой ошибки в production режиме?") -> dict[str, object]:
    return {
        "usable": True,
        "reason": "ok",
        "query": query,
        "answer": "Redis повторяет запросы после сетевой ошибки.",
        "evidence": "Redis повторяет запросы после сетевой ошибки",
        "requested_intent": "how_to",
        "actual_intent": "how_to",
        "disambiguators": ["Redis", "production"],
    }


def response_row(custom_id: str, output: dict[str, object], *, usage: dict[str, int] | None = None) -> dict[str, object]:
    body: dict[str, object] = {"output_text": json.dumps(output)}
    if usage is not None:
        body["usage"] = usage
    return {"custom_id": custom_id, "response": {"status_code": 200, "body": body}}


def prepare_generation_batches(
    targets,
    generation_context,
    config,
    output_dir,
    *,
    source_corpus_fingerprint,
    pilot_generation_root=None,
):
    root = Path(output_dir)
    target_rows = targets.to_dicts() if hasattr(targets, "to_dicts") else list(targets)
    context_rows = generation_context.to_dicts() if hasattr(generation_context, "to_dicts") else list(generation_context)
    write_bound_parquet_artifact(
        pl.DataFrame(target_rows),
        root / "targets.parquet",
        root / "targets_state.json",
        artifact_kind="targets",
        source_corpus_fingerprint=source_corpus_fingerprint,
        passages_sha256=SOURCE_PASSAGES_SHA256,
        config={"test_contract": 1},
        upstream_sha256="b" * 64,
    )
    write_bound_parquet_artifact(
        pl.DataFrame(context_rows),
        root / "generation_context.parquet",
        root / "generation_context_state.json",
        artifact_kind="generation_context",
        source_corpus_fingerprint=source_corpus_fingerprint,
        passages_sha256=SOURCE_PASSAGES_SHA256,
        config={"test_contract": 1},
        upstream_sha256=sha256_file(root / "targets.parquet"),
    )
    return prepare_generation_batches_core(
        target_rows,
        context_rows,
        config,
        root,
        source_corpus_fingerprint=source_corpus_fingerprint,
        source_passages_sha256=SOURCE_PASSAGES_SHA256,
        targets_sha256=sha256_file(root / "targets.parquet"),
        generation_context_sha256=sha256_file(root / "generation_context.parquet"),
        targets_state_sha256=sha256_file(root / "targets_state.json"),
        generation_context_state_sha256=sha256_file(root / "generation_context_state.json"),
        pilot_generation_root=pilot_generation_root,
    )


def passing_pilot_root(root: Path) -> Path:
    pilot_root = root / "pilot"
    pilot_target = target(999_999)
    pilot_state = prepare_generation_batches(
        [pilot_target],
        context(pilot_target),
        GeneratorConfig(),
        pilot_root,
        source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT,
    )
    metrics = {
        "version": 1,
        "generation_fingerprint": pilot_state["generation_fingerprint"],
        "request_count": 1,
        "usable_rate": 0.70,
        "deterministic_gate_pass_rate": 0.60,
        "source_corpus_fingerprint": SOURCE_CORPUS_FINGERPRINT,
        "source_passages_sha256": SOURCE_PASSAGES_SHA256,
        "validator_version": batch_module.VALIDATOR_VERSION,
        "finalizer_version": batch_module.FINALIZER_VERSION,
    }
    batch_module._write_json_atomic(pilot_root / "pilot_metrics.json", metrics)
    pilot_state["pilot_metrics_sha256"] = batch_module._sha256_file(pilot_root / "pilot_metrics.json")
    batch_module._write_json_atomic(
        pilot_root / "generation_state.json",
        pilot_state,
    )
    return pilot_root


def test_shards_respect_caps(tmp_path: Path):
    shards = write_batch_shards(requests(1001), tmp_path, max_requests=1000, max_bytes=100_000_000)

    assert [item.request_count for item in shards] == [1000, 1]
    assert all(item.byte_count <= 100_000_000 for item in shards)
    assert all(item.path.exists() for item in shards)


def test_shards_reject_duplicate_custom_ids(tmp_path: Path):
    duplicate = requests(2)
    duplicate[1]["custom_id"] = duplicate[0]["custom_id"]

    with pytest.raises(ValueError, match="duplicate custom_id"):
        write_batch_shards(duplicate, tmp_path)


def test_prepare_rejects_a_generation_run_from_an_unapproved_source_corpus(tmp_path: Path):
    row = target()

    with pytest.raises(ValueError, match="source corpus manifest fingerprint"):
        prepare_generation_batches([row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint="different-corpus")


def test_submit_is_idempotent(tmp_path: Path):
    state = prepare_generation_batches(
        [target()], context(target()), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )
    client = FakeOpenAI()

    first = submit_pending_shards(state, client, tmp_path)
    second_client = FakeOpenAI()
    second = submit_pending_shards(first, second_client, tmp_path)

    assert second == first
    assert len(client.uploads) == 1
    assert client.creates == ["file-1"]
    assert second_client.uploads == []
    assert second_client.creates == []
    assert first["shards"][0]["input_file_id"] == "file-1"
    assert first["shards"][0]["batch_id"] == "batch-1"


def test_submit_refuses_a_request_checksum_mismatch(tmp_path: Path):
    state = prepare_generation_batches(
        [target()], context(target()), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )
    request_path = tmp_path / state["shards"][0]["request_path"]
    request_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        submit_pending_shards(state, FakeOpenAI(), tmp_path)


def test_collection_accepts_only_http_200_strict_output_text_and_keeps_rejections(tmp_path: Path):
    row = target()
    state = prepare_generation_batches(
        [row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )
    client = FakeOpenAI()
    submitted = submit_pending_shards(state, client, tmp_path)
    custom_id = submitted["shards"][0]["custom_ids"][0]
    output = {
        "usable": True,
        "reason": "ok",
        "query": "Как Redis повторяет запросы после сетевой ошибки в production режиме?",
        "answer": "Redis повторяет запросы после сетевой ошибки.",
        "evidence": "Redis повторяет запросы после сетевой ошибки",
        "requested_intent": "how_to",
        "actual_intent": "how_to",
        "disambiguators": ["Redis", "production"],
    }
    client.batch_objects["batch-1"] = FakeObject(
        id="batch-1", output_file_id="output-1", error_file_id="error-1", errors={"data": [{"code": "batch_error"}]}
    )
    client.file_contents["output-1"] = (
        "\n".join(
            (
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "response": {
                            "status_code": 200,
                            "body": {
                                "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(output)}]}]
                            },
                        },
                    }
                ),
                json.dumps({"custom_id": "unknown", "response": {"status_code": 500, "body": {}}}),
            )
        )
        + "\n"
    )
    client.file_contents["error-1"] = "{not json}\n"

    collected = collect_completed_shards(submitted, client, tmp_path)

    records = [json.loads(line) for line in (tmp_path / "generation_collected.jsonl").read_text(encoding="utf-8").splitlines()]
    diagnostics = [
        json.loads(line) for line in (tmp_path / "generation_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert collected["counts"] == {"prepared": 1, "accepted": 1, "rejected": 0, "diagnostics": 3}
    assert records == [{"custom_id": custom_id, "output": output, "reason_codes": [], "status": "accepted"}]
    assert {record["reason"] for record in diagnostics} == {
        "unknown_custom_id",
        "malformed_batch_row",
        "top_level_batch_error",
    }
    assert collected["shards"][0]["batch_errors"] == {"data": [{"code": "batch_error"}]}
    assert client.downloads == ["output-1", "error-1"]
    assert collect_completed_shards(collected, client, tmp_path) == collected
    assert client.downloads == ["output-1", "error-1"]


@pytest.mark.parametrize("failed_field", ("input_file_id", "batch_id"))
def test_submit_reconciles_remote_success_after_local_state_persistence_failure(tmp_path: Path, monkeypatch, failed_field: str):
    row = target()
    state = prepare_generation_batches(
        [row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )
    client = FakeOpenAI()
    persist = batch_module._persist_state

    def fail_after_remote(value, output_dir):
        if value["shards"][0].get(failed_field):
            raise OSError("simulated local state persistence failure")
        return persist(value, output_dir)

    monkeypatch.setattr(batch_module, "_persist_state", fail_after_remote)
    with pytest.raises(OSError, match="simulated"):
        submit_pending_shards(state, client, tmp_path)
    monkeypatch.setattr(batch_module, "_persist_state", persist)

    retried_state = prepare_generation_batches(
        [row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )
    retried = submit_pending_shards(retried_state, client, tmp_path)

    assert len(client.uploads) == 1
    assert len(client.creates) == 1
    assert retried["shards"][0]["input_file_id"] == "file-1"
    assert retried["shards"][0]["batch_id"] == "batch-1"


def test_collection_requires_completed_shards_to_expose_at_least_one_artifact(tmp_path: Path):
    row = target()
    state = prepare_generation_batches(
        [row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )
    client = FakeOpenAI()
    submitted = submit_pending_shards(state, client, tmp_path)
    client.batch_objects["batch-1"] = FakeObject(id="batch-1", status="completed")

    with pytest.raises(ValueError, match="collection is not ready"):
        collect_completed_shards(submitted, client, tmp_path)

    assert not (tmp_path / "generation_collected.jsonl").exists()


@pytest.mark.parametrize("output_file_id,error_file_id", (("output-1", None), (None, "error-1")))
def test_collection_accepts_completed_shards_with_only_one_artifact(
    tmp_path: Path, output_file_id: str | None, error_file_id: str | None
):
    row = target()
    state = prepare_generation_batches(
        [row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )
    client = FakeOpenAI()
    submitted = submit_pending_shards(state, client, tmp_path)
    custom_id = submitted["shards"][0]["custom_ids"][0]
    client.batch_objects["batch-1"] = FakeObject(id="batch-1", output_file_id=output_file_id, error_file_id=error_file_id)
    if output_file_id:
        client.file_contents[output_file_id] = json.dumps(
            {
                "custom_id": custom_id,
                "response": {
                    "status_code": 200,
                    "body": {
                        "output_text": json.dumps(
                            {
                                "usable": False,
                                "reason": "insufficient_context",
                                "query": "",
                                "answer": "",
                                "evidence": "",
                                "requested_intent": "how_to",
                                "actual_intent": "how_to",
                                "disambiguators": [],
                            }
                        )
                    },
                },
            }
        )
    if error_file_id:
        client.file_contents[error_file_id] = json.dumps({"custom_id": custom_id, "error": {"code": "failed"}})

    collected = collect_completed_shards(submitted, client, tmp_path)

    assert collected["counts"] == {
        "prepared": 1,
        "accepted": 0,
        "rejected": 1,
        "diagnostics": 0,
    }


def test_collection_enforces_per_shard_unique_complete_mapping_and_preserves_error_rows(tmp_path: Path):
    rows = [target(index) for index in range(3)]
    contexts = [item for row in rows for item in context(row)]
    state = prepare_generation_batches(
        rows,
        contexts,
        GeneratorConfig(max_requests_per_shard=1),
        tmp_path,
        source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT,
    )
    client = FakeOpenAI()
    submitted = submit_pending_shards(
        state,
        client,
        tmp_path,
        scale_authorized=True,
        pilot_generation_root=passing_pilot_root(tmp_path),
    )
    custom_ids = [shard["custom_ids"][0] for shard in submitted["shards"]]
    output = {
        "usable": True,
        "reason": "ok",
        "query": "Как Redis повторяет запросы после сетевой ошибки в production режиме?",
        "answer": "Redis повторяет запросы после сетевой ошибки.",
        "evidence": "Redis повторяет запросы после сетевой ошибки",
        "requested_intent": "how_to",
        "actual_intent": "how_to",
        "disambiguators": ["Redis", "production"],
    }

    def response(custom_id: str):
        return {"custom_id": custom_id, "response": {"status_code": 200, "body": {"output_text": json.dumps(output)}}}

    for index, shard in enumerate(submitted["shards"], start=1):
        client.batch_objects[f"batch-{index}"] = FakeObject(
            id=f"batch-{index}", output_file_id=f"output-{index}", error_file_id=f"error-{index}"
        )
    client.file_contents["output-1"] = "\n".join((json.dumps(response(custom_ids[0])), json.dumps(response(custom_ids[0]))))
    client.file_contents["error-1"] = json.dumps({"custom_id": custom_ids[0], "error": {"code": "diagnostic"}})
    client.file_contents["output-2"] = "\n".join((json.dumps(response(custom_ids[0])), json.dumps(response(custom_ids[1]))))
    client.file_contents["error-2"] = json.dumps({"custom_id": custom_ids[1], "error": {"code": "diagnostic"}})
    client.file_contents["output-3"] = json.dumps(response("unknown"))
    client.file_contents["error-3"] = "\n"

    collected = collect_completed_shards(submitted, client, tmp_path)
    records = [json.loads(line) for line in (tmp_path / "generation_collected.jsonl").read_text(encoding="utf-8").splitlines()]
    diagnostics = [
        json.loads(line) for line in (tmp_path / "generation_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(records) == len(custom_ids)
    assert {record["custom_id"] for record in records} == set(custom_ids)
    assert all(record["status"] == "rejected" for record in records)
    terminal_by_id = {record["custom_id"]: record["reason"] for record in records}
    assert terminal_by_id == {
        custom_ids[0]: "duplicate_custom_id",
        custom_ids[1]: "duplicate_custom_id",
        custom_ids[2]: "missing_custom_id",
    }
    assert any(record["reason"] == "cross_shard_custom_id" and record.get("custom_id") == custom_ids[0] for record in diagnostics)
    assert any(record["reason"] == "unknown_custom_id" for record in diagnostics)
    assert collected["counts"] == {"prepared": 3, "accepted": 0, "rejected": 3, "diagnostics": len(diagnostics)}
    assert collected["shards"][0]["output_sha256"]
    assert collected["shards"][0]["error_sha256"]


def test_collection_persists_downloaded_artifacts_before_final_collection_write(tmp_path: Path, monkeypatch):
    row = target()
    state = prepare_generation_batches(
        [row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )
    client = FakeOpenAI()
    submitted = submit_pending_shards(state, client, tmp_path)
    custom_id = submitted["shards"][0]["custom_ids"][0]
    output = {
        "usable": False,
        "reason": "insufficient_context",
        "query": "",
        "answer": "",
        "evidence": "",
        "requested_intent": "how_to",
        "actual_intent": "how_to",
        "disambiguators": [],
    }
    client.batch_objects["batch-1"] = FakeObject(id="batch-1", output_file_id="output-1", error_file_id="error-1")
    client.file_contents["output-1"] = json.dumps(
        {"custom_id": custom_id, "response": {"status_code": 200, "body": {"output_text": json.dumps(output)}}}
    )
    client.file_contents["error-1"] = ""
    write = batch_module._write_bytes_atomic

    def fail_final_write(path, value):
        if path.name == "generation_collected.jsonl":
            raise OSError("simulated final collection write failure")
        return write(path, value)

    monkeypatch.setattr(batch_module, "_write_bytes_atomic", fail_final_write)
    with pytest.raises(OSError, match="simulated"):
        collect_completed_shards(submitted, client, tmp_path)
    persisted = json.loads((tmp_path / "generation_state.json").read_text(encoding="utf-8"))
    assert persisted["shards"][0]["output_sha256"]
    assert persisted["shards"][0]["error_sha256"]

    monkeypatch.setattr(batch_module, "_write_bytes_atomic", write)
    resumed = prepare_generation_batches(
        [row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )
    assert collect_completed_shards(resumed, client, tmp_path)["counts"] == {
        "prepared": 1,
        "accepted": 0,
        "rejected": 1,
        "diagnostics": 0,
    }
    assert client.downloads == ["output-1", "error-1"]


def test_submit_reconciliation_ignores_same_named_file_with_wrong_purpose(tmp_path: Path):
    row = target()
    state = prepare_generation_batches(
        [row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )
    shard = state["shards"][0]
    client = FakeOpenAI()
    client.file_objects.append(
        FakeObject(
            id="wrong-purpose-file",
            filename=Path(shard["request_path"]).name,
            bytes=shard["request_bytes"],
            purpose="assistants",
        )
    )

    submitted = submit_pending_shards(state, client, tmp_path)

    assert submitted["shards"][0]["input_file_id"] == "file-1"
    assert len(client.uploads) == 1


def test_submit_fails_clearly_when_the_local_process_lock_cannot_be_acquired(tmp_path: Path, monkeypatch):
    row = target()
    state = prepare_generation_batches(
        [row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )

    def fail_lock(*_args):
        raise OSError("lock unavailable")

    monkeypatch.setattr(batch_module.fcntl, "flock", fail_lock)
    with pytest.raises(RuntimeError, match="unable to acquire local generation submission lock"):
        submit_pending_shards(state, FakeOpenAI(), tmp_path)


def test_collection_recovers_orphan_downloaded_artifact_without_redownloading(tmp_path: Path):
    row = target()
    state = prepare_generation_batches(
        [row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )
    client = FakeOpenAI()
    submitted = submit_pending_shards(state, client, tmp_path)
    custom_id = submitted["shards"][0]["custom_ids"][0]
    client.batch_objects["batch-1"] = FakeObject(id="batch-1", output_file_id="output-1")
    orphan = tmp_path / "generation_outputs" / "output-1.jsonl"
    orphan.parent.mkdir()
    orphan.write_text(
        json.dumps(
            {
                "custom_id": custom_id,
                "response": {
                    "status_code": 200,
                    "body": {
                        "output_text": json.dumps(
                            {
                                "usable": False,
                                "reason": "insufficient_context",
                                "query": "",
                                "answer": "",
                                "evidence": "",
                                "requested_intent": "how_to",
                                "actual_intent": "how_to",
                                "disambiguators": [],
                            }
                        )
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    collected = collect_completed_shards(submitted, client, tmp_path)

    assert collected["shards"][0]["output_sha256"]
    assert client.downloads == []


def test_batch_state_binds_separate_target_and_context_artifact_checksums(tmp_path: Path):
    row = target()

    state = prepare_generation_batches(
        [row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )

    assert state["version"] == 3
    assert state["targets"]["path"] == "targets.parquet"
    assert state["targets"]["sha256"]
    assert state["generation_context"]["path"] == "generation_context.parquet"
    assert state["generation_context"]["sha256"]
    assert state["targets"]["sha256"] != state["generation_context"]["sha256"]


def test_submit_refuses_mutated_generation_inputs_without_any_remote_call(tmp_path: Path):
    row = target()
    state = prepare_generation_batches(
        [row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )
    (tmp_path / "targets.parquet").write_bytes(b"mutated")
    client = FakeOpenAI()

    with pytest.raises(ValueError, match="targets artifact checksum mismatch"):
        submit_pending_shards(state, client, tmp_path)

    assert client.uploads == []
    assert client.creates == []
    assert client.retrieves == []


def test_scale_gate_rejects_101_checksummed_rows_declared_as_100_without_remote_calls(tmp_path: Path):
    rows = [target(index) for index in range(101)]
    state = prepare_generation_batches(
        rows,
        [item for row in rows for item in context(row)],
        GeneratorConfig(),
        tmp_path / "full",
        source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT,
        pilot_generation_root=tmp_path / "pilot",
    )
    state["shards"][0]["request_count"] = 100
    client = FakeOpenAI()

    with pytest.raises(ValueError, match="request_count.*101"):
        submit_pending_shards(state, client, tmp_path / "full")

    assert (client.file_lists, client.uploads, client.batch_lists, client.creates, client.retrieves) == (0, [], 0, [], [])


def test_state_validation_reconciles_declared_custom_ids_with_checksummed_rows(tmp_path: Path):
    row = target()
    state = prepare_generation_batches(
        [row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )
    state["shards"][0]["custom_ids"] = ["forged-custom-id"]
    client = FakeOpenAI()

    with pytest.raises(ValueError, match="custom_ids.*request shard"):
        submit_pending_shards(state, client, tmp_path)

    assert (client.file_lists, client.uploads, client.batch_lists, client.creates, client.retrieves) == (0, [], 0, [], [])


def test_state_validation_rejects_duplicate_jsonl_custom_ids_before_remote_calls(tmp_path: Path):
    row = target()
    state = prepare_generation_batches(
        [row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )
    shard = state["shards"][0]
    request_path = tmp_path / shard["request_path"]
    original = request_path.read_bytes()
    request_path.write_bytes(original + original)
    shard["request_sha256"] = batch_module._sha256_file(request_path)
    shard["request_bytes"] = len(original) * 2
    shard["request_count"] = 2
    shard["custom_ids"] = [shard["custom_ids"][0], "forged-custom-id"]
    client = FakeOpenAI()

    with pytest.raises(ValueError, match="duplicate custom_id in request shard"):
        submit_pending_shards(state, client, tmp_path)

    assert (client.file_lists, client.uploads, client.batch_lists, client.creates, client.retrieves) == (0, [], 0, [], [])


def test_prepare_generation_batches_requires_all_v3_source_and_sidecar_bindings(tmp_path: Path):
    row = target()

    with pytest.raises(TypeError, match="source_passages_sha256"):
        prepare_generation_batches_core(
            [row],
            context(row),
            GeneratorConfig(),
            tmp_path,
            source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT,
        )


@pytest.mark.parametrize(
    "missing",
    ("source_passages_sha256", "targets", "generation_context", "targets_sidecar", "context_sidecar"),
)
def test_v3_state_validation_requires_complete_core_bindings_before_remote_calls(tmp_path: Path, missing: str):
    row = target()
    state = prepare_generation_batches(
        [row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )
    if missing in {"source_passages_sha256", "targets", "generation_context"}:
        state.pop(missing)
    elif missing == "targets_sidecar":
        state["targets"]["state_path"] = None
        state["targets"]["state_sha256"] = None
    else:
        state["generation_context"]["state_path"] = None
        state["generation_context"]["state_sha256"] = None
    client = FakeOpenAI()

    with pytest.raises(ValueError, match="v3.*binding"):
        submit_pending_shards(state, client, tmp_path)

    assert (client.file_lists, client.uploads, client.batch_lists, client.creates, client.retrieves) == (0, [], 0, [], [])


def test_submit_rejects_legacy_v2_state_before_remote_calls(tmp_path: Path):
    row = target()
    state = prepare_generation_batches(
        [row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )
    state["version"] = 2
    client = FakeOpenAI()

    with pytest.raises(ValueError, match="legacy v2.*submit"):
        submit_pending_shards(state, client, tmp_path)

    assert (client.file_lists, client.uploads, client.batch_lists, client.creates, client.retrieves) == (0, [], 0, [], [])


def test_retry_rejects_legacy_v2_state_before_remote_calls(tmp_path: Path):
    row = target()
    state = prepare_generation_batches(
        [row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )
    state["version"] = 2
    state["shards"][0].update({"input_file_id": "legacy-input", "batch_id": "legacy-batch", "status": "failed"})
    client = FakeOpenAI()
    client.batch_objects["legacy-batch"] = FakeObject(id="legacy-batch", status="failed")

    with pytest.raises(ValueError, match="legacy v2.*retry"):
        batch_module.retry_failed_shards(state, client, tmp_path)

    assert (client.file_lists, client.uploads, client.batch_lists, client.creates, client.retrieves) == (0, [], 0, [], [])


def test_scale_submit_requires_authorization_and_passing_checksummed_pilot_metrics(tmp_path: Path):
    rows = [target(index) for index in range(101)]
    contexts = [item for row in rows for item in context(row)]
    state = prepare_generation_batches(
        rows,
        contexts,
        GeneratorConfig(),
        tmp_path / "full",
        source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT,
        pilot_generation_root=tmp_path / "pilot",
    )
    client = FakeOpenAI()

    with pytest.raises(ValueError, match="explicit scale authorization"):
        submit_pending_shards(
            state,
            client,
            tmp_path / "full",
            scale_authorized=False,
            pilot_generation_root=tmp_path / "pilot",
        )

    assert client.uploads == []
    assert client.creates == []

    with pytest.raises(ValueError, match="checksummed pilot metrics"):
        submit_pending_shards(
            state,
            client,
            tmp_path / "full",
            scale_authorized=True,
            pilot_generation_root=tmp_path / "pilot",
        )

    submitted = submit_pending_shards(
        state,
        client,
        tmp_path / "full",
        scale_authorized=True,
        pilot_generation_root=passing_pilot_root(tmp_path),
    )
    assert submitted["counts"] == {"submitted": 1}
    assert len(client.uploads) == 1
    assert len(client.creates) == 1


def test_scale_authorization_requires_pilot_source_passage_sha_before_remote_calls(tmp_path: Path):
    rows = [target(index) for index in range(101)]
    state = prepare_generation_batches(
        rows,
        [item for row in rows for item in context(row)],
        GeneratorConfig(),
        tmp_path / "full",
        source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT,
        pilot_generation_root=tmp_path / "pilot",
    )
    pilot_root = passing_pilot_root(tmp_path)
    pilot_state_path = pilot_root / "generation_state.json"
    pilot_state = json.loads(pilot_state_path.read_text(encoding="utf-8"))
    pilot_state.pop("source_passages_sha256")
    batch_module._write_json_atomic(pilot_state_path, pilot_state)
    client = FakeOpenAI()

    with pytest.raises(ValueError, match="pilot source passage SHA-256"):
        submit_pending_shards(
            state,
            client,
            tmp_path / "full",
            scale_authorized=True,
            pilot_generation_root=pilot_root,
        )

    assert (client.file_lists, client.uploads, client.batch_lists, client.creates, client.retrieves) == (0, [], 0, [], [])


def test_scale_authorization_rejects_legacy_pilot_state_before_remote_calls(tmp_path: Path):
    rows = [target(index) for index in range(101)]
    state = prepare_generation_batches(
        rows,
        [item for row in rows for item in context(row)],
        GeneratorConfig(),
        tmp_path / "full",
        source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT,
        pilot_generation_root=tmp_path / "pilot",
    )
    pilot_root = passing_pilot_root(tmp_path)
    pilot_state_path = pilot_root / "generation_state.json"
    pilot_state = json.loads(pilot_state_path.read_text(encoding="utf-8"))
    pilot_state["version"] = 2
    batch_module._write_json_atomic(pilot_state_path, pilot_state)
    client = FakeOpenAI()

    with pytest.raises(ValueError, match="fully bound v3 pilot"):
        submit_pending_shards(
            state,
            client,
            tmp_path / "full",
            scale_authorized=True,
            pilot_generation_root=pilot_root,
        )

    assert (client.file_lists, client.uploads, client.batch_lists, client.creates, client.retrieves) == (0, [], 0, [], [])


def test_multi_shard_submit_requires_the_scale_gate_even_below_100_requests(tmp_path: Path):
    rows = [target(index) for index in range(2)]
    state = prepare_generation_batches(
        rows,
        [item for row in rows for item in context(row)],
        GeneratorConfig(max_requests_per_shard=1),
        tmp_path / "full",
        source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT,
        pilot_generation_root=tmp_path / "pilot",
    )
    client = FakeOpenAI()

    with pytest.raises(ValueError, match="explicit scale authorization"):
        submit_pending_shards(
            state,
            client,
            tmp_path / "full",
            scale_authorized=False,
            pilot_generation_root=tmp_path / "pilot",
        )

    assert client.uploads == []
    assert client.creates == []


def test_collection_applies_global_dedup_and_persists_checksummed_pilot_metrics(tmp_path: Path):
    rows = [target(index) for index in range(2)]
    state = prepare_generation_batches(
        rows,
        [item for row in rows for item in context(row)],
        GeneratorConfig(),
        tmp_path,
        source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT,
    )
    client = FakeOpenAI()
    submitted = submit_pending_shards(state, client, tmp_path)
    custom_ids = submitted["shards"][0]["custom_ids"]
    output = accepted_output()
    client.batch_objects["batch-1"] = FakeObject(id="batch-1", output_file_id="output-1")
    client.file_contents["output-1"] = "\n".join(
        json.dumps(response_row(custom_id, output, usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}))
        for custom_id in custom_ids
    )

    collected = collect_completed_shards(submitted, client, tmp_path)

    records = [json.loads(line) for line in (tmp_path / "generation_collected.jsonl").read_text().splitlines()]
    metrics = json.loads((tmp_path / "pilot_metrics.json").read_text(encoding="utf-8"))
    assert [record["status"] for record in records] == ["accepted", "rejected"]
    assert records[0]["reason_codes"] == []
    assert "duplicate_normalized_query" in records[1]["reason_codes"]
    assert metrics["generation_fingerprint"] == state["generation_fingerprint"]
    assert metrics["schema_success_rate"] == 1.0
    assert metrics["usable_rate"] == 1.0
    assert metrics["deterministic_gate_pass_rate"] == 0.5
    assert metrics["intent_agreement_rate"] == 1.0
    assert metrics["evidence_validity_rate"] == 1.0
    assert metrics["duplicate_count"] == 1
    assert metrics["token_usage"] == {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
    assert collected["pilot_metrics_sha256"] == batch_module._sha256_file(tmp_path / "pilot_metrics.json")


def test_pilot_metrics_count_unusable_empty_evidence_as_invalid(tmp_path: Path):
    row = target()
    state = prepare_generation_batches(
        [row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )
    client = FakeOpenAI()
    submitted = submit_pending_shards(state, client, tmp_path)
    custom_id = submitted["shards"][0]["custom_ids"][0]
    unusable = accepted_output()
    unusable.update(usable=False, reason="insufficient_context", query="", answer="", evidence="")
    client.batch_objects["batch-1"] = FakeObject(id="batch-1", output_file_id="output-1")
    client.file_contents["output-1"] = json.dumps(response_row(custom_id, unusable))

    collect_completed_shards(submitted, client, tmp_path)

    metrics = json.loads((tmp_path / "pilot_metrics.json").read_text(encoding="utf-8"))
    assert metrics["usable_rate"] == 0.0
    assert metrics["evidence_valid_count"] == 0
    assert metrics["evidence_validity_rate"] == 0.0


def test_retry_failed_shard_preserves_history_and_starts_only_one_new_attempt(tmp_path: Path):
    row = target()
    state = prepare_generation_batches(
        [row],
        context(row),
        GeneratorConfig(max_batch_attempts=2),
        tmp_path,
        source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT,
    )
    client = FakeOpenAI()
    submitted = submit_pending_shards(state, client, tmp_path)
    client.batch_objects["batch-1"] = FakeObject(
        id="batch-1", status="failed", output_file_id="partial-output", error_file_id="partial-error"
    )

    retried = batch_module.retry_failed_shards(submitted, client, tmp_path)

    shard = retried["shards"][0]
    assert shard["attempt"] == 2
    assert shard["batch_id"] == "batch-2"
    assert len(shard["history"]) == 1
    assert shard["history"][0]["batch_id"] == "batch-1"
    assert shard["history"][0]["status"] == "failed"
    assert shard["history"][0]["output_file_id"] == "partial-output"
    assert shard["history"][0]["error_file_id"] == "partial-error"
    assert client.creates == ["file-1", "file-1"]


@pytest.mark.parametrize("status", ("completed", "in_progress", "validating"))
def test_retry_rejects_non_retryable_statuses_without_creating_a_batch(tmp_path: Path, status: str):
    row = target()
    state = prepare_generation_batches(
        [row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )
    client = FakeOpenAI()
    submitted = submit_pending_shards(state, client, tmp_path)
    client.batch_objects["batch-1"] = FakeObject(id="batch-1", status=status)

    with pytest.raises(ValueError, match="no failed, expired, or cancelled shards"):
        batch_module.retry_failed_shards(submitted, client, tmp_path)

    assert client.creates == ["file-1"]


def test_retry_enforces_total_batch_attempt_cap_without_creating_a_batch(tmp_path: Path):
    row = target()
    state = prepare_generation_batches(
        [row],
        context(row),
        GeneratorConfig(max_batch_attempts=2),
        tmp_path,
        source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT,
    )
    client = FakeOpenAI()
    submitted = submit_pending_shards(state, client, tmp_path)
    submitted["shards"][0]["attempt"] = 2
    client.batch_objects["batch-1"] = FakeObject(id="batch-1", status="expired")

    with pytest.raises(ValueError, match="attempt cap of 2"):
        batch_module.retry_failed_shards(submitted, client, tmp_path)

    assert client.creates == ["file-1"]


def test_version_2_collection_without_finalizer_version_is_refinalized_from_downloaded_raw_output(tmp_path: Path):
    row = target()
    row["overlap_prefix_chars"] = len("Redis повторяет запросы после сетевой ошибки")
    state = prepare_generation_batches(
        [row], context(row), GeneratorConfig(), tmp_path, source_corpus_fingerprint=SOURCE_CORPUS_FINGERPRINT
    )
    client = FakeOpenAI()
    submitted = submit_pending_shards(state, client, tmp_path)
    custom_id = submitted["shards"][0]["custom_ids"][0]
    client.batch_objects["batch-1"] = FakeObject(id="batch-1", output_file_id="output-1")
    client.file_contents["output-1"] = json.dumps(response_row(custom_id, accepted_output()))
    collected = collect_completed_shards(submitted, client, tmp_path)

    legacy_record = {"custom_id": custom_id, "output": accepted_output(), "status": "accepted"}
    legacy_bytes = (json.dumps(legacy_record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (tmp_path / "generation_collected.jsonl").write_bytes(legacy_bytes)
    legacy_state = dict(collected)
    legacy_state["version"] = 2
    legacy_state.pop("source_passages_sha256", None)
    legacy_state.pop("targets", None)
    legacy_state.pop("generation_context", None)
    legacy_state.pop("generation_config", None)
    legacy_state.pop("validator_version", None)
    legacy_state.pop("finalizer_version", None)
    legacy_state["collected_sha256"] = batch_module._sha256_bytes(legacy_bytes)
    batch_module._write_json_atomic(tmp_path / "generation_state.json", legacy_state)

    refinalized = collect_completed_shards(legacy_state, client, tmp_path)

    record = json.loads((tmp_path / "generation_collected.jsonl").read_text(encoding="utf-8"))
    assert record["status"] == "rejected"
    assert "evidence_overlap_only" in record["reason_codes"]
    assert refinalized["version"] == 2
    assert refinalized["finalizer_version"]
    assert collect_completed_shards(refinalized, client, tmp_path) == refinalized
    assert client.downloads == ["output-1"]
