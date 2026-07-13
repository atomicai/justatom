from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from justatom.tooling.ir_dataset.batch import (
    collect_completed_shards,
    prepare_generation_batches,
    submit_pending_shards,
    write_batch_shards,
)
from justatom.tooling.ir_dataset.generation import GeneratorConfig


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


class FakeFiles:
    def __init__(self, owner: "FakeOpenAI") -> None:
        self.owner = owner

    def create(self, *, file, purpose: str):
        assert purpose == "batch"
        self.owner.uploads.append((Path(file.name).name, file.read()))
        return FakeObject(id=f"file-{len(self.owner.uploads)}")

    def content(self, file_id: str):
        self.owner.downloads.append(file_id)
        return self.owner.file_contents[file_id]


class FakeBatches:
    def __init__(self, owner: "FakeOpenAI") -> None:
        self.owner = owner

    def create(self, *, input_file_id: str, endpoint: str, completion_window: str):
        assert endpoint == "/v1/responses"
        assert completion_window == "24h"
        self.owner.creates.append(input_file_id)
        return FakeObject(id=f"batch-{len(self.owner.creates)}")

    def retrieve(self, batch_id: str):
        self.owner.retrieves.append(batch_id)
        return self.owner.batch_objects[batch_id]


class FakeOpenAI:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, bytes]] = []
        self.creates: list[str] = []
        self.retrieves: list[str] = []
        self.downloads: list[str] = []
        self.file_contents: dict[str, object] = {}
        self.batch_objects: dict[str, FakeObject] = {}
        self.files = FakeFiles(self)
        self.batches = FakeBatches(self)


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


def test_submit_is_idempotent(tmp_path: Path):
    state = prepare_generation_batches([target()], context(target()), GeneratorConfig(model="test-model"), tmp_path)
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
    state = prepare_generation_batches([target()], context(target()), GeneratorConfig(model="test-model"), tmp_path)
    request_path = tmp_path / state["shards"][0]["request_path"]
    request_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        submit_pending_shards(state, FakeOpenAI(), tmp_path)


def test_collection_accepts_only_http_200_strict_output_text_and_keeps_rejections(tmp_path: Path):
    row = target()
    state = prepare_generation_batches([row], context(row), GeneratorConfig(model="test-model"), tmp_path)
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
    client.batch_objects["batch-1"] = FakeObject(id="batch-1", output_file_id="output-1", error_file_id="error-1")
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
    assert collected["counts"] == {"accepted": 1, "rejected": 2}
    assert records[0]["status"] == "accepted"
    assert records[0]["output"] == output
    assert {record["reason"] for record in records[1:]} == {"unknown_custom_id", "malformed_batch_row"}
    assert client.downloads == ["output-1", "error-1"]
    assert collect_completed_shards(collected, FakeOpenAI(), tmp_path) == collected
