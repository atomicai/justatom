from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import polars as pl
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from justatom.tooling.ir_dataset.artifacts import sha256_file, write_bound_parquet_artifact
from justatom.tooling.ir_dataset.batch import _generation_fingerprint, _target_context_fingerprint
from justatom.tooling.ir_dataset.chunking import CHUNKER_VERSION
from justatom.tooling.ir_dataset.generation import GeneratorConfig
from justatom.tooling.ir_dataset.release import (
    GenerationBinding,
    finalize_release,
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


def test_materialization_requires_one_request_binding_per_target():
    duplicate_bindings = bindings()
    duplicate_bindings["gen-rejected"] = replace(
        duplicate_bindings["gen-rejected"],
        passage_id="p1",
        article_id="a1",
        source_hash="source-1",
    )

    with pytest.raises(ValueError, match="one-to-one.*targets"):
        materialize_release_frames(
            records=records(),
            targets=targets(),
            corpus=corpus(),
            bindings=duplicate_bindings,
            generator_model="test-model",
        )


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows)


def write_release_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "source"
    generation_root = tmp_path / "generation"
    release_root = tmp_path / "release"
    source_root.mkdir()
    generation_root.mkdir()
    (generation_root / "generation_requests").mkdir()

    source_frame = corpus()
    passages_path = source_root / "passages.parquet"
    source_frame.write_parquet(passages_path)
    passages_sha256 = sha256_file(passages_path)
    source_fingerprint = "source-fingerprint-v4"
    manifest_path = source_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "fingerprint": source_fingerprint,
                "chunker_version": CHUNKER_VERSION,
                "passages_sha256": passages_sha256,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    target_path = generation_root / "targets.parquet"
    target_state_path = generation_root / "targets_state.json"
    write_bound_parquet_artifact(
        targets(),
        target_path,
        target_state_path,
        artifact_kind="targets",
        source_corpus_fingerprint=source_fingerprint,
        passages_sha256=passages_sha256,
        config={"fixture": 1},
        upstream_sha256=sha256_file(manifest_path),
    )
    context_frame = pl.DataFrame(
        [
            {
                "target_passage_id": target_id,
                "candidate_passage_id": candidate_id,
                "candidate_serialized_passage": candidate_text,
                "context_index": context_index,
            }
            for target_id, candidate_id, candidate_text in (
                ("p1", "p2", source_frame.row(1, named=True)["serialized_passage"]),
                ("p2", "p1", source_frame.row(0, named=True)["serialized_passage"]),
            )
            for context_index in range(3)
        ]
    )
    context_path = generation_root / "generation_context.parquet"
    context_state_path = generation_root / "generation_context_state.json"
    write_bound_parquet_artifact(
        context_frame,
        context_path,
        context_state_path,
        artifact_kind="generation_context",
        source_corpus_fingerprint=source_fingerprint,
        passages_sha256=passages_sha256,
        config={"fixture": 1},
        upstream_sha256=sha256_file(target_path),
    )

    generator_config = GeneratorConfig()
    request_rows = []
    for custom_id, binding in bindings().items():
        request_rows.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": generator_config.model,
                    "metadata": {
                        "article_id": binding.article_id,
                        "generation_attempt": str(binding.generation_attempt),
                        "passage_id": binding.passage_id,
                        "prompt_hash": binding.prompt_hash,
                        "source_hash": binding.source_hash,
                    },
                },
            }
        )
    request_bytes = _jsonl(request_rows)
    request_path = generation_root / "generation_requests" / "generation-00000.jsonl"
    request_path.write_bytes(request_bytes)

    collected_bytes = _jsonl(records())
    collected_path = generation_root / "generation_collected.jsonl"
    collected_path.write_bytes(collected_bytes)
    diagnostics_path = generation_root / "generation_diagnostics.jsonl"
    diagnostics_path.write_bytes(b"")
    metrics_path = generation_root / "pilot_metrics.json"
    metrics_path.write_text(json.dumps({"request_count": 2}, sort_keys=True), encoding="utf-8")

    target_context_fingerprint = _target_context_fingerprint(targets().to_dicts(), context_frame.to_dicts())
    generation_fingerprint = _generation_fingerprint(
        source_fingerprint,
        target_context_fingerprint,
        generator_config,
        request_rows,
    )
    state = {
        "version": 3,
        "source_corpus_fingerprint": source_fingerprint,
        "source_passages_sha256": passages_sha256,
        "target_context_fingerprint": target_context_fingerprint,
        "generation_fingerprint": generation_fingerprint,
        "generation_config": asdict(generator_config),
        "targets": {
            "path": target_path.name,
            "sha256": sha256_file(target_path),
            "state_path": target_state_path.name,
            "state_sha256": sha256_file(target_state_path),
        },
        "generation_context": {
            "path": context_path.name,
            "sha256": sha256_file(context_path),
            "state_path": context_state_path.name,
            "state_sha256": sha256_file(context_state_path),
        },
        "shards": [
            {
                "attempt": 1,
                "batch_id": "batch-1",
                "custom_ids": list(bindings()),
                "request_path": request_path.relative_to(generation_root).as_posix(),
                "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
                "request_count": 2,
                "status": "completed",
            }
        ],
        "counts": {"prepared": 2, "accepted": 1, "rejected": 1, "diagnostics": 0},
        "collected_path": collected_path.name,
        "collected_sha256": hashlib.sha256(collected_bytes).hexdigest(),
        "diagnostics_path": diagnostics_path.name,
        "diagnostics_sha256": hashlib.sha256(b"").hexdigest(),
        "pilot_metrics_path": metrics_path.name,
        "pilot_metrics_sha256": sha256_file(metrics_path),
        "validator_version": 1,
        "finalizer_version": 1,
    }
    (generation_root / "generation_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return source_root, generation_root, release_root


def test_finalize_rejects_tampered_collected_output(tmp_path):
    source_root, generation_root, release_root = write_release_workspace(tmp_path)
    with (generation_root / "generation_collected.jsonl").open("ab") as stream:
        stream.write(b"{}\n")

    with pytest.raises(ValueError, match="collected artifact checksum mismatch"):
        finalize_release(source_root, generation_root, release_root, git_sha="test-sha", git_dirty=False)


def test_finalize_rejects_missing_or_ambiguous_request_binding(tmp_path):
    source_root, generation_root, release_root = write_release_workspace(tmp_path)
    request_path = next((generation_root / "generation_requests").glob("*.jsonl"))
    request_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="request shard checksum mismatch"):
        finalize_release(source_root, generation_root, release_root, git_sha="test-sha", git_dirty=False)


def _rewrite_bound_state(generation_root: Path, artifact: str, **changes: object) -> None:
    state_path = generation_root / f"{artifact}_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(changes)
    unsigned = {key: value for key, value in state.items() if key != "contract_sha256"}
    state["contract_sha256"] = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    generation_state_path = generation_root / "generation_state.json"
    generation_state = json.loads(generation_state_path.read_text(encoding="utf-8"))
    generation_state[artifact]["state_sha256"] = sha256_file(state_path)
    generation_state_path.write_text(json.dumps(generation_state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.parametrize("artifact", ["targets", "generation_context"])
def test_finalize_rejects_broken_artifact_upstream_chain(tmp_path, artifact):
    source_root, generation_root, release_root = write_release_workspace(tmp_path)
    _rewrite_bound_state(generation_root, artifact, upstream_sha256="0" * 64)

    with pytest.raises(ValueError, match=rf"{artifact} upstream checksum mismatch"):
        finalize_release(source_root, generation_root, release_root, git_sha="test-sha", git_dirty=False)


@pytest.mark.parametrize("field", ["target_context_fingerprint", "generation_fingerprint"])
def test_finalize_recomputes_generation_fingerprints(tmp_path, field):
    source_root, generation_root, release_root = write_release_workspace(tmp_path)
    state_path = generation_root / "generation_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state[field] = "0" * 64
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=field.replace("_", " ")):
        finalize_release(source_root, generation_root, release_root, git_sha="test-sha", git_dirty=False)


def test_finalize_writes_hf_layout_manifest_and_review_sheet(tmp_path):
    source_root, generation_root, release_root = write_release_workspace(tmp_path)

    result = finalize_release(source_root, generation_root, release_root, git_sha="test-sha", git_dirty=False)

    assert result.pair_count == 1
    assert result.review_count == 2
    assert (release_root / "data/pairs/train.parquet").exists()
    assert (release_root / "data/pairs/validation.parquet").exists()
    assert (release_root / "data/pairs/test.parquet").exists()
    assert (release_root / "data/corpus-100k/corpus.parquet").exists()
    assert (release_root / "data/qrels/train.parquet").exists()
    assert (release_root / "audit/pilot-review.csv").exists()
    assert (release_root / "README.md").exists()
    manifest = json.loads((release_root / "data/manifests/release-manifest.json").read_text())
    assert manifest["git"] == {"sha": "test-sha", "dirty": False}
    assert manifest["counts"]["pairs"] == 1
    assert manifest["counts"]["audit_rows"] == 2
    assert all(item["sha256"] for item in manifest["artifacts"])
    review = pl.read_csv(release_root / "audit/pilot-review.csv")
    assert review.columns[-10:] == [
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
    ]


def test_finalize_escapes_spreadsheet_formulas_in_audit_csv(tmp_path):
    source_root, generation_root, release_root = write_release_workspace(tmp_path)
    collected_path = generation_root / "generation_collected.jsonl"
    unsafe_records = records()
    unsafe_records[0]["output"]["query"] = '=HYPERLINK("https://example.test")'
    collected_bytes = _jsonl(unsafe_records)
    collected_path.write_bytes(collected_bytes)
    state_path = generation_root / "generation_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["collected_sha256"] = hashlib.sha256(collected_bytes).hexdigest()
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    finalize_release(source_root, generation_root, release_root, git_sha="test-sha", git_dirty=False)

    with (release_root / "audit/pilot-review.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["query"] == '\'=HYPERLINK("https://example.test")'


def test_finalize_writes_hugging_face_compatible_arrow_types(tmp_path):
    source_root, generation_root, release_root = write_release_workspace(tmp_path)

    finalize_release(source_root, generation_root, release_root, git_sha="test-sha", git_dirty=False)

    pair_schema = pq.read_schema(release_root / "data/pairs/train.parquet")
    corpus_schema = pq.read_schema(release_root / "data/corpus-100k/corpus.parquet")
    assert pair_schema.field("query").type == pa.string()
    assert pair_schema.field("topic_flows").type == pa.list_(pa.string())
    assert pair_schema.field("topic_hubs").type == pa.list_(pa.string())
    assert pair_schema.field("tags").type == pa.list_(pa.string())
    assert corpus_schema.field("flows").type == pa.list_(pa.string())
    assert corpus_schema.field("hubs").type == pa.list_(pa.string())
    assert corpus_schema.field("tags").type == pa.list_(pa.string())


def test_finalize_exactly_reuses_matching_release(tmp_path):
    source_root, generation_root, release_root = write_release_workspace(tmp_path)
    first = finalize_release(source_root, generation_root, release_root, git_sha="test-sha", git_dirty=False)
    second = finalize_release(source_root, generation_root, release_root, git_sha="test-sha", git_dirty=False)

    assert first.reused is False
    assert second.reused is True
    assert first.root == second.root
    assert first.manifest_path == second.manifest_path
    assert first.pair_count == second.pair_count
    assert first.corpus_count == second.corpus_count
    assert first.qrel_count == second.qrel_count
    assert first.review_count == second.review_count
    assert first.fingerprint == second.fingerprint


def test_finalize_rejects_existing_release_missing_required_artifact(tmp_path):
    source_root, generation_root, release_root = write_release_workspace(tmp_path)
    finalize_release(source_root, generation_root, release_root, git_sha="test-sha", git_dirty=False)
    manifest_path = release_root / "data/manifests/release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = "audit/pilot-review.csv"
    manifest["artifacts"] = [item for item in manifest["artifacts"] if item["path"] != missing]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (release_root / missing).unlink()

    with pytest.raises(ValueError, match="required artifact set mismatch"):
        finalize_release(source_root, generation_root, release_root, git_sha="test-sha", git_dirty=False)


def test_finalize_rejects_existing_release_with_false_counts(tmp_path):
    source_root, generation_root, release_root = write_release_workspace(tmp_path)
    finalize_release(source_root, generation_root, release_root, git_sha="test-sha", git_dirty=False)
    manifest_path = release_root / "data/manifests/release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["pairs"] = 999
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="release counts mismatch"):
        finalize_release(source_root, generation_root, release_root, git_sha="test-sha", git_dirty=False)
