from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from justatom.tooling.ir_dataset.generation import (
    GENERATOR_SCHEMA,
    GeneratorConfig,
    build_generator_request,
    normalize_query,
    validate_generator_result,
)


def slot(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "article_id": "article-1",
        "passage_id": "passage-1",
        "source_hash": "source-sha256",
        "content": (
            "В режиме production сервис Redis повторяет запросы после сетевой ошибки. "
            "Очередь задач сохраняет состояние перед повторной попыткой."
        ),
        "serialized_passage": (
            "passage: Redis\nРежим production\n\nВ режиме production сервис Redis повторяет запросы после сетевой ошибки. "
            "Очередь задач сохраняет состояние перед повторной попыткой."
        ),
        "token_count": 120,
        "requested_intent": "how_to",
        "split": "train",
        "slot_index": 0,
    }
    row.update(overrides)
    return row


def context(target: dict[str, object] | None = None) -> list[dict[str, object]]:
    target = target or slot()
    return [
        {
            "target_article_id": target["article_id"],
            "target_passage_id": target["passage_id"],
            "target_source_hash": target["source_hash"],
            "target_content": target["content"],
            "candidate_passage_id": f"neighbor-{index}",
            "candidate_serialized_passage": f"passage: Neighbor {index}\n\nНеобходимый контекст {index}.",
            "context_index": index,
        }
        for index in range(3)
    ]


def config(**overrides: object) -> GeneratorConfig:
    values: dict[str, object] = {"attempt": 2}
    values.update(overrides)
    return GeneratorConfig(**values)


def good_output(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "usable": True,
        "reason": "ok",
        "query": "Как Redis повторяет запросы после сетевой ошибки в production?",
        "answer": "Сервис повторяет запросы после сетевой ошибки.",
        "evidence": "Redis повторяет запросы после сетевой ошибки.",
        "requested_intent": "how_to",
        "actual_intent": "how_to",
        "disambiguators": ["Redis", "production", "сетевая ошибка"],
    }
    result.update(overrides)
    return result


def test_request_uses_responses_input_text_low_reasoning_and_strict_schema():
    request = build_generator_request(slot(), context(), config())

    assert request["method"] == "POST"
    assert request["url"] == "/v1/responses"
    assert request["body"]["model"] == "gpt-5.6-terra"
    assert request["body"]["reasoning"] == {"effort": "low"}
    assert request["body"]["store"] is False
    assert request["body"]["input"][0]["content"][0]["type"] == "input_text"
    assert request["body"]["input"][1]["content"][0]["type"] == "input_text"
    assert request["body"]["text"]["format"]["strict"] is True
    assert request["body"]["text"]["format"]["schema"] == GENERATOR_SCHEMA


def test_full_checked_in_generation_mapping_is_accepted_without_relaxing_shard_limits():
    generation_mapping = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "datasets" / "habr-ir.yaml").read_text(encoding="utf-8")
    )["generation"]

    request = build_generator_request(slot(), context(), generation_mapping)
    result = validate_generator_result(good_output(), slot(), context(), config=generation_mapping)

    assert request["body"]["model"] == generation_mapping["model"]
    assert result.accepted
    with pytest.raises(ValueError, match="max_requests_per_shard"):
        GeneratorConfig(max_requests_per_shard=1_001)


def test_schema_requires_every_output_field_and_disallows_unknown_properties():
    assert GENERATOR_SCHEMA["additionalProperties"] is False
    assert set(GENERATOR_SCHEMA["required"]) == set(GENERATOR_SCHEMA["properties"])
    assert set(GENERATOR_SCHEMA["required"]) == {
        "usable",
        "reason",
        "query",
        "answer",
        "evidence",
        "requested_intent",
        "actual_intent",
        "disambiguators",
    }


def test_custom_id_is_stable_and_contains_prompt_hash_metadata():
    first = build_generator_request(slot(), context(), config())
    second = build_generator_request(slot(), context(), config())
    changed = build_generator_request(slot(), context(), config(attempt=3))

    expected_digest = hashlib.sha256(
        ":".join(("article-1", "passage-1", "how_to", first["body"]["metadata"]["prompt_hash"], "2")).encode("utf-8")
    ).hexdigest()
    assert first["custom_id"] == f"gen-{expected_digest}"
    assert first["custom_id"] == second["custom_id"]
    assert first["custom_id"] != changed["custom_id"]
    assert first["body"]["metadata"]["source_hash"] == "source-sha256"


def test_request_requires_exactly_three_matching_context_rows():
    invalid = context()[:2]

    try:
        build_generator_request(slot(), invalid, config())
    except ValueError as exc:
        assert "exactly three" in str(exc)
    else:
        raise AssertionError("request accepted incomplete collision context")

    mismatched = context()
    mismatched[1]["target_source_hash"] = "other-source"
    try:
        build_generator_request(slot(), mismatched, config())
    except ValueError as exc:
        assert "source_hash" in str(exc)
    else:
        raise AssertionError("request accepted mismatched target identity")


def test_gates_accept_grounded_output_without_mutating_model_payload():
    output = good_output()
    original = copy.deepcopy(output)

    result = validate_generator_result(output, slot(), context(), config=config())

    assert result.accepted
    assert result.reason_codes == ()
    assert result.output is output
    assert output == original


def test_gates_reject_missing_generation_context():
    result = validate_generator_result(good_output(), slot(), config=config())

    assert not result.accepted
    assert "generation_context_missing" in result.reason_codes


def test_gates_require_exact_evidence():
    result = validate_generator_result(good_output(evidence="несуществующая цитата"), slot(), context(), config=config())

    assert "evidence_not_substring" in result.reason_codes


def test_gates_reject_schema_and_usable_contract_violations():
    malformed = good_output()
    malformed["extra"] = "not allowed"

    assert "schema_invalid" in validate_generator_result(malformed, slot(), context(), config=config()).reason_codes
    assert (
        "usable_false"
        in validate_generator_result(
            good_output(usable=False, reason="unsupported_intent", query="", answer="", evidence=""),
            slot(),
            context(),
            config=config(),
        ).reason_codes
    )


def test_gates_reject_empty_fields_word_limits_banned_context_and_intent_mismatch():
    short = good_output(query="Как настроить Redis?")
    long = good_output(query=" ".join(["слово"] * 31))
    contextual = good_output(query="Как в статье Redis повторяет запросы после сетевой ошибки в production?")
    mismatch = good_output(actual_intent="why")

    assert "query_word_count" in validate_generator_result(short, slot(), context(), config=config()).reason_codes
    assert "query_word_count" in validate_generator_result(long, slot(), context(), config=config()).reason_codes
    assert "banned_context_phrase" in validate_generator_result(contextual, slot(), context(), config=config()).reason_codes
    assert "intent_mismatch" in validate_generator_result(mismatch, slot(), context(), config=config()).reason_codes
    assert "empty_answer" in validate_generator_result(good_output(answer=""), slot(), context(), config=config()).reason_codes


def test_gates_reject_context_reference_inflections():
    for reference in ("статье", "автора", "автором"):
        result = validate_generator_result(
            good_output(query=f"Как {reference} Redis повторяет запросы после сетевой ошибки в production?"),
            slot(),
            context(),
            config=config(),
        )

        assert "banned_context_phrase" in result.reason_codes


def test_gates_reject_normalized_duplicates_long_copied_spans_and_target_contract_mismatches():
    duplicate = validate_generator_result(
        good_output(),
        slot(),
        context(),
        accepted_normalized_queries={normalize_query("Как Redis повторяет запросы после сетевой ошибки в production?")},
        config=config(),
    )
    copied = validate_generator_result(
        good_output(
            query=(
                "В режиме production сервис Redis повторяет запросы после сетевой ошибки "
                "очередь задач сохраняет состояние перед повторной попыткой"
            )
        ),
        slot(),
        context(),
        config=config(),
    )
    oversized = validate_generator_result(good_output(), slot(token_count=505), context(), config=config())
    mismatched_context = context()
    mismatched_context[0]["target_passage_id"] = "wrong-passage"
    identity = validate_generator_result(good_output(), slot(), mismatched_context, config=config())

    assert "duplicate_normalized_query" in duplicate.reason_codes
    assert "copied_content_span_gt_8" in copied.reason_codes
    assert "target_length_exceeded" in oversized.reason_codes
    assert "target_identity_mismatch" in identity.reason_codes


def test_gates_reject_long_copied_english_prose_but_allow_identifier_only_overlap():
    english_target = slot(
        content=("The worker retries failed network requests after timeout and stores job state before each attempt."),
        serialized_passage=(
            "passage: Worker\n\nThe worker retries failed network requests after timeout and stores job state before each attempt."
        ),
    )
    copied_english = validate_generator_result(
        good_output(
            query="The worker retries failed network requests after timeout and stores job state before each attempt.",
            answer="The worker retries requests.",
            evidence="The worker retries failed network requests after timeout and stores job state before each attempt.",
        ),
        english_target,
        context(english_target),
        config=config(),
    )
    identifier_target = slot(
        content="HTTP_2 v2 API JSON TLS SSH TCP UDP IPv6",
        serialized_passage="passage: Protocols\n\nHTTP_2 v2 API JSON TLS SSH TCP UDP IPv6",
    )
    identifier_overlap = validate_generator_result(
        good_output(
            query="How HTTP_2 v2 API JSON TLS SSH TCP UDP IPv6 works?",
            answer="It works with the listed identifiers.",
            evidence="HTTP_2 v2 API JSON TLS SSH TCP UDP IPv6",
        ),
        identifier_target,
        context(identifier_target),
        config=config(),
    )

    assert "copied_content_span_gt_8" in copied_english.reason_codes
    assert identifier_overlap.accepted
