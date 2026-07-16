from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


INTENTS = (
    "how_to",
    "why",
    "troubleshooting",
    "concept",
    "comparison",
    "requirements",
    "limitations",
    "factual",
)
UNUSABLE_REASONS = (
    "unsupported_intent",
    "insufficient_context",
    "ambiguous_target",
    "duplicate_with_neighbor",
    "malformed_source",
)
GENERATOR_SYSTEM_PROMPT = """Ты создаёшь один пример для русскоязычного benchmark по information retrieval.

Дан TARGET PASSAGE и несколько похожих, но нецелевых passages. Сформулируй
естественный вопрос пользователя, для которого TARGET является единственным
самодостаточным источником ответа среди показанных passages.

Требования:
1. Используй только информацию из TARGET. Не добавляй внешние знания.
2. Вопрос должен соответствовать REQUESTED INTENT.
3. Включи в вопрос технологию, компонент, версию, режим, условие или момент,
   если без них возможны разные ответы.
4. Не используй выражения "в статье", "автор", "этот подход", "выше" или
   другие ссылки на невидимый контекст.
5. Вопрос должен иметь одну устойчивую интерпретацию и краткий ответ.
6. Сохрани необходимые технические названия, но не копируй предложение,
   заголовок или длинную фразу из TARGET.
7. EVIDENCE должен быть точной непрерывной цитатой из TARGET.
8. Если TARGET не поддерживает REQUESTED INTENT или вопрос нельзя сделать
   однозначным, верни usable=false. Ничего не выдумывай.
9. Верни только объект, соответствующий JSON schema."""

GENERATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "usable",
        "reason",
        "query",
        "answer",
        "evidence",
        "requested_intent",
        "actual_intent",
        "disambiguators",
    ],
    "properties": {
        "usable": {"type": "boolean"},
        "reason": {"type": "string", "enum": ["ok", *UNUSABLE_REASONS]},
        "query": {"type": "string"},
        "answer": {"type": "string"},
        "evidence": {"type": "string"},
        "requested_intent": {"type": "string", "enum": list(INTENTS)},
        "actual_intent": {"type": "string", "enum": list(INTENTS)},
        "disambiguators": {"type": "array", "items": {"type": "string"}},
    },
}

_BANNED_CONTEXT_PHRASES = (
    "в статье",
    "этот подход",
    "эта статья",
    "данная статья",
    "в этом тексте",
    "этот пример",
    "предыдущий раздел",
    "следующий раздел",
    "выше",
    "ниже",
)
_BANNED_CONTEXT_PATTERNS = (
    re.compile(r"(?<!\w)стат(?:ья|ьи|ье|ью|ьей|ьёй|ей)(?!\w)"),
    re.compile(r"(?<!\w)автор(?:а|у|ом|е|ы|ов|ам|ами|ах)?(?!\w)"),
    *(re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)") for phrase in _BANNED_CONTEXT_PHRASES),
)
_STOP_WORDS = frozenset(
    {
        "а",
        "без",
        "в",
        "во",
        "для",
        "до",
        "и",
        "из",
        "как",
        "к",
        "на",
        "не",
        "о",
        "об",
        "от",
        "по",
        "после",
        "при",
        "с",
        "со",
        "что",
        "это",
    }
)
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_CONTENT_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    model: str = "gpt-5.6-terra"
    reasoning_effort: str = "low"
    prompt_cache_mode: str = "auto"
    attempt: int = 1
    accepted_max_tokens: int = 504
    max_requests_per_shard: int = 1_000
    max_shard_bytes: int = 100_000_000
    max_batch_attempts: int = 2
    scale_authorized: bool = False

    def __post_init__(self) -> None:
        if self.model != "gpt-5.6-terra":
            raise ValueError("generation.model must be gpt-5.6-terra")
        if self.reasoning_effort != "low":
            raise ValueError("generation.reasoning_effort must be low")
        if self.prompt_cache_mode not in {"auto", "explicit"}:
            raise ValueError("generation.prompt_cache_mode must be one of: auto, explicit")
        if self.attempt < 1:
            raise ValueError("generation.attempt must be >= 1")
        if self.accepted_max_tokens < 1:
            raise ValueError("generation.accepted_max_tokens must be > 0")
        if not isinstance(self.max_requests_per_shard, int) or isinstance(self.max_requests_per_shard, bool):
            raise ValueError("generation.max_requests_per_shard must be an integer")
        if not 1 <= self.max_requests_per_shard <= 1_000:
            raise ValueError("generation.max_requests_per_shard must be within [1, 1000]")
        if not isinstance(self.max_shard_bytes, int) or isinstance(self.max_shard_bytes, bool):
            raise ValueError("generation.max_shard_bytes must be an integer")
        if not 1 <= self.max_shard_bytes <= 100_000_000:
            raise ValueError("generation.max_shard_bytes must be within [1, 100000000]")
        if not isinstance(self.max_batch_attempts, int) or isinstance(self.max_batch_attempts, bool):
            raise ValueError("generation.max_batch_attempts must be an integer")
        if self.max_batch_attempts < 1:
            raise ValueError("generation.max_batch_attempts must be >= 1")
        if not isinstance(self.scale_authorized, bool):
            raise ValueError("generation.scale_authorized must be a boolean")


@dataclass(frozen=True, slots=True)
class GeneratorResult:
    output: Mapping[str, Any]
    accepted: bool
    reason_codes: tuple[str, ...]

    @property
    def usable(self) -> Any:
        return self.output.get("usable")

    @property
    def query(self) -> Any:
        return self.output.get("query")

    @property
    def answer(self) -> Any:
        return self.output.get("answer")

    @property
    def evidence(self) -> Any:
        return self.output.get("evidence")


def _config(config: GeneratorConfig | Mapping[str, Any] | None) -> GeneratorConfig:
    if config is None:
        return GeneratorConfig()
    if isinstance(config, GeneratorConfig):
        return config
    if isinstance(config, Mapping):
        return GeneratorConfig(**dict(config))
    raise TypeError("generation config must be a GeneratorConfig or mapping")


def _normalized_text(value: object) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value))).strip()


def normalize_query(query: object) -> str:
    return " ".join(token.casefold() for token in _WORD_RE.findall(_normalized_text(query)))


def _content_tokens(value: object) -> list[str]:
    tokens: list[str] = []
    for original in _CONTENT_TOKEN_RE.findall(_normalized_text(value)):
        normalized = original.casefold()
        is_identifier = _is_identifier_like(original)
        if normalized not in _STOP_WORDS and not is_identifier:
            tokens.append(normalized)
    return tokens


def _is_identifier_like(token: str) -> bool:
    return (
        any(character.isdigit() for character in token)
        or "_" in token
        or (len(token) > 1 and token.isupper())
        or any(character.isupper() for character in token[1:])
    )


def _has_copied_content_span(query: object, target_content: object, threshold: int = 8) -> bool:
    query_tokens = _content_tokens(query)
    target_tokens = _content_tokens(target_content)
    if len(query_tokens) <= threshold or len(target_tokens) <= threshold:
        return False
    target_spans = {tuple(target_tokens[index : index + threshold + 1]) for index in range(len(target_tokens) - threshold)}
    return any(tuple(query_tokens[index : index + threshold + 1]) in target_spans for index in range(len(query_tokens) - threshold))


def _records(context: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]] | Any) -> list[Mapping[str, Any]]:
    if hasattr(context, "to_dicts"):
        rows = context.to_dicts()
    else:
        rows = list(context)
    if not all(isinstance(row, Mapping) for row in rows):
        raise TypeError("generation context rows must be mappings")
    return rows


def _slot_value(slot: Mapping[str, Any], key: str) -> str:
    value = slot.get(key)
    if value is None or not str(value):
        raise ValueError(f"target slot is missing {key}")
    return str(value)


def _context_mismatch_codes(slot: Mapping[str, Any], context: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    codes: list[str] = []
    if len(context) != 3:
        codes.append("context_count_invalid")
        return tuple(codes)
    expected_indices = {0, 1, 2}
    if {row.get("context_index") for row in context} != expected_indices:
        codes.append("context_index_invalid")
    target_article_id = str(slot.get("article_id", ""))
    target_passage_id = str(slot.get("passage_id", ""))
    target_source_hash = str(slot.get("source_hash", ""))
    candidate_ids: set[str] = set()
    for row in context:
        if str(row.get("target_article_id", "")) != target_article_id or str(row.get("target_passage_id", "")) != target_passage_id:
            codes.append("target_identity_mismatch")
            break
    for row in context:
        if str(row.get("target_source_hash", "")) != target_source_hash:
            codes.append("source_hash_mismatch")
            break
    for row in context:
        candidate_id = str(row.get("candidate_passage_id", ""))
        if not candidate_id or candidate_id == target_passage_id or candidate_id in candidate_ids:
            codes.append("context_candidate_invalid")
            break
        candidate_ids.add(candidate_id)
    return tuple(codes)


def _require_valid_context(slot: Mapping[str, Any], context: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    mismatch_codes = _context_mismatch_codes(slot, context)
    if mismatch_codes:
        if mismatch_codes == ("context_count_invalid",):
            raise ValueError("generation context must contain exactly three candidate rows")
        raise ValueError(f"generation context is invalid: {', '.join(mismatch_codes)}")
    return sorted(context, key=lambda row: int(row["context_index"]))


def _render_user_prompt(slot: Mapping[str, Any], context: Sequence[Mapping[str, Any]]) -> str:
    neighbor_passages = "\n\n".join(
        "\n".join(
            (
                f"PASSAGE ID: {row['candidate_passage_id']}",
                str(row.get("candidate_serialized_passage") or row.get("candidate_content") or ""),
            )
        )
        for row in context
    )
    target_text = str(slot.get("serialized_passage") or slot["content"])
    return "\n".join(
        (
            "LANGUAGE: ru",
            f"REQUESTED INTENT: {slot['requested_intent']}",
            "",
            f"TARGET PASSAGE ID: {slot['passage_id']}",
            "TARGET:",
            target_text,
            "",
            "NEARBY NON-TARGET PASSAGES:",
            neighbor_passages,
        )
    )


def _prompt_hash(user_prompt: str) -> str:
    return hashlib.sha256(f"{GENERATOR_SYSTEM_PROMPT}\n\n{user_prompt}".encode("utf-8")).hexdigest()


def _custom_id(slot: Mapping[str, Any], prompt_hash: str, attempt: int) -> str:
    value = ":".join(
        (
            str(slot["article_id"]),
            str(slot["passage_id"]),
            str(slot["requested_intent"]),
            prompt_hash,
            str(attempt),
        )
    )
    return f"gen-{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def build_generator_request(
    slot: Mapping[str, Any],
    generation_context: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]] | Any,
    config: GeneratorConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one OpenAI Batch request for a prepared target slot."""
    if not isinstance(slot, Mapping):
        raise TypeError("target slot must be a mapping")
    active_config = _config(config)
    for key in ("article_id", "passage_id", "source_hash", "content", "requested_intent"):
        _slot_value(slot, key)
    if str(slot["requested_intent"]) not in INTENTS:
        raise ValueError(f"unsupported requested intent: {slot['requested_intent']}")
    context = _require_valid_context(slot, _records(generation_context))
    user_prompt = _render_user_prompt(slot, context)
    prompt_hash = _prompt_hash(user_prompt)
    body: dict[str, Any] = {
        "model": active_config.model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": GENERATOR_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        "reasoning": {"effort": active_config.reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "habr_ir_generator_result",
                "strict": True,
                "schema": GENERATOR_SCHEMA,
            }
        },
        "store": False,
        "metadata": {
            "prompt_hash": prompt_hash,
            "article_id": str(slot["article_id"]),
            "passage_id": str(slot["passage_id"]),
            "source_hash": str(slot["source_hash"]),
            "generation_attempt": str(active_config.attempt),
        },
    }
    if active_config.prompt_cache_mode == "explicit":
        body["prompt_cache_options"] = {"mode": "explicit"}
    return {
        "custom_id": _custom_id(slot, prompt_hash, active_config.attempt),
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def _schema_is_valid(output: Mapping[str, Any]) -> bool:
    if set(output) != set(GENERATOR_SCHEMA["required"]):
        return False
    if not isinstance(output["usable"], bool):
        return False
    if not all(
        isinstance(output[key], str) for key in ("reason", "query", "answer", "evidence", "requested_intent", "actual_intent")
    ):
        return False
    if output["reason"] not in {"ok", *UNUSABLE_REASONS}:
        return False
    if output["requested_intent"] not in INTENTS or output["actual_intent"] not in INTENTS:
        return False
    return isinstance(output["disambiguators"], list) and all(isinstance(value, str) for value in output["disambiguators"])


def _contains_banned_context_phrase(query: str) -> bool:
    normalized = _normalized_text(query).casefold()
    return any(pattern.search(normalized) for pattern in _BANNED_CONTEXT_PATTERNS)


def validate_generator_result(
    output: Mapping[str, Any],
    slot: Mapping[str, Any],
    generation_context: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]] | Any | None = None,
    *,
    accepted_normalized_queries: Iterable[str] = (),
    config: GeneratorConfig | Mapping[str, Any] | None = None,
) -> GeneratorResult:
    """Apply deterministic generator gates without changing the model output."""
    if not isinstance(output, Mapping):
        return GeneratorResult(output={}, accepted=False, reason_codes=("schema_invalid",))
    if not isinstance(slot, Mapping):
        raise TypeError("target slot must be a mapping")
    active_config = _config(config)
    if not _schema_is_valid(output):
        return GeneratorResult(output=output, accepted=False, reason_codes=("schema_invalid",))

    reason_codes: list[str] = []
    if generation_context is None:
        reason_codes.append("generation_context_missing")
    else:
        reason_codes.extend(_context_mismatch_codes(slot, _records(generation_context)))
    token_count = slot.get("token_count")
    if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count > active_config.accepted_max_tokens:
        reason_codes.append("target_length_exceeded")

    if not output["usable"]:
        reason_codes.append("usable_false")
        if output["reason"] not in UNUSABLE_REASONS or any(output[key] != "" for key in ("query", "answer", "evidence")):
            reason_codes.append("unusable_contract_invalid")
        return GeneratorResult(output=output, accepted=False, reason_codes=tuple(dict.fromkeys(reason_codes)))

    if output["reason"] != "ok":
        reason_codes.append("usable_reason_invalid")
    normalized_fields = {key: _normalized_text(output[key]) for key in ("query", "answer", "evidence")}
    for key, value in normalized_fields.items():
        if not value:
            reason_codes.append(f"empty_{key}")
    normalized_evidence = normalized_fields["evidence"]
    raw_content = str(slot.get("content", ""))
    raw_evidence = output["evidence"]
    if not normalized_evidence or raw_evidence not in raw_content:
        reason_codes.append("evidence_not_substring")
    overlap_prefix_chars = slot.get("overlap_prefix_chars", 0)
    if isinstance(overlap_prefix_chars, int) and not isinstance(overlap_prefix_chars, bool) and overlap_prefix_chars > 0:
        raw_overlap = raw_content[:overlap_prefix_chars]
        if normalized_evidence and raw_evidence in raw_overlap:
            reason_codes.append("evidence_overlap_only")
    query_words = output["query"].split()
    if not 5 <= len(query_words) <= 30:
        reason_codes.append("query_word_count")
    if _contains_banned_context_phrase(output["query"]):
        reason_codes.append("banned_context_phrase")
    if output["requested_intent"] != slot.get("requested_intent") or output["actual_intent"] != output["requested_intent"]:
        reason_codes.append("intent_mismatch")
    normalized_query = normalize_query(output["query"])
    if normalized_query in set(accepted_normalized_queries):
        reason_codes.append("duplicate_normalized_query")
    if _has_copied_content_span(output["query"], slot.get("content", "")):
        reason_codes.append("copied_content_span_gt_8")
    return GeneratorResult(output=output, accepted=not reason_codes, reason_codes=tuple(dict.fromkeys(reason_codes)))


__all__ = [
    "GENERATOR_SCHEMA",
    "GENERATOR_SYSTEM_PROMPT",
    "GeneratorConfig",
    "GeneratorResult",
    "build_generator_request",
    "normalize_query",
    "validate_generator_result",
]
