from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import httpx

from justatom.agentic.schemas import (
    AgentAction,
    AttemptTrace,
    CallStatus,
    CostUsage,
    ErrorCategory,
    ErrorTrace,
    EvidenceDocument,
    PlannerDecision,
    PlannerReply,
    PlannerRequest,
    TokenUsage,
    sha256_text,
)

DEFAULT_SYSTEM_PROMPT = """You are the planner for a retrieval-augmented question answering agent.
Choose exactly one action for the current step:
- search: provide a focused, non-empty search query when more evidence is needed.
- answer: provide a complete, non-empty answer grounded in the supplied context.

Every output must contain exactly these five fields: action, query, answer, reason, cited_document_ids.
For search, query must be a non-empty string, answer must be null, and cited_document_ids must be [].
For answer, query must be null, answer must be a non-empty string, and cited_document_ids must contain
only identifiers from context_documents. reason may be a string or null.

Return only the JSON object required by the response schema. Never invent document identifiers."""


_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "const": AgentAction.SEARCH.value},
                "query": {"type": "string", "minLength": 1},
                "answer": {"type": "null"},
                "reason": {"type": ["string", "null"]},
                "cited_document_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "maxItems": 0,
                },
            },
            "required": ["action", "query", "answer", "reason", "cited_document_ids"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "const": AgentAction.ANSWER.value},
                "query": {"type": "null"},
                "answer": {"type": "string", "minLength": 1},
                "reason": {"type": ["string", "null"]},
                "cited_document_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
            },
            "required": ["action", "query", "answer", "reason", "cited_document_ids"],
            "additionalProperties": False,
        },
    ],
}

_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "agentic_planner_decision",
        "strict": True,
        "schema": _DECISION_SCHEMA,
    },
}

_MISSING = object()


class OpenAICompatibleChatError(RuntimeError):
    """A sanitized failure while calling an OpenAI-compatible chat endpoint."""

    def __init__(self, message: str, *, attempts: tuple[AttemptTrace, ...]) -> None:
        super().__init__(message)
        self.attempts = attempts


class OpenAICompatibleResponseError(OpenAICompatibleChatError):
    """A sanitized failure to decode or validate a chat completion."""


class _InvalidResponse(ValueError):
    pass


class _ResponseTooLarge(ValueError):
    pass


class OpenAICompatibleChatBackend:
    """Small, dependency-light client for OpenAI-compatible chat completions."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_response_bytes: int = 1_048_576,
        seed: int | None = None,
        system_prompt: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature < 0
        ):
            raise ValueError("temperature must be a finite non-negative number")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int) or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or null")
        if system_prompt is not None and (not isinstance(system_prompt, str) or not system_prompt.strip()):
            raise ValueError("system_prompt must be a non-empty string or null")
        if api_key is not None and not isinstance(api_key, str):
            raise ValueError("api_key must be a string or null")
        if transport is not None and client is not None:
            raise ValueError("transport and client are mutually exclusive")

        self.base_url = base_url.strip().rstrip("/")
        self.model = model.strip()
        self._endpoint = f"{self.base_url}/chat/completions"
        self._timeout_seconds = float(timeout_seconds)
        self._temperature = float(temperature)
        self._max_tokens = max_tokens
        self._max_response_bytes = max_response_bytes
        self._seed = seed
        self._system_prompt = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
        self._prompt_fingerprint = sha256_text(self._system_prompt)
        config_payload = {
            "backend": "openai-compatible",
            "base_url_sha256": sha256_text(self.base_url),
            "model": self.model,
            "timeout_seconds": self._timeout_seconds,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "max_response_bytes": self._max_response_bytes,
            "seed": self._seed,
            "prompt_fingerprint": self._prompt_fingerprint,
        }
        self._config_fingerprint = sha256_text(
            json.dumps(config_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        )
        self._request_headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = client if client is not None else httpx.AsyncClient(timeout=self._timeout_seconds, transport=transport)
        self._closed = False

    @property
    def backend_name(self) -> str:
        return "openai-compatible"

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def prompt_fingerprint(self) -> str:
        return self._prompt_fingerprint

    @property
    def config_fingerprint(self) -> str:
        return self._config_fingerprint

    async def plan(self, request: PlannerRequest) -> PlannerReply:
        started = time.perf_counter()
        if self._closed:
            error = ErrorTrace(
                component=self.backend_name,
                category=ErrorCategory.BACKEND,
                code="client_closed",
                retryable=False,
            )
            attempt = self._attempt(started, status=CallStatus.ERROR, error=error)
            raise OpenAICompatibleChatError("OpenAI-compatible chat backend is closed", attempts=(attempt,))

        try:
            user_content = _encode_request(request)
        except (AttributeError, TypeError, ValueError, OverflowError):
            error = ErrorTrace(
                component=self.backend_name,
                category=ErrorCategory.VALIDATION,
                code="invalid_request",
                retryable=False,
            )
            attempt = self._attempt(started, status=CallStatus.ERROR, error=error)
            raise OpenAICompatibleChatError("Planner request could not be serialized", attempts=(attempt,)) from None

        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "response_format": deepcopy(_RESPONSE_FORMAT),
        }
        if self._seed is not None:
            request_body["seed"] = self._seed

        response: httpx.Response | None = None
        provider_request_id: str | None = None
        try:
            async with self._client.stream(
                "POST",
                self._endpoint,
                json=request_body,
                headers=self._request_headers,
                timeout=self._timeout_seconds,
            ) as response:
                provider_request_id = _provider_request_id(response)
                response.raise_for_status()
                response_body = await _read_bounded_response(response, self._max_response_bytes)
        except httpx.TimeoutException as exc:
            error = ErrorTrace(
                component=self.backend_name,
                category=ErrorCategory.TIMEOUT,
                code="http_timeout",
                exception_type=type(exc).__name__,
                retryable=True,
            )
            attempt = self._attempt(
                started,
                status=CallStatus.TIMEOUT,
                provider_request_id=provider_request_id,
                error=error,
            )
            raise OpenAICompatibleChatError("OpenAI-compatible chat request timed out", attempts=(attempt,)) from None
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            error = ErrorTrace(
                component=self.backend_name,
                category=ErrorCategory.BACKEND,
                code="http_status",
                exception_type=type(exc).__name__,
                retryable=_retryable_status(status_code),
            )
            attempt = self._attempt(
                started,
                status=CallStatus.ERROR,
                provider_request_id=provider_request_id,
                error=error,
            )
            raise OpenAICompatibleChatError(
                f"OpenAI-compatible chat request failed with HTTP {status_code}",
                attempts=(attempt,),
            ) from None
        except _ResponseTooLarge as exc:
            error = ErrorTrace(
                component=self.backend_name,
                category=ErrorCategory.BACKEND,
                code="response_too_large",
                exception_type=type(exc).__name__,
                retryable=False,
            )
            attempt = self._attempt(
                started,
                status=CallStatus.ERROR,
                provider_request_id=provider_request_id,
                error=error,
            )
            raise OpenAICompatibleResponseError(
                "OpenAI-compatible chat response exceeded max_response_bytes",
                attempts=(attempt,),
            ) from None
        except httpx.HTTPError as exc:
            error = ErrorTrace(
                component=self.backend_name,
                category=ErrorCategory.BACKEND,
                code="http_error",
                exception_type=type(exc).__name__,
                retryable=True,
            )
            attempt = self._attempt(
                started,
                status=CallStatus.ERROR,
                provider_request_id=provider_request_id,
                error=error,
            )
            raise OpenAICompatibleChatError("OpenAI-compatible chat request failed", attempts=(attempt,)) from None
        except Exception as exc:
            error = ErrorTrace(
                component=self.backend_name,
                category=ErrorCategory.BACKEND,
                code="client_error",
                exception_type=type(exc).__name__,
                retryable=False,
            )
            attempt = self._attempt(
                started,
                status=CallStatus.ERROR,
                provider_request_id=provider_request_id,
                error=error,
            )
            raise OpenAICompatibleChatError("OpenAI-compatible chat client failed", attempts=(attempt,)) from None

        try:
            response_payload = json.loads(
                response_body,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            error = ErrorTrace(
                component=self.backend_name,
                category=ErrorCategory.PARSE,
                code="invalid_json",
                exception_type=type(exc).__name__,
                retryable=False,
            )
            attempt = self._attempt(
                started,
                status=CallStatus.ERROR,
                provider_request_id=provider_request_id,
                error=error,
            )
            raise OpenAICompatibleResponseError(
                "OpenAI-compatible chat response was not valid JSON",
                attempts=(attempt,),
            ) from None

        provider_request_id = provider_request_id or _payload_request_id(response_payload)

        try:
            decision, response_model, finish_reason, usage, cost, cache_hit = _parse_completion(response_payload, self.model)
        except (TypeError, ValueError, OverflowError) as exc:
            error = ErrorTrace(
                component=self.backend_name,
                category=ErrorCategory.VALIDATION,
                code="invalid_response",
                exception_type=type(exc).__name__,
                retryable=False,
            )
            attempt = self._attempt(
                started,
                status=CallStatus.ERROR,
                provider_request_id=provider_request_id,
                error=error,
            )
            raise OpenAICompatibleResponseError(
                "OpenAI-compatible chat response did not match the planner schema",
                attempts=(attempt,),
            ) from None

        attempt = self._attempt(
            started,
            status=CallStatus.OK,
            provider_request_id=provider_request_id,
        )
        return PlannerReply(
            decision=decision,
            model=response_model,
            provider_request_id=provider_request_id,
            usage=usage,
            cost=cost,
            attempts=(attempt,),
            cache_hit=cache_hit,
            finish_reason=finish_reason,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._client.aclose()

    @staticmethod
    def _attempt(
        started: float,
        *,
        status: CallStatus,
        provider_request_id: str | None = None,
        error: ErrorTrace | None = None,
    ) -> AttemptTrace:
        return AttemptTrace(
            attempt_index=0,
            status=status,
            latency_ms=max(0.0, (time.perf_counter() - started) * 1000.0),
            provider_request_id=provider_request_id,
            error=error,
        )


def _encode_request(request: PlannerRequest) -> str:
    payload = {
        "question": request.question,
        "observations": [
            {
                "query": observation.query,
                # Passage text lives only in the bounded context below. Keeping
                # observations as ranked references avoids duplicating prompt
                # content on every hop and preserves max_context_chars as the
                # actual content budget.
                "documents": [_document_reference_payload(document) for document in observation.documents],
            }
            for observation in request.observations
        ],
        "context_documents": [_document_payload(document) for document in request.context_documents],
        "remaining_retrieval_calls": request.remaining_retrieval_calls,
        "remaining_steps": request.remaining_steps,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _document_payload(document: EvidenceDocument) -> dict[str, Any]:
    return {
        "document_id": document.document_id,
        "content": document.content,
        "score": document.score,
        "rank": document.rank,
        "retrieval_index": document.retrieval_index,
    }


def _document_reference_payload(document: EvidenceDocument) -> dict[str, Any]:
    return {
        "document_id": document.document_id,
        "score": document.score,
        "rank": document.rank,
        "retrieval_index": document.retrieval_index,
    }


def _provider_request_id(response: httpx.Response) -> str | None:
    for header_name in ("x-request-id", "request-id", "x-amzn-requestid"):
        value = response.headers.get(header_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _payload_request_id(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    for key in ("request_id", "id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _retryable_status(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429} or status_code >= 500


async def _read_bounded_response(response: httpx.Response, max_response_bytes: int) -> bytes:
    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError:
            declared_size = None
        if declared_size is not None and declared_size > max_response_bytes:
            raise _ResponseTooLarge("declared response size exceeds the configured limit")

    body = bytearray()
    chunk_size = min(65_536, max_response_bytes + 1)
    async for chunk in response.aiter_bytes(chunk_size=chunk_size):
        if len(body) + len(chunk) > max_response_bytes:
            raise _ResponseTooLarge("streamed response size exceeds the configured limit")
        body.extend(chunk)
    return bytes(body)


def _parse_completion(
    payload: Any,
    configured_model: str,
) -> tuple[PlannerDecision, str, str | None, TokenUsage | None, CostUsage | None, bool | None]:
    if not isinstance(payload, Mapping):
        raise _InvalidResponse("completion must be an object")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise _InvalidResponse("completion must contain a choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise _InvalidResponse("choice must contain a message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise _InvalidResponse("message content must be a non-empty string")

    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise _InvalidResponse("finish_reason must be a string or null")

    response_model = payload.get("model", configured_model)
    if response_model is None:
        response_model = configured_model
    if not isinstance(response_model, str) or not response_model.strip():
        raise _InvalidResponse("model must be a non-empty string or null")

    decision = _parse_decision(content)
    usage, cost, cache_hit = _parse_usage(payload.get("usage", _MISSING))
    return decision, response_model, finish_reason, usage, cost, cache_hit


def _parse_decision(content: str) -> PlannerDecision:
    try:
        value = json.loads(
            content,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise _InvalidResponse("decision content must be strict JSON") from exc
    if not isinstance(value, Mapping):
        raise _InvalidResponse("decision must be an object")

    allowed_keys = {"action", "query", "answer", "reason", "cited_document_ids"}
    if set(value) != allowed_keys:
        raise _InvalidResponse("decision fields must exactly match the response schema")

    action_value = value.get("action")
    try:
        action = AgentAction(action_value)
    except (TypeError, ValueError) as exc:
        raise _InvalidResponse("decision action is invalid") from exc

    query = value.get("query")
    answer = value.get("answer")
    reason = value.get("reason")
    cited_document_ids = value.get("cited_document_ids", [])
    for field_name, field_value in (("query", query), ("answer", answer), ("reason", reason)):
        if field_value is not None and not isinstance(field_value, str):
            raise _InvalidResponse(f"decision {field_name} must be a string or null")
    if reason is not None and not reason.strip():
        raise _InvalidResponse("decision reason must be a non-empty string or null")
    if not isinstance(cited_document_ids, list) or any(
        not isinstance(document_id, str) or not document_id.strip() for document_id in cited_document_ids
    ):
        raise _InvalidResponse("cited_document_ids must contain non-empty strings")

    try:
        return PlannerDecision(
            action=action,
            query=query,
            answer=answer,
            reason=reason,
            cited_document_ids=tuple(cited_document_ids),
        )
    except ValueError as exc:
        raise _InvalidResponse("decision fields do not match its action") from exc


def _parse_usage(value: Any) -> tuple[TokenUsage | None, CostUsage | None, bool | None]:
    if value is _MISSING or value is None:
        return None, None, None
    if not isinstance(value, Mapping):
        raise _InvalidResponse("usage must be an object or null")

    input_tokens = _first_optional_int(value, "prompt_tokens", "input_tokens")
    output_tokens = _first_optional_int(value, "completion_tokens", "output_tokens")
    total_tokens = _optional_int(value.get("total_tokens"), "total_tokens") if "total_tokens" in value else None
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    cached_input_tokens = _first_optional_int(
        value,
        "cached_input_tokens",
        "prompt_cache_hit_tokens",
        "cached_tokens",
    )
    prompt_details = value.get("prompt_tokens_details", value.get("input_tokens_details", _MISSING))
    if prompt_details is not _MISSING and prompt_details is not None:
        if not isinstance(prompt_details, Mapping):
            raise _InvalidResponse("input token details must be an object or null")
        if cached_input_tokens is None and "cached_tokens" in prompt_details:
            cached_input_tokens = _optional_int(prompt_details.get("cached_tokens"), "cached_tokens")

    reasoning_tokens = _first_optional_int(value, "reasoning_tokens")
    completion_details = value.get("completion_tokens_details", value.get("output_tokens_details", _MISSING))
    if completion_details is not _MISSING and completion_details is not None:
        if not isinstance(completion_details, Mapping):
            raise _InvalidResponse("output token details must be an object or null")
        if reasoning_tokens is None and "reasoning_tokens" in completion_details:
            reasoning_tokens = _optional_int(completion_details.get("reasoning_tokens"), "reasoning_tokens")

    cache_hit_value = value.get("cache_hit", _MISSING)
    if cache_hit_value is _MISSING:
        cache_hit = None if cached_input_tokens is None else cached_input_tokens > 0
    elif cache_hit_value is None:
        cache_hit = None
    elif isinstance(cache_hit_value, bool):
        cache_hit = cache_hit_value
    else:
        raise _InvalidResponse("cache_hit must be a boolean or null")

    usage = TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        reasoning_tokens=reasoning_tokens,
        source="provider",
    )
    return usage, _parse_provider_cost(value), cache_hit


def _parse_provider_cost(usage: Mapping[str, Any]) -> CostUsage | None:
    raw_cost = usage.get("cost_usd", usage.get("cost", _MISSING))
    if (
        raw_cost is _MISSING
        or raw_cost is None
        or isinstance(raw_cost, bool)
        or not isinstance(raw_cost, (int, float))
        or not math.isfinite(raw_cost)
        or raw_cost < 0
    ):
        return None
    pricing_id = usage.get("pricing_id")
    if not isinstance(pricing_id, str) or not pricing_id.strip():
        pricing_id = None
    return CostUsage(usd=float(raw_cost), source="provider", pricing_id=pricing_id)


def _first_optional_int(mapping: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in mapping:
            return _optional_int(mapping.get(key), key)
    return None


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise _InvalidResponse(f"{name} must be a non-negative integer or null")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _InvalidResponse("JSON object keys must be unique")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise _InvalidResponse(f"non-standard JSON constant {value!r}")


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "OpenAICompatibleChatBackend",
    "OpenAICompatibleChatError",
    "OpenAICompatibleResponseError",
]
