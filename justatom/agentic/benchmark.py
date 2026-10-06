from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from justatom.agentic.contracts import AgentRetriever


class SearchBudgetExceeded(RuntimeError):
    """Raised when a search would exceed the session's retrieval-call budget."""


class InvalidSearchQuery(ValueError):
    """Invalid model input, distinct from a retriever's operational failure."""


@dataclass(frozen=True, slots=True)
class SearchBudget:
    top_k: int = 5
    max_searches: int = 4
    max_context_documents: int = 20
    max_document_chars: int = 2_000
    max_context_chars: int = 24_000
    max_query_chars: int = 4_000

    def __post_init__(self) -> None:
        for name in (
            "top_k",
            "max_searches",
            "max_context_documents",
            "max_document_chars",
            "max_context_chars",
            "max_query_chars",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class SearchSession:
    """Bounded state for the search tool used by an agent benchmark sample."""

    def __init__(self, retriever: AgentRetriever, budget: SearchBudget) -> None:
        if not isinstance(budget, SearchBudget):
            raise TypeError("budget must be a SearchBudget")
        self._retriever = retriever
        self._budget = budget
        self._lock = asyncio.Lock()
        self._context: dict[str, str] = {}
        self._context_chars = 0
        self.calls: list[dict[str, Any]] = []

    @property
    def context_documents(self) -> list[dict[str, str]]:
        return [{"chunk_id": chunk_id, "content": content} for chunk_id, content in self._context.items()]

    @property
    def context_ids(self) -> list[str]:
        return list(self._context)

    async def search(self, query: str) -> dict[str, Any]:
        validated_query = _validate_query(query, self._budget.max_query_chars)
        async with self._lock:
            if len(self.calls) >= self._budget.max_searches:
                raise SearchBudgetExceeded(f"search budget exhausted after {self._budget.max_searches} calls")

            record: dict[str, Any] = {
                "index": len(self.calls) + 1,
                "query": validated_query,
                "status": "pending",
                "latency_ms": 0.0,
                "documents": [],
                "error_type": None,
            }
            self.calls.append(record)
            started_ns = time.perf_counter_ns()
            try:
                returned = await self._retriever.retrieve(validated_query, top_k=self._budget.top_k)
                matches, contents = _normalize_documents(returned, self._budget.top_k)
            except BaseException as error:
                record["status"] = "error"
                record["latency_ms"] = _elapsed_ms(started_ns)
                record["error_type"] = type(error).__name__
                raise

            new_context = self._admit_context(contents, matches)
            record["status"] = "ok"
            record["latency_ms"] = _elapsed_ms(started_ns)
            record["documents"] = [dict(match) for match in matches]
            return {
                "query": validated_query,
                "matches": [dict(match) for match in matches],
                "new_context": new_context,
                "remaining_searches": self._budget.max_searches - len(self.calls),
            }

    def _admit_context(
        self,
        contents: Sequence[str],
        matches: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, str]]:
        new_context: list[dict[str, str]] = []
        for content, match in zip(contents, matches, strict=True):
            chunk_id = match["chunk_id"]
            if chunk_id in self._context:
                continue
            if len(self._context) >= self._budget.max_context_documents:
                continue
            remaining_chars = self._budget.max_context_chars - self._context_chars
            if remaining_chars <= 0:
                continue
            retained_content = content[: self._budget.max_document_chars][:remaining_chars]
            self._context[chunk_id] = retained_content
            self._context_chars += len(retained_content)
            new_context.append({"chunk_id": chunk_id, "content": retained_content})
        return new_context


def score_searches(
    calls: Sequence[Mapping[str, Any]],
    context_ids: Sequence[str],
    gold_ids: Sequence[str],
) -> dict[str, Any]:
    """Score retrieval calls and delivered context without exposing labels to search."""

    validated_gold = _validate_identifiers(gold_ids, "gold_ids", require_nonempty=True, require_unique=True)
    validated_context = _validate_identifiers(context_ids, "context_ids")
    if isinstance(calls, (str, bytes)) or not isinstance(calls, Sequence):
        raise TypeError("calls must be a sequence")

    gold = set(validated_gold)
    cumulative_ids: list[str] = []
    per_hop: list[dict[str, Any]] = []
    cumulative: list[dict[str, Any]] = []
    first_complete_hop: int | None = None

    for hop, call in enumerate(calls, start=1):
        if not isinstance(call, Mapping):
            raise TypeError("calls entries must be mappings")
        hop_ids: list[str] = []
        if call.get("status") == "ok":
            documents = call.get("documents")
            if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
                raise ValueError("successful call documents must be a sequence")
            for document in documents:
                if not isinstance(document, Mapping):
                    raise ValueError("call documents must be mappings")
                hop_ids.extend(_validate_identifiers([document.get("chunk_id")], "call document chunk_id"))

        per_hop.append(_recall_metrics(hop_ids, gold))
        cumulative_ids.extend(hop_ids)
        cumulative_metrics = _recall_metrics(cumulative_ids, gold)
        cumulative.append(cumulative_metrics)
        if first_complete_hop is None and cumulative_metrics["all_gold"]:
            first_complete_hop = hop

    return {
        "per_hop": per_hop,
        "cumulative": cumulative,
        "final_context": _recall_metrics(validated_context, gold),
        "first_complete_hop": first_complete_hop,
        "search_calls": len(calls),
    }


def _validate_query(query: Any, max_chars: int) -> str:
    if not isinstance(query, str) or not query.strip():
        raise InvalidSearchQuery("query must be a non-empty string")
    if len(query) > max_chars:
        raise InvalidSearchQuery(f"query exceeds max_query_chars={max_chars}")
    return query


def _normalize_documents(documents: Any, top_k: int) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(documents, list):
        raise TypeError("retriever must return a list of documents")
    matches: list[dict[str, Any]] = []
    contents: list[str] = []
    for rank, document in enumerate(documents[:top_k], start=1):
        chunk_id = getattr(document, "id", None)
        if not isinstance(chunk_id, str):
            raise TypeError("retriever document id must be a string")
        if not chunk_id.strip():
            raise ValueError("retriever document id must not be blank")
        content = getattr(document, "content", None)
        if not isinstance(content, str):
            raise TypeError("retriever document content must be a string")
        contents.append(content)
        matches.append(
            {
                "chunk_id": chunk_id,
                "rank": rank,
                "score": _finite_score(getattr(document, "score", None)),
            }
        )
    return matches, contents


def _finite_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return score if math.isfinite(score) else None


def _validate_identifiers(
    values: Any,
    name: str,
    *,
    require_nonempty: bool = False,
    require_unique: bool = False,
) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    validated: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must contain only non-empty strings")
        validated.append(value)
    if require_nonempty and not validated:
        raise ValueError(f"{name} must not be empty")
    if require_unique and len(set(validated)) != len(validated):
        raise ValueError(f"{name} must contain unique identifiers")
    return validated


def _recall_metrics(retrieved_ids: Sequence[str], gold: set[str]) -> dict[str, Any]:
    hits = len(set(retrieved_ids) & gold)
    gold_count = len(gold)
    return {
        "all_gold": hits == gold_count,
        "recall": hits / gold_count,
        "hits": hits,
        "gold_count": gold_count,
        "retrieved_count": len(retrieved_ids),
    }


def _elapsed_ms(started_ns: int) -> float:
    return max((time.perf_counter_ns() - started_ns) / 1_000_000.0, 0.0)


__all__ = ["SearchBudget", "SearchBudgetExceeded", "SearchSession", "score_searches"]
