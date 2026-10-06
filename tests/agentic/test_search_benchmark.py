from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, dataclass
from typing import Any

import pytest

from justatom.agentic.benchmark import SearchBudget, SearchBudgetExceeded, SearchSession, score_searches


@dataclass
class FakeDocument:
    id: Any
    content: Any
    score: Any = None
    meta: dict[str, Any] | None = None


class ScriptedRetriever:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, int]] = []
        self.close_calls = 0

    async def retrieve(self, query: str, *, top_k: int = 5, **kwargs: Any) -> list[FakeDocument]:
        assert kwargs == {}
        self.requests.append((query, top_k))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    async def close(self) -> None:
        self.close_calls += 1


def test_search_budget_defaults_are_frozen_and_validate_every_positive_integer() -> None:
    budget = SearchBudget()

    assert budget == SearchBudget(
        top_k=5,
        max_searches=4,
        max_context_documents=20,
        max_document_chars=2_000,
        max_context_chars=24_000,
        max_query_chars=4_000,
    )
    with pytest.raises(FrozenInstanceError):
        budget.top_k = 9  # type: ignore[misc]

    for field in (
        "top_k",
        "max_searches",
        "max_context_documents",
        "max_document_chars",
        "max_context_chars",
        "max_query_chars",
    ):
        for invalid in (0, -1, True, 1.5, "1"):
            with pytest.raises(ValueError, match=field):
                SearchBudget(**{field: invalid})


def test_search_clips_results_and_builds_first_seen_bounded_context_without_leaking_metadata() -> None:
    retriever = ScriptedRetriever(
        [
            [
                FakeDocument("a", "abcdef", 0.9, {"gold": True}),
                FakeDocument("a", "duplicate", 0.8, {"private": "value"}),
                FakeDocument("b", "wxyz", float("inf")),
                FakeDocument("c", "not admitted", 0.6),
                FakeDocument(7, {"invalid": "but outside top_k"}, 1.0),
            ],
            [FakeDocument("c", "still rejected", 0.7), FakeDocument("a", "do not resend", 0.5)],
        ]
    )
    session = SearchSession(
        retriever,
        SearchBudget(
            top_k=4,
            max_searches=2,
            max_context_documents=2,
            max_document_chars=4,
            max_context_chars=7,
            max_query_chars=20,
        ),
    )

    first = asyncio.run(session.search("  focused query  "))
    second = asyncio.run(session.search("follow-up"))

    assert retriever.requests == [("  focused query  ", 4), ("follow-up", 4)]
    assert first == {
        "query": "  focused query  ",
        "matches": [
            {"chunk_id": "a", "rank": 1, "score": 0.9},
            {"chunk_id": "a", "rank": 2, "score": 0.8},
            {"chunk_id": "b", "rank": 3, "score": None},
            {"chunk_id": "c", "rank": 4, "score": 0.6},
        ],
        "new_context": [
            {"chunk_id": "a", "content": "abcd"},
            {"chunk_id": "b", "content": "wxy"},
        ],
        "remaining_searches": 1,
    }
    assert second == {
        "query": "follow-up",
        "matches": [
            {"chunk_id": "c", "rank": 1, "score": 0.7},
            {"chunk_id": "a", "rank": 2, "score": 0.5},
        ],
        "new_context": [],
        "remaining_searches": 0,
    }
    assert session.context_documents == [
        {"chunk_id": "a", "content": "abcd"},
        {"chunk_id": "b", "content": "wxy"},
    ]
    assert session.context_ids == ["a", "b"]
    assert [call["index"] for call in session.calls] == [1, 2]
    assert [call["status"] for call in session.calls] == ["ok", "ok"]
    assert session.calls[0]["documents"] == first["matches"]
    assert session.calls[0]["error_type"] is None
    assert session.calls[0]["latency_ms"] >= 0
    assert "gold" not in repr(first)
    assert "private" not in repr(first)
    assert retriever.close_calls == 0


def test_context_document_limit_is_independent_of_the_character_limit() -> None:
    retriever = ScriptedRetriever([[FakeDocument("a", "one"), FakeDocument("b", "two"), FakeDocument("c", "three")]])
    session = SearchSession(
        retriever,
        SearchBudget(top_k=3, max_searches=1, max_context_documents=2, max_context_chars=100),
    )

    result = asyncio.run(session.search("query"))

    assert [match["chunk_id"] for match in result["matches"]] == ["a", "b", "c"]
    assert result["new_context"] == [
        {"chunk_id": "a", "content": "one"},
        {"chunk_id": "b", "content": "two"},
    ]
    assert session.context_ids == ["a", "b"]


def test_invalid_queries_are_rejected_without_spending_search_budget() -> None:
    retriever = ScriptedRetriever([[FakeDocument("a", "text")]])
    session = SearchSession(retriever, SearchBudget(max_searches=1, max_query_chars=4))

    for invalid in (None, "", "   ", "abcde"):
        with pytest.raises(ValueError, match="query"):
            asyncio.run(session.search(invalid))  # type: ignore[arg-type]

    assert session.calls == []
    assert retriever.requests == []
    assert asyncio.run(session.search("okay"))["remaining_searches"] == 0


def test_backend_failure_is_recorded_without_exception_text_and_spends_the_budget() -> None:
    retriever = ScriptedRetriever([RuntimeError("secret backend details")])
    session = SearchSession(retriever, SearchBudget(max_searches=1))

    with pytest.raises(RuntimeError, match="secret backend details"):
        asyncio.run(session.search("query"))

    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["index"] == 1
    assert call["query"] == "query"
    assert call["status"] == "error"
    assert call["documents"] == []
    assert call["error_type"] == "RuntimeError"
    assert "secret backend details" not in repr(call)
    with pytest.raises(SearchBudgetExceeded):
        asyncio.run(session.search("second query"))
    assert retriever.requests == [("query", 5)]


def test_malformed_backend_documents_fail_closed_without_partial_context() -> None:
    retriever = ScriptedRetriever([[FakeDocument("valid", "would otherwise be admitted"), FakeDocument("", "bad id")]])
    session = SearchSession(retriever, SearchBudget(max_searches=1))

    with pytest.raises(ValueError, match="id"):
        asyncio.run(session.search("query"))

    assert session.context_documents == []
    assert session.context_ids == []
    assert session.calls[0]["status"] == "error"
    assert session.calls[0]["documents"] == []
    assert session.calls[0]["error_type"] == "ValueError"


@pytest.mark.parametrize(
    "document",
    [
        FakeDocument(7, "content"),
        FakeDocument("id", None),
        FakeDocument("id", {"not": "text"}),
    ],
)
def test_document_ids_and_content_must_be_strings(document: FakeDocument) -> None:
    session = SearchSession(ScriptedRetriever([[document]]), SearchBudget(max_searches=1))

    with pytest.raises((TypeError, ValueError)):
        asyncio.run(session.search("query"))

    assert session.calls[0]["status"] == "error"
    assert session.context_documents == []


def test_concurrent_searches_never_issue_more_backend_calls_than_the_budget() -> None:
    class GatedRetriever:
        def __init__(self) -> None:
            self.release = asyncio.Event()
            self.requests: list[str] = []

        async def retrieve(self, query: str, *, top_k: int = 5, **kwargs: Any) -> list[FakeDocument]:
            self.requests.append(query)
            await self.release.wait()
            return [FakeDocument(query, query)]

    async def scenario() -> tuple[GatedRetriever, SearchSession, list[Any]]:
        retriever = GatedRetriever()
        session = SearchSession(retriever, SearchBudget(max_searches=2))
        tasks = [asyncio.create_task(session.search(f"q{index}")) for index in range(3)]
        await asyncio.sleep(0)
        retriever.release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return retriever, session, results

    retriever, session, results = asyncio.run(scenario())

    assert retriever.requests == ["q0", "q1"]
    assert [call["query"] for call in session.calls] == ["q0", "q1"]
    assert sum(isinstance(result, SearchBudgetExceeded) for result in results) == 1
    assert sum(isinstance(result, dict) for result in results) == 2


def test_score_searches_uses_occurrence_depth_but_unique_gold_hits() -> None:
    calls = [
        {
            "index": 1,
            "query": "first",
            "status": "ok",
            "documents": [
                {"chunk_id": "a", "rank": 1, "score": 0.9},
                {"chunk_id": "a", "rank": 2, "score": 0.8},
                {"chunk_id": "x", "rank": 3, "score": 0.7},
            ],
        },
        {
            "index": 2,
            "query": "failed",
            "status": "error",
            "documents": [{"chunk_id": "b", "rank": 1, "score": 1.0}],
            "error_type": "TimeoutError",
        },
        {
            "index": 3,
            "query": "third",
            "status": "ok",
            "documents": [{"chunk_id": "b", "rank": 1, "score": 0.6}],
        },
    ]

    result = score_searches(calls, ["a"], ["a", "b"])

    assert result == {
        "per_hop": [
            {"all_gold": False, "recall": 0.5, "hits": 1, "gold_count": 2, "retrieved_count": 3},
            {"all_gold": False, "recall": 0.0, "hits": 0, "gold_count": 2, "retrieved_count": 0},
            {"all_gold": False, "recall": 0.5, "hits": 1, "gold_count": 2, "retrieved_count": 1},
        ],
        "cumulative": [
            {"all_gold": False, "recall": 0.5, "hits": 1, "gold_count": 2, "retrieved_count": 3},
            {"all_gold": False, "recall": 0.5, "hits": 1, "gold_count": 2, "retrieved_count": 3},
            {"all_gold": True, "recall": 1.0, "hits": 2, "gold_count": 2, "retrieved_count": 4},
        ],
        "final_context": {
            "all_gold": False,
            "recall": 0.5,
            "hits": 1,
            "gold_count": 2,
            "retrieved_count": 1,
        },
        "first_complete_hop": 3,
        "search_calls": 3,
    }


@pytest.mark.parametrize("gold_ids", [[], ["a", "a"], [""], ["   "], [1], "a"])
def test_score_searches_rejects_malformed_or_non_unique_gold_ids(gold_ids: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="gold_ids"):
        score_searches([], [], gold_ids)
