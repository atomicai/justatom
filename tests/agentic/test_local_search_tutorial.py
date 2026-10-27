"""Execute tutorial cells with real retrieval and Inspect; replace only external services."""

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebook/agentic-tutorials/local-react-vs-native.ipynb"


def notebook_cells():
    assert NOTEBOOK.is_file(), "The runnable local search tutorial is missing"
    return json.loads(NOTEBOOK.read_text())["cells"]


def tutorial_helpers():
    namespace = {}
    for cell in notebook_cells():
        if cell.get("id") in {"imports", "endpoint", "comparison-helper"}:
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
    return namespace


async def execute_tutorial_cell(cell_id, namespace):
    cells = {cell.get("id"): cell for cell in notebook_cells()}
    assert cell_id in cells, f"The executable tutorial cell {cell_id!r} is missing"
    code = compile("".join(cells[cell_id]["source"]), str(NOTEBOOK), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    result = eval(code, namespace)
    if result is not None:
        await result


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:18081/v1",
        "http://localhost:2211",
        "http://[::1]:18081/v1",
        "https://api.openai.com/v1",
        "https://openrouter.ai/api/v1",
    ],
)
def test_tutorial_accepts_local_and_remote_http_services(url):
    tutorial_helpers()["validate_endpoint"](url)


@pytest.mark.parametrize(
    "url",
    [
        "http://secret@127.0.0.1:18081/v1",
        "https://user:secret@api.example.com/v1",
        "http://127.0.0.1:18081/v1?key=secret",
        "http://127.0.0.1:18081/v1#secret",
        "file:///tmp/model",
        "https:///v1",
        "localhost:18081/v1",
    ],
)
def test_tutorial_rejects_unsafe_or_malformed_endpoints(url):
    with pytest.raises(ValueError):
        tutorial_helpers()["validate_endpoint"](url)


def test_tutorial_keeps_failure_and_unknown_usage_visible_despite_complete_retrieval():
    row = {
        "method": "native",
        "status": "error",
        "termination_reason": "provider_max_tokens",
        "model_calls": 1,
        "duration_ms": 2500,
        "calls": [{}, {}],
        "evaluation": {"final_context": {"hits": 3, "gold_count": 3, "recall": 1.0, "all_gold": True}, "first_complete_hop": 2},
        "usage": {"input_tokens": None, "output_tokens": None, "total_tokens": None, "cost_usd": None},
    }
    result = tutorial_helpers()["comparison_row"](row)
    assert result["status"] == "error"
    assert result["all_gold"] is True
    assert result["seconds"] == 2.5
    assert result["searches"] == 2
    assert result["tokens"] is None
    assert result["provider_cost_usd"] is None


def test_all_tutorial_code_cells_compile_with_notebook_await():
    for index, cell in enumerate(notebook_cells()):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{NOTEBOOK}:cell{index}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


class IndexingDatabase:
    """Stateful Weaviate boundary; Indexer, retriever and runtime remain real."""

    def __init__(self, documents=(), *, failure=None, error_type=RuntimeError):
        self.documents = {document.id: document for document in documents}
        self.failure = failure
        self.error_type = error_type
        self.connections = []
        self.events = []
        self.batches = []
        self.queries = []
        self.initial_ids = ("a",)

    async def connect(self, **config):
        if self.failure == "connect" or (self.failure == "reconnect" and self.connections):
            raise self.error_type("connect failed")
        database = self

        class Store:
            closed = False

            async def count_documents(self):
                assert not self.closed
                if database.failure == "count":
                    raise database.error_type("count failed")
                return len(database.documents)

            async def delete_collection(self):
                assert not self.closed
                database.events.append("delete")
                if database.failure == "delete":
                    raise database.error_type("delete failed")
                database.documents.clear()

            async def write_documents(self, documents):
                assert not self.closed
                if database.failure == "write":
                    raise database.error_type("write failed")
                database.batches.append(documents)
                database.documents.update((document.id, document) for document in documents)
                return len(documents)

            async def search_keywords(self, queries, *, top_k, filters=None):
                assert not self.closed
                if database.failure == "search":
                    raise database.error_type("search failed")
                database.queries.extend((query, top_k) for query in queries)
                return [
                    [
                        database.documents[doc_id]
                        for doc_id in (database.initial_ids if query == "original" else ("b",))
                        if doc_id in database.documents
                    ][:top_k]
                    for query in queries
                ]

            async def close(self):
                self.closed = True
                database.events.append("close")

        self.events.append("connect")
        store = Store()
        self.connections.append(store)
        return store


def tutorial_run_fixture(tmp_path, *, rows=None, **database_options):
    import polars as pl

    from justatom.agentic.benchmark_data import BenchmarkCase

    pytest.importorskip("inspect_ai")
    namespace = tutorial_helpers()
    asyncio.run(execute_tutorial_cell("components", namespace))
    corpus = tmp_path / "corpus.parquet"
    pl.DataFrame(
        rows
        if rows is not None
        else {
            "chunk_id": ["a", "b", "c"],
            "content": ["Alpha", "Beta", "Gamma"],
        }
    ).write_parquet(corpus)
    database = IndexingDatabase(**database_options)
    namespace.update(
        corpus_path=corpus,
        model={"name": "local-test", "base_url": "http://127.0.0.1:18081/v1", "api_key": "fixture-key"},
        store_config={"collection": "Agentic_tutorial_bm25", "url": "http://localhost:2211", "grpc_port": 50051},
        output_dir=tmp_path / "runs",
        overwrite=True,
        batch_size=2,
        task_options={"max_model_calls": 3, "max_output_tokens": 128, "token_limit": 24000, "time_limit": 180},
        case=BenchmarkCase("fixture-case", "original", ("a", "b", "gold-only-secret"), {}),
        budget=namespace["SearchBudget"](top_k=1, max_searches=3),
        WeaviateDocumentStore=SimpleNamespace(connect=database.connect),
    )
    return namespace, database


def install_model_boundaries(namespace, *, native_failure=None):
    """Keep both real harnesses, supplying deterministic external LLM replies."""
    import httpx
    from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput, get_model
    from inspect_ai.tool import ToolCall

    from justatom.agentic.openai_compatible import OpenAICompatibleChatBackend

    requests = {"react": [], "native": []}

    def react_output(messages, tools, tool_choice, config):
        requests["react"].append(json.dumps([message.model_dump() for message in messages]))
        if len(requests["react"]) % 2:
            return ModelOutput(
                model="fixture",
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessageAssistant(
                            content="",
                            tool_calls=[
                                ToolCall(
                                    id="s1",
                                    function="search",
                                    arguments={"query": "followup"},
                                )
                            ],
                        ),
                        stop_reason="tool_calls",
                    )
                ],
            )
        return ModelOutput.from_content("fixture", "DONE")

    def native_response(request):
        assert str(request.url) == namespace["model"]["base_url"] + "/chat/completions"
        assert request.headers["authorization"] == "Bearer fixture-key"
        body = json.loads(request.content)
        requests["native"].append(json.dumps(body))
        assert body["model"] == "local-test"
        assert body["temperature"] == 0
        assert body["max_tokens"] == 128
        assert body["response_format"]["json_schema"]["schema"]["properties"]["action"]["enum"] == ["search", "stop"]
        if native_failure is not None:
            raise native_failure("provider unavailable")
        search = len(requests["native"]) % 2 == 1
        decision = {
            "action": "search" if search else "stop",
            "query": "followup" if search else None,
            "answer": None,
            "reason": None,
            "cited_document_ids": [],
        }
        return httpx.Response(
            200,
            json={
                "model": "local-test",
                "choices": [{"message": {"role": "assistant", "content": json.dumps(decision)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )

    def fixture_model(name, **kwargs):
        if name != "mockllm/unused":
            assert kwargs["base_url"] == namespace["model"]["base_url"]
            assert kwargs["api_key"] == "fixture-key"
        return get_model("mockllm/tutorial", custom_outputs=react_output, memoize=False)

    namespace["get_model"] = fixture_model
    namespace["OpenAICompatibleChatBackend"] = lambda *args, **kwargs: OpenAICompatibleChatBackend(
        *args,
        **kwargs,
        transport=httpx.MockTransport(native_response),
    )
    return requests


async def run_tutorial(namespace):
    await execute_tutorial_cell("corpus", namespace)
    await execute_tutorial_cell("run", namespace)


def test_corpus_cell_prepares_original_ids_before_database_access(tmp_path):
    namespace, database = tutorial_run_fixture(tmp_path)
    asyncio.run(execute_tutorial_cell("corpus", namespace))
    assert namespace["documents"] == [
        {"id": "a", "content": "Alpha"},
        {"id": "b", "content": "Beta"},
        {"id": "c", "content": "Gamma"},
    ]
    assert database.events == []


@pytest.mark.parametrize(
    "rows",
    [
        {"chunk_id": [], "content": []},
        {"chunk_id": ["a"]},
        {"content": ["Text"]},
        {"chunk_id": ["a", "a"], "content": ["Text", "Other"]},
        {"chunk_id": [" "], "content": ["Text"]},
        {"chunk_id": [None], "content": ["Text"]},
        {"chunk_id": [1], "content": ["Text"]},
        {"chunk_id": ["a"], "content": [" "]},
        {"chunk_id": ["a"], "content": [None]},
        {"chunk_id": ["a"], "content": [1]},
    ],
)
def test_invalid_corpus_is_rejected_before_any_database_access(rows, tmp_path):
    namespace, database = tutorial_run_fixture(tmp_path, rows=rows)
    with pytest.raises(ValueError):
        asyncio.run(run_tutorial(namespace))
    assert database.events == []


@pytest.mark.parametrize("existing", [False, True])
def test_full_tutorial_overwrites_original_ids_without_embeddings(existing, tmp_path):
    from justatom.etc.schema import Document

    namespace, database = tutorial_run_fixture(
        tmp_path,
        documents=[Document(id="stale", content="Old")] if existing else [],
    )
    install_model_boundaries(namespace)
    asyncio.run(run_tutorial(namespace))
    assert [(doc.id, doc.content, doc.embedding) for doc in database.documents.values()] == [
        ("a", "Alpha", None),
        ("b", "Beta", None),
        ("c", "Gamma", None),
    ]
    assert [len(batch) for batch in database.batches] == [2, 1]
    assert all(store.closed for store in database.connections)
    assert namespace["baseline"]["context_ids"] == ["a"]


@pytest.mark.parametrize("existing", [False, True])
def test_without_overwrite_reuses_existing_corpus_or_indexes_empty_store(existing, tmp_path):
    from justatom.etc.schema import Document

    namespace, database = tutorial_run_fixture(
        tmp_path,
        documents=[Document(id="a", content="Keep original")] if existing else [],
    )
    namespace["overwrite"] = False
    install_model_boundaries(namespace)
    asyncio.run(run_tutorial(namespace))
    assert "delete" not in database.events
    if existing:
        assert [(doc.id, doc.content) for doc in database.documents.values()] == [("a", "Keep original")]
        assert not database.batches
    else:
        assert list(database.documents) == ["a", "b", "c"]
        assert [len(batch) for batch in database.batches] == [2, 1]
    assert all(store.closed for store in database.connections)


@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
@pytest.mark.parametrize("failure", ["count", "delete", "reconnect", "write", "search"])
def test_tutorial_closes_owned_connections_after_failure_or_cancellation(failure, error_type, tmp_path):
    from justatom.etc.schema import Document

    namespace, database = tutorial_run_fixture(
        tmp_path,
        documents=[Document(id="old", content="Old")],
        failure=failure,
        error_type=error_type,
    )
    namespace["overwrite"] = failure != "count"
    with pytest.raises(BaseException) as caught:
        asyncio.run(run_tutorial(namespace))
    if error_type is asyncio.CancelledError:
        assert isinstance(caught.value, asyncio.CancelledError)
    else:
        assert isinstance(caught.value, Exception)
    assert database.connections
    assert all(store.closed for store in database.connections)


@pytest.mark.parametrize("setting", ["model", "store_config", "batch_size"])
def test_run_validates_endpoints_and_batch_size_before_database_access(setting, tmp_path):
    namespace, database = tutorial_run_fixture(tmp_path)
    if setting == "batch_size":
        namespace["batch_size"] = 0
    else:
        key = "base_url" if setting == "model" else "url"
        namespace[setting][key] = "https://user:secret@api.example.com/v1"
    with pytest.raises(ValueError):
        asyncio.run(run_tutorial(namespace))
    assert database.events == []


def test_connection_failure_explains_how_to_start_weaviate(tmp_path):
    from justatom.etc.errors import DocumentStoreError

    namespace, database = tutorial_run_fixture(
        tmp_path,
        failure="connect",
        error_type=DocumentStoreError,
    )
    with pytest.raises(RuntimeError, match="docker compose up -d weaviate") as caught:
        asyncio.run(run_tutorial(namespace))
    assert "repo" in str(caught.value).lower()
    assert database.connections == []


@pytest.mark.parametrize("base_url", ["http://127.0.0.1:18081/v1", "https://provider.example.com/v1"])
def test_full_tutorial_keeps_gold_off_model_inputs_and_exports_both_protocols(base_url, tmp_path):
    namespace, database = tutorial_run_fixture(tmp_path)
    namespace["model"]["base_url"] = base_url
    requests = install_model_boundaries(namespace)
    asyncio.run(run_tutorial(namespace))
    run_dir = namespace["run_dir"]
    assert isinstance(run_dir, Path)
    assert run_dir.parent == namespace["output_dir"]
    baseline = namespace["baseline"]
    assert baseline["context_ids"] == ["a"]
    assert baseline["model_calls"] == 0
    assert baseline["evaluation"]["final_context"]["recall"] == pytest.approx(1 / 3)
    assert json.loads((run_dir / "bm25.json").read_text())["context_ids"] == ["a"]
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert {row["method"] for row in rows} == {"react", "native"}
    for method in ("react", "native"):
        result = namespace[method]
        assert result["status"] == "success"
        assert result["output_dir"] == run_dir
        assert result["context_ids"] == ["a", "b"]
        assert result["model_calls"] == 2
        assert result["evaluation"]["final_context"]["recall"] == pytest.approx(2 / 3)
        assert result["evaluation"]["final_context"]["all_gold"] is False
        assert result["config"]["budget"]["top_k"] == 1
        assert len(requests[method]) == 2
        assert all("gold-only-secret" not in request for request in requests[method])
        assert list((run_dir / "logs" / method).glob("*.eval"))
    assert namespace["native"]["native_trace"]["schema_version"] == 2
    assert database.queries == [("original", 1), ("original", 1), ("followup", 1), ("original", 1), ("followup", 1)]
    assert all(store.closed for store in database.connections)
    asyncio.run(execute_tutorial_cell("trace-helpers", namespace))
    assert namespace["agent_decisions"](namespace["react"]) == [
        {"tool": "search", "arguments": {"query": "followup"}},
    ]


def test_rerunning_tutorial_uses_fresh_sessions_and_preserves_previous_artifacts(tmp_path):
    namespace, database = tutorial_run_fixture(tmp_path)
    install_model_boundaries(namespace)
    asyncio.run(run_tutorial(namespace))
    first_run = namespace["run_dir"]
    first_results = (first_run / "results.jsonl").read_bytes()
    asyncio.run(run_tutorial(namespace))
    second_run = namespace["run_dir"]
    assert first_run != second_run
    assert (first_run / "results.jsonl").read_bytes() == first_results
    assert (second_run / "results.jsonl").is_file()
    assert len(namespace["baseline"]["calls"]) == 1
    assert len(namespace["react"]["calls"]) == len(namespace["native"]["calls"]) == 2
    assert all(store.closed for store in database.connections)


def test_native_provider_failure_remains_visible_in_export_and_closes_store(tmp_path):
    namespace, database = tutorial_run_fixture(tmp_path)
    install_model_boundaries(namespace, native_failure=RuntimeError)
    asyncio.run(run_tutorial(namespace))
    assert namespace["react"]["status"] == "success"
    assert namespace["native"]["status"] == "error"
    rows = [json.loads(line) for line in (namespace["run_dir"] / "results.jsonl").read_text().splitlines()]
    assert [(row["method"], row["status"]) for row in rows] == [("react", "success"), ("native", "error")]
    assert all(store.closed for store in database.connections)


def test_native_cancellation_preserves_completed_react_export_and_closes_store(tmp_path):
    namespace, database = tutorial_run_fixture(tmp_path)
    install_model_boundaries(namespace, native_failure=asyncio.CancelledError)
    try:
        asyncio.run(run_tutorial(namespace))
    except asyncio.CancelledError:
        pass
    rows = [json.loads(line) for line in (namespace["run_dir"] / "results.jsonl").read_text().splitlines()]
    assert rows[0]["method"] == "react"
    assert rows[0]["status"] == "success"
    assert rows[1]["method"] == "native"
    assert rows[1]["status"] in {"error", "missing"}
    assert all(store.closed for store in database.connections)


def test_baseline_distinguishes_retrieved_ids_from_truncated_context(tmp_path):
    namespace, database = tutorial_run_fixture(tmp_path)
    namespace["budget"] = namespace["SearchBudget"](top_k=2, max_context_documents=1)
    database.initial_ids = ("a", "b")
    install_model_boundaries(namespace)
    asyncio.run(run_tutorial(namespace))
    result = namespace["baseline"]
    assert [doc["chunk_id"] for doc in result["calls"][0]["documents"]] == ["a", "b"]
    assert result["context_ids"] == ["a"]
    assert result["evaluation"]["final_context"]["recall"] == pytest.approx(1 / 3)


def reporting_helpers():
    namespace = tutorial_helpers()
    for cell in notebook_cells():
        if cell.get("id") == "trace-helpers":
            exec("".join(cell["source"]), namespace)
    return namespace


def test_native_decision_display_handles_missing_trace_after_failure():
    assert reporting_helpers()["agent_decisions"]({"method": "native", "status": "error", "native_trace": None}) == []


def execute_display_cell(cell_id, namespace):
    cell = next(cell for cell in notebook_cells() if cell.get("id") == cell_id)
    parsed = ast.parse("".join(cell["source"]))
    assert isinstance(parsed.body[-1], ast.Expr), "Display cells must end with a preview expression"
    exec(compile(ast.Module(body=parsed.body[:-1], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return eval(compile(ast.Expression(parsed.body[-1].value), str(NOTEBOOK), "eval"), namespace)


@pytest.mark.parametrize(
    "cell_id, variable, populated_rows",
    [
        ("comparison", "response_result", 3),
        ("react-searches", "react_steps", 15),
        ("native-searches", "native_steps", 15),
        ("react-decisions", "react_decisions", 15),
        ("native-decisions", "native_decisions", 15),
        ("evidence", "context_result", 15),
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_display_cells_retain_full_dataframes_and_show_at_most_ten_rows(cell_id, variable, populated_rows, empty, tmp_path):
    import polars as pl

    namespace = reporting_helpers()
    count = 0 if empty else 15
    result = {
        "method": "native",
        "status": "error" if empty else "success",
        "calls": [
            {"index": index, "query": f"Search {index}", "documents": [{"chunk_id": f"id-{index}"}]} for index in range(count)
        ],
        "context_documents": [{"chunk_id": f"id-{index}", "content": f"Evidence {index}"} for index in range(count)],
        "evaluation": {
            "cumulative": [{"hits": index + 1, "gold_count": 15} for index in range(count)],
            "final_context": {"hits": count, "gold_count": 15, "recall": count / 15, "all_gold": count == 15},
            "first_complete_hop": 15 if count else None,
        },
        "native_trace": {"steps": [{"decision": {"action": "search", "query_text": f"Search {index}"}} for index in range(count)]},
        "output_dir": tmp_path,
    }
    if count:
        (tmp_path / "logs" / "react").mkdir(parents=True)
        (tmp_path / "logs" / "react" / "fixture.eval").touch()
    namespace["read_eval_log"] = lambda path: SimpleNamespace(
        samples=[
            SimpleNamespace(
                messages=[
                    SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(function="search", arguments={"query": f"Search {index}"}) for index in range(count)
                        ],
                    )
                ]
            )
        ]
    )
    namespace.update(baseline={**result, "method": "bm25"}, react={**result, "method": "react"}, native=result)
    preview = execute_display_cell(cell_id, namespace)
    dataframe = namespace[variable]
    assert isinstance(dataframe, pl.DataFrame)
    assert dataframe.height == (3 if cell_id == "comparison" else (0 if empty else populated_rows))
    assert isinstance(preview, pl.DataFrame)
    assert preview.equals(dataframe.head(10))
    if empty and cell_id == "comparison":
        assert dataframe["status"].to_list() == ["error", "error", "error"]
