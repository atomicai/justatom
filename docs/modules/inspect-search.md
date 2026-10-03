# Inspect search harness

An optional experiment adapter compares Inspect AI's ready-made **ReAct** agent
with the existing `AgenticRAGRuntime` in context-acquisition mode. It does not
replace the production runtime, change `/searching/agentic`, index documents,
or upload datasets. Inspect is pinned because its agent API evolves independently.
The supported pin is `inspect-ai==0.3.258` with OpenAI SDK `>=2.45,<3`, compatible
with the existing server extra. Newer Inspect releases requiring OpenAI SDK 3
must be evaluated separately before upgrading.

## Install and run

```bash
pip install '.[inspect]'
python -m justatom.agentic.inspect_cli --help
```

Supply a local JSONL or Parquet query file. Each row requires `query_id`, `query`,
and a nonempty unique `chunk_ids` list. Optional `domain`, `language`, `universe`,
and `depth` are retained for reporting. `answer` and other fields are ignored.
The whole file is validated before applying `--limit` or making model calls.
IDs must already match those returned by the search service: they are never
regenerated or translated to Weaviate object UUIDs.

Use an **already running** justatom search service configured with
`retrieval.mode: keyword`, the intended corpus collection, and a recorded index
revision. The harness only POSTs `text` and `top_k` to `/searching`; it does not
set per-sample domain/universe filters or connect directly to the database.
The service does not expose index identity: `--corpus-revision` and
`--retrieval-mode` record the operator's declaration, not an automatic check.
Verify the service configuration before comparing runs.

For example, with a local OpenAI-compatible llama-server supporting both tool
calls and structured JSON responses:

```bash
python -m justatom.agentic.inspect_cli \
  --dataset /path/to/queries.parquet \
  --search-url http://127.0.0.1:8000/searching \
  --base-url http://127.0.0.1:18081/v1 \
  --model YOUR_SERVED_MODEL_ID \
  --corpus-revision anchorbank-1.0.0-your-index-revision \
  --method both --limit 20 --concurrency 1 \
  --top-k 5 --max-searches 4 --max-context-documents 20 \
  --max-model-calls 6 --token-limit 8192 --time-limit 60 \
  --output .data/inspect-search/pilot-001
```

The output directory must be new. Start small to check the server's capabilities:
OpenAI-compatible transport alone does not guarantee reliable tool calling or
support for the native planner's JSON schema. For a hosted endpoint, put its key
in the environment variable named by `--api-key-env` (default `OPENAI_API_KEY`);
never put a key into an endpoint URL. Running against a hosted model incurs its
normal provider charges. This harness does not set a dollar spending limit.

## Comparison protocol

Both methods search the original question first. This initial retrieval counts
as hop 1 and consumes one search slot. Subsequent queries can only come from the
agent; ground-truth IDs and answers are not model input. Inspect uses
`attempts=1`, no submission tool, and no correctness feedback or judge-model
retry. The objective is to gather context, not generate a final answer.

The independently controlled budgets are:

| Parameter | Meaning |
| --- | --- |
| `--top-k K` | At most K returned passages per search, including hop 1 |
| `--max-searches H` | At most H searches; at most H × K returned slots |
| `--max-context-documents C` | At most C unique passages retained, in first-seen order |
| `--max-document-chars` | Per-passage text cap |
| `--max-context-chars` | Total retained passage-text cap, excluding prompts/IDs |
| `--max-model-calls` | Maximum planner/model generations per question |
| `--token-limit` | Observed token budget, not a prepaid or exact pre-call cap |
| `--time-limit` | Per-sample time limit, in seconds |

Search requests and context admission are bounded outside the model. The ReAct
continuation hook prevents a next model call when the generation budget is
exhausted; Inspect's post-generation turn limit is only an additional safeguard.
Both models use temperature 0 and a 512-token per-call output cap. Missing
provider token usage remains unknown; it must not be reported as free inference.
Automatic retries are disabled both in Inspect's generation configuration and
in its underlying OpenAI SDK client; a transport error does not buy extra attempts.
An in-flight call can cross a token threshold. Time limits are cooperative and
cannot preempt a backend that blocks the Python event loop.

For single-shot BM25 use `--max-searches 1`: neither method calls the model.
For curves over K = 3, 5, 7, 10, run separate output directories, keeping corpus,
query IDs, H, context limits, and model configuration fixed. Also report H × K,
actual searches, model usage, and latency: equal K is not equal retrieval work.

This compares **two complete agent protocols**, not identical prompts. Native
uses its existing structured `search | stop` planner, repeated-query and
no-progress guards. ReAct uses tool calls and accumulated chat history. Prompt
format, stop behavior, and input token consumption consequently differ. Native
query normalization also follows the production runtime. No production policy
is silently changed to make the results match.

## Metrics and artifacts

`results.jsonl` contains one record per requested query and method, including
failures and missing samples. `summary.json` aggregates over the full requested
denominator. Inspect's `.eval` logs retain the external message/tool history;
native runs additionally retain their real schema-v2 trace. The small common
benchmark record is not a replacement for that production trace schema.

Coverage is measured separately for each hop, the union of retrieved passages,
and the final retained context:

- `recall`: fraction of labeled passage IDs recovered.
- `all_gold`: whether **every** labeled passage ID was recovered.
- `first_complete_hop`: first hop where the cumulative union contains all IDs.

Union coverage can exceed final-context coverage when context limits discard
passages. Matching a chunk ID is a retrieval proxy: truncating its text can remove
the required fact, and alternative valid passages may be unlabeled. These metrics
do **not** establish factual answer correctness or semantic fact completeness.
Dataset development/tuning results are not an untouched test-set claim.

Native model usage comes from native telemetry, not Inspect's unused model.
The export does not estimate dollar cost without verified provider pricing.
Full traces include question and passage text: treat them as sensitive artifacts
and review them before publishing or constructing training examples. Successful
retrieval alone does not automatically make a trajectory a validated SFT target.

To inspect runs interactively:

```bash
inspect view --log-dir .data/inspect-search/pilot-001/logs
```

## Python integration and testing

`justatom.agentic.inspect_harness.build_task(cases, retriever, ...)` accepts the
same `AgentRetriever` interface as the native runtime. The caller owns the
retriever's lifecycle. Native mode additionally accepts `planner_factory`, which
must create a separate planner for each sample. Gold stays inside the offline
scorer, not inside the search session.

The optional CI job exercises the **real Inspect evaluator and ReAct loop** using
Inspect's `mockllm` provider and deterministic retrieval fixtures. It tests budget
enforcement, label isolation, error accounting, and compatibility without a paid
model, Docker, or external corpus. These are integration/contract tests, not a
claim that ReAct improves retrieval quality on AnchorBank.
