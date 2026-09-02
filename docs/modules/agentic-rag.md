# Agentic RAG

Agentic RAG adds a planner-controlled retrieval loop on top of the retrieval
runtime. The configured `objective` determines what the loop produces:

- `answer` runs a `search | answer` planner and returns a generated answer with
  citations.
- `context` runs a `search | stop` planner and returns only the accumulated
  evidence context. It never asks the planner to generate a final answer.

Both objectives are deliberately bounded: every admitted run has explicit
work, observed-token, content-size, and wall-clock budgets, and every admitted
exit is recorded in a structured trace. Pre-admission capacity rejection is a
separate process-level event because no run ID or trajectory is created.

The trace contract is currently **schema v2**. Consumers should check
`schema_version` and reject versions they do not understand instead of
assuming forward compatibility.
Schema v1 was the pre-merge answer-only prototype; v2 makes `objective` and
the retrieval/context budget fields mandatory and deliberately rejects v1
artifacts rather than guessing missing benchmark settings.

## Bounded Loop

A run proceeds as follows:

1. Retrieve evidence for the original question. The first search is always the
   original question, so a planner cannot skip the baseline retrieval.
2. Give the planner the question, accumulated search observations, current
   context, and its remaining step and retrieval budgets.
3. Accept one objective-specific structured action. The `answer` objective
   allows `search | answer`; the `context` objective allows `search | stop`.
4. For `search`, retrieve another bounded `top_k` result set, merge it into the
   context, and repeat. `answer` finishes an answer-producing run. `stop`
   finishes a context-only run without generating an answer or citations.

The two decision schemas are disjoint. In context mode a `stop` decision has
`query: null`, `answer: null`, and `cited_document_ids: []`; an `answer` action
is invalid. The completed result and HTTP response likewise contain
`answer: null` and an empty citation list, while `evidence` contains the final
context, including its bounded passage text. A planner-selected stop uses the
`agent_stop` termination reason; a hard budget or progress guard retains its
more specific reason.

`max_steps`, `max_retrieval_calls`, and `max_llm_calls` are local hard limits
rather than tuning hints. `total_timeout_seconds` is a response deadline for
async components that yield control to the event loop. Already-started backend
work is not coroutine-cancelled at that deadline: it is detached and tracked,
and retains its backend-capacity permit until it really exits. This matters for
`to_thread`, executor, native, and GPU work, whose worker can outlive a
cancelled asyncio task. No asyncio runtime can preempt third-party code that
synchronously blocks the event-loop thread. The
initial search counts against the retrieval-call and step budgets. A local
`top_k` slice is applied even if a retriever over-returns; the trace records the
backend count and number discarded.

`max_tokens` is a **post-call observed-usage budget**. It prevents the next
planner/search action after reported `total_tokens` reaches the limit; when a
provider omits the total, input plus output is used if both are present. It
cannot predict input size or undo the call that crossed the threshold, and it
cannot be enforced for calls with missing usage. `planner.max_tokens`
separately sets the provider's per-call output limit. Token coverage makes
missing usage visible. Normalized repeated queries and no-progress states also
stop the loop. The trace records a specific
`termination_reason`, such as `answered`, `agent_stop`, `max_steps`,
`max_retrieval_calls`, `max_llm_calls`, `max_tokens`, `max_duration`,
`no_progress`, or `repeated_query`.

`max_concurrency` bounds active runs in one agentic runtime and
`max_queued_runs` bounds admitted waiters. Separate component and trace permits
prevent cancellation-draining work from accumulating behind apparently free
run slots. Time spent waiting for a run slot is recorded separately from
execution time.

Search observations carry query/rank/document references. Passage text is sent
once, through the deduplicated context, so `max_context_chars` remains a real
upper bound instead of being bypassed by duplicated observation text.

The trace records `objective` explicitly. Its limits also retain `top_k`,
`max_context_documents`, and `max_context_chars`, so a context benchmark can
reconstruct both the permitted retrieval work and the context delivered to a
downstream consumer.

Every accepted planner decision is attached to its step before runtime routing.
This keeps a planned search observable even when a retrieval, step, token,
repeat-query, or no-progress guard blocks its execution.

## Library Components

The public components are intentionally small and composable:

```python
from justatom.agentic.openai_compatible import OpenAICompatibleChatBackend
from justatom.agentic.runtime import (
    AgenticRAGRuntime,
    AgenticRuntimeConfig,
    build_agentic_runtime,
)
from justatom.agentic.telemetry import (
    JsonlTraceSink,
    derive_run_metrics,
    iter_jsonl_traces,
)
```

`AgenticRAGRuntime` accepts any retriever implementing `AgentRetriever`, so an
existing `RetrievalRuntime` can be reused. `OpenAICompatibleChatBackend` sends
structured planner requests to `/chat/completions`; it does not own retrieval
or indexing. `JsonlTraceSink` appends one complete, compact UTF-8 JSON object
per run.

```python
import os

from justatom.agentic import AgentObjective
from justatom.agentic.openai_compatible import OpenAICompatibleChatBackend
from justatom.agentic.runtime import AgenticRAGRuntime, AgenticRuntimeConfig
from justatom.agentic.telemetry import JsonlTraceSink

# `retrieval` is an existing RetrievalRuntime (or another AgentRetriever).
chat = OpenAICompatibleChatBackend(
    base_url="http://localhost:8000/v1",
    model="planner-model",
    api_key=os.getenv("CHAT_API_KEY"),
    timeout_seconds=30.0,
    temperature=0.0,
    max_tokens=512,
    seed=42,
    objective=AgentObjective.ANSWER,
)
agent = AgenticRAGRuntime(
    retrieval,
    chat,
    config=AgenticRuntimeConfig(
        objective=AgentObjective.ANSWER,
        max_steps=6,
        max_retrieval_calls=4,
        max_llm_calls=6,
        max_tokens=4096,
        total_timeout_seconds=45.0,
        trace_timeout_seconds=5.0,
        top_k=5,
        seed=42,
    ),
    trace_sink=JsonlTraceSink(".data/traces/agentic-rag.jsonl"),
)

try:
    result = await agent.run(
        "Which documents explain the deployment architecture?",
        request_id="request-42",
        filters={"language": "en"},
        metadata={"experiment": "bounded-loop-v1"},
    )
    print(result.answer)
    print(result.evidence)
    print(result.metrics)
finally:
    await agent.close()
```

For context acquisition, set `objective=AgentObjective.CONTEXT` on both
`AgenticRuntimeConfig` and `OpenAICompatibleChatBackend`. The runtime rejects a
mismatch rather than sending an answer-mode schema to a context-mode loop. The
config-driven builder propagates the single top-level `agentic.objective`
setting to both components automatically. In context mode, consume
`result.evidence`; `result.answer` is always `None` and the trace has no final
citations.

The result exposes `run_id`, `answer`, `evidence`, `trace`, and `metrics`. Keep
`result.trace` when an experiment needs offline evidence evaluation; the
metrics mapping is only a derived view of that trace.

Each trace stores the overall, planner, and retrieval configuration
fingerprints. Planner sampling/output settings and prompt hash are covered;
retrieval mode, implementation types, collection/index revision, hybrid alpha,
resolved store endpoint/gRPC settings, local device, and available embedder
settings are covered. Credentials are excluded and endpoint URLs are hashed.
Set `retrieval.index_revision` to an immutable corpus/index build identifier;
without one, changing documents in the same collection cannot be inferred
from runtime configuration alone.

For a measured benchmark window, treat the corpus as immutable: finish
indexing first, set a new `retrieval.index_revision`, start a fresh service
runtime, and do not call `/indexing` or `/delete` until the window ends. The
fingerprint identifies runtime configuration and the declared revision; it is
not a content hash and cannot detect an in-place corpus mutation by itself.

The config-driven `build_agentic_runtime` is the service-oriented counterpart
to direct construction with `AgenticRuntimeConfig`,
`OpenAICompatibleChatBackend`, and `JsonlTraceSink`. In both forms, close the
agentic runtime during application shutdown so its owned backend and trace sink
can flush and close. The shared retriever remains caller-owned and must be
closed separately.

## Service Configuration

The `agentic` section is separate from `retrieval`; the agent uses the already
built retrieval runtime rather than creating another store or embedder.

```yaml
agentic:
  enabled: true
  objective: answer
  max_steps: 6
  max_retrieval_calls: 4
  max_llm_calls: 6
  max_tokens: 4096
  total_timeout_seconds: 45.0
  retrieval_timeout_seconds: 15.0
  planner_timeout_seconds: 30.0
  max_concurrency: 8
  max_queued_runs: 16
  top_k: 5
  max_request_bytes: 65536
  max_query_chars: 2000
  max_answer_chars: 16000
  max_reason_chars: 4000
  max_identifier_chars: 512
  max_filter_chars: 16000
  max_metadata_chars: 8000
  max_document_chars: 12000
  max_context_chars: 48000
  max_context_documents: 20
  no_progress_limit: 2
  seed: 42

  planner:
    backend: openai-compatible
    base_url: ${CHAT_BASE_URL}
    model: ${CHAT_MODEL}
    api_key: ${CHAT_API_KEY}
    timeout_seconds: 30.0
    temperature: 0.0
    max_tokens: 512
    max_response_bytes: 1048576
    seed: null

  trace:
    path: .data/traces/agentic-rag.jsonl
    capture_text: hash
    required: true
    timeout_seconds: 5.0
    max_pending_writes: 64
```

The shared retrieval section may declare the corpus build used by both normal
and agentic search:

```yaml
retrieval:
  index_revision: habr-ir-corpus-2026-09-01
```

The character-limit settings bound generated queries, answers, planner reasons,
request/document/citation identifiers, canonical filter/metadata JSON,
individual documents, and accumulated context;
`max_context_documents` also caps context cardinality and final citation count.
At the HTTP boundary, `max_request_bytes` rejects an oversized `/searching/agentic`
body with `413` while Quart is receiving it, before JSON parsing and the
smaller field-level limits.
Keep them finite even when the upstream chat service enforces a token limit.
`planner.max_response_bytes` caps decoded response bytes while they stream, so
an untrusted or misconfigured endpoint cannot make the client buffer an
unbounded body before schema validation.
The retrieval and planner timeouts bound individual calls, while
`total_timeout_seconds` bounds queueing plus the search/planner trajectory.
At most `max_concurrency` runs execute and at most `max_queued_runs` additional
runs wait for admission. A larger burst is rejected immediately instead of
retaining an unbounded set of request payloads and tasks.
Trace persistence happens after that trajectory and has its own
`trace.timeout_seconds`, so a slow external sink cannot hold the response
past its confirmation deadline as long as the sink does not synchronously
block the event loop. A timed-out asynchronous sink write is cancelled,
tracked, and keeps a bounded trace permit until it exits. The built-in JSONL
sink performs filesystem I/O off-loop, accepts at most
`trace.max_pending_writes`, and fails fast when that bounded backlog is full.
Runtime shutdown drains accepted writes before closing the sink, so shutdown
may wait for slow storage even though request handling does not.

`objective` is a service-level choice, not a per-request override. Use
`answer` for answer generation or `context` for adaptive context acquisition.
Changing it changes the planner schema and is included in the trace and
configuration fingerprints, so benchmark variants cannot silently mix the two
contracts.

The top-level `seed` labels the experiment. In config-driven construction it is
also sent to the OpenAI-compatible planner unless `planner.seed` explicitly
overrides it with a non-null value. Provider support and determinism still
depend on that provider.
For direct library construction, pass the same seed to both
`AgenticRuntimeConfig` and `OpenAICompatibleChatBackend` when required.

The built-in planner calls an OpenAI-compatible `/chat/completions` endpoint
with strict `response_format.type: json_schema`. Compatibility therefore means
the endpoint accepts that request field and returns the decision as a JSON
string in `choices[0].message.content` with exactly `action`, `query`, `answer`,
`reason`, and `cited_document_ids`. The wire schema intentionally avoids regex
constraints so it remains compatible with providers such as `llama-server`
that implement a subset of JSON Schema. The schema permits `search | answer`
for the `answer` objective and `search | stop` for `context`; the context
schema requires null `query` and `answer` plus empty citations on `stop`. The
client still rejects empty or whitespace-only decision values after decoding.
The backend performs one provider attempt per planner call and does not retry
automatically; custom `ChatBackend` implementations may report multiple
attempts in the same trace.

`trace.required: true` is appropriate when losing a run record invalidates an
experiment: it requires a non-null `trace.path`, and a sink failure or timeout
is surfaced instead of being treated as a fully recorded success. With
`required: false`, `path: null` explicitly selects a discard sink; failures of
another best-effort sink do not turn an otherwise completed trajectory into an
error. Best-effort failures emit a sanitized warning for service monitoring.
Direct construction likewise rejects a sink explicitly marked
`required: false` when `AgenticRuntimeConfig.trace_required` is true.
The sink serializes the already-redacted `RunTrace` and does not apply a second
capture policy.

## HTTP API

The service route is:

```text
POST /searching/agentic
Content-Type: application/json
```

Use this endpoint for the configured agent objective. In `answer` mode it
returns a generated answer; in `context` mode it returns adaptively collected
evidence without a final generator call. Continue to use `POST /searching` for
a single, static retrieval request. Runtime composition, objective, planner
credentials, budgets, and trace paths belong in service configuration and are
not accepted as per-request overrides.
When `agentic.enabled` is false, the route returns `503` instead of starting a
run. A required trace that cannot be accepted or confirmed also returns a
sanitized `503`; the trajectory result is not represented as a successfully
recorded experiment in that case. A full run queue returns `429` with a
`Retry-After` header; rejected requests do not create a run trace because no
run was admitted. They increment the process-local `rejected_run_count`
available from `await runtime.admission_metrics()` and emit a sanitized service
warning, so admission pressure can still be exported by monitoring.

```console
curl -sS http://localhost:5555/searching/agentic \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Which documents explain the deployment architecture?",
    "request_id": "request-42",
    "filter_by": {"language": "en"},
    "metadata": {"experiment": "bounded-loop-v1"}
  }'
```

The request fields are `text`, optional `request_id`, optional `filter_by`,
and optional `metadata`. A successful response has the following shape:

```json
{
  "run_id": "...",
  "objective": "answer",
  "answer": "...",
  "status": "completed",
  "termination_reason": "answered",
  "cited_document_ids": ["document-17"],
  "evidence": [
    {"id": "document-17", "rank": 1, "score": 0.91}
  ],
  "metrics": {}
}
```

With `agentic.objective: context`, the same request shape produces a
context-only response:

```json
{
  "run_id": "...",
  "objective": "context",
  "answer": null,
  "status": "completed",
  "termination_reason": "agent_stop",
  "cited_document_ids": [],
  "evidence": [
    {"id": "document-17", "rank": 1, "score": 0.91, "content": "..."},
    {"id": "document-29", "rank": 2, "score": 0.87, "content": "..."}
  ],
  "metrics": {}
}
```

`status: "completed"` means the run completed operationally; inspect
`objective` and `termination_reason` to distinguish an answer, a
planner-selected context stop, and a bounded stop. `evidence` identifies the
deduplicated final context in both modes. To preserve the answer-mode API's
minimal disclosure, passage `content` is included only for the `context`
objective; it is already bounded by `max_document_chars` and
`max_context_chars`. The metrics object is derived from the same raw `RunTrace`
passed to the configured sink.
Timed-out and failed runs return the same diagnostic shape with HTTP `504` and
`502`, respectively. An unexpected internal runtime failure is distinguished
from an upstream planner/retrieval failure and returns `500`.

## Trace Privacy

Text capture is an explicit deployment choice. Clear-text questions, generated
queries, retrieved passages, and answers may contain personal or confidential
information.

| `capture_text` | Trace content |
| --- | --- |
| `none` | Raw text is omitted and the corresponding text hashes are `null`. |
| `hash` | Raw text is omitted; SHA-256 values are retained for equality and diversity analysis. This is the default. |
| `full` | Captured text and its hashes are retained; document text remains subject to configured length limits. |

The same policy applies to request filters: `none` stores neither filters nor a
hash, `hash` stores only a hash of canonical JSON, and `full` stores both. This
allows runs to be grouped by identical filter policy without exposing tenant or
attribute values in the default trace.

The selected policy is stored in every trace. Planner decisions use the same
policy for planned queries, answers, and reasons. `DocumentTrace.content_chars`
and its hash describe the full retriever value, while clear-text `content` in
`full` mode is clipped to `max_document_chars`.

Prefer `hash` or `none` for normal operations; use `full` only for controlled
evaluation with an access policy, retention period, and secured trace
directory.

Hashes are pseudonymous identifiers, not anonymization. Low-entropy questions
and known corpus passages may be recoverable by dictionary comparison. Also
avoid placing secrets in `request_id`, `experiment_id`, `variant`, document
IDs, or free-form metadata because those identifiers remain useful even when
text is redacted.

The capture policy controls persisted trace fields only. The planner still
receives the raw question, every raw observation query, remaining-budget
counts, and each bounded context document's ID, content, score, rank, and
retrieval index. The retriever receives raw search queries and request filters.
The `context` HTTP objective also returns bounded passage content to its caller,
independently of the trace capture policy, because that content is the endpoint's
product. Protect it with the same authorization and transport controls as
`POST /searching`.
Treat those endpoints as data processors and require trusted TLS transport,
appropriate retention, and access controls; `capture_text: none` does not
redact outbound provider payloads.

`JsonlTraceSink` writes exactly the trace it receives. It appends one line and
flushes it for each `RunTrace`; it does not re-redact fields. Treat the
resulting JSONL as sensitive data under the selected capture policy. On POSIX,
new trace directories/files are restricted to `0700`/`0600`; existing trace
files are tightened to `0600`. The API image pre-creates
`/app/.data/traces` for its non-root user. When another path is used in a
container, mount a writable volume owned by UID/GID `10001:10001`.

## Metrics Semantics

The raw schema-v2 trace is the source of truth. `derive_run_metrics(trace)` is
a reproducible operational summary, not a replacement for the trace and not a
quality score.

| Field | Meaning |
| --- | --- |
| experiment/config identity fields | Schema version, experiment ID, variant, seed, and overall/planner/retrieval fingerprints copied from the trace. |
| `objective` | The run contract: `answer` or `context`. |
| `duration_ms` | Queue plus trajectory duration, excluding trace-sink persistence. |
| `queue_latency_ms` | Time waiting for the concurrency slot. |
| `execution_ms` | Time spent executing the bounded loop. |
| `operational_success` | The run completed operationally. This is distinct from answering. |
| `answered` | The run completed with termination reason `answered`. It does not assert correctness or groundedness. |
| `agent_stopped` | The context planner explicitly selected `stop`; budget and progress-guard exits remain false. |
| `calls_by_kind`, `calls_by_status` | Counts reconstructed from recorded calls. |
| call error/attempt fields | Sanitized error categories, attempt status counts, retry counts, and attempt latency. |
| `call_latency_ms_by_kind` | Count, sum, and mean latency for retrieval, planner, reranker, and answer calls. Every latency summary includes `sum_overflow`; an overflowing sum and its mean are reported as `null`, never `Infinity`. |
| call queue/execution latency fields | Backend-capacity wait and `call_latency - queue_latency`, with coverage for traces that predate queue timing. Admission timeouts create no backend attempt. |
| `token_totals` | Sum of provider- or tokenizer-reported values that are actually present. |
| `token_coverage` | Numerator, denominator, and rate showing how much token accounting was observed. |
| `token_budget` | Configured limit, effective observed total, coverage, reached state, and exact overrun when fully observed. |
| cost/cache/TTFT fields | Known totals or rates plus coverage; unavailable provider data stays `null`. Cost-sum overflow is explicit and never emitted as non-standard `Infinity`. |
| `retrieval_query_hash_coverage` | Number of retrieval calls with a captured normalized-query hash divided by all retrieval calls. |
| `retrieval_requested_slot_count` | Sum of `top_k_requested` across retrieval calls, including failed calls with a retrieval payload. |
| `retrieval_slot_budget` | Configured maximum retrieval slots, `max_retrieval_calls * top_k`. |
| `retrieval_slot_budget_utilization` | Requested slots divided by the configured retrieval-slot budget, with explicit numerator and denominator. |
| retrieval query/document diversity | Within-run unique normalized queries or documents divided by their within-run occurrence counts. |
| per-hop novelty/Jaccard | New evidence contributed by each retrieval and overlap with the previous/all earlier hops. |
| context/citation fields | Final context size and the fraction of answer citations that refer to context documents. |
| `final_context_document_budget`, `final_context_char_budget` | Final context cardinality or characters divided by the corresponding configured limit. |
| backend/truncation fields | Upstream result counts, their coverage, and documents discarded by the local `top_k` guard. |

Missing token counts remain unknown; they are not silently converted to zero.
The raw call retains the usage source (`provider`, `tokenizer`, or `unknown`).
Per-call cost data is aggregated only when a backend supplies a numeric USD
value. `cost_coverage` prevents a partial provider report from looking like a
complete run cost. Cache-hit rate and time-to-first-token follow the same
known-value-plus-coverage rule.

Citation IDs are preserved as the planner emitted them. Duplicate or
out-of-context IDs are not silently accepted as grounded evidence: the metrics
report unique counts and in-context coverage so evaluation can reject or score
them explicitly.

Aggregate reports deliberately separate trajectory behavior from workload
coverage. `retrieval_*_diversity` combines per-run numerators and denominators;
`workload_unique_query_count` and `corpus_unique_document_count` describe global
coverage across all runs. Repeating the same evaluation query over several
seeds therefore does not masquerade as agent redundancy.
Aggregate output also includes value counts for schema version, experiment ID,
variant, seed, and objective, plus fingerprint counts and homogeneous flags.
`homogeneous_objective` explicitly detects a mixed `answer`/`context` report.
Inspect this composition and group incompatible configurations before
comparing variants; the aggregator deliberately reports mixtures instead of
silently guessing a grouping policy. Filter-hash counts, coverage, and
homogeneity do the same for per-request retrieval policies; `capture_text:
none` intentionally leaves that identity unknown. Requested-slot and final
context utilization are also aggregated from summed numerators and
denominators rather than averaging per-run rates.

A retrieval `CallTrace.latency_ms` measures the complete component call:
`call latency = component-capacity queue latency + component execution
latency`. The per-component timeout covers both terms. With the current
retrieval runtime, execution includes query embedding **and** document-store
search; it is not store-only latency. `AttemptTrace.latency_ms` covers only the
reported/synthesized backend attempt after admission. Provider-attempt,
component-queue, and whole-run latency therefore have different boundaries.

These are application-runtime metrics, not host telemetry. CPU, RSS, GPU
utilization, power, and VRAM are intentionally outside the core trace so the
agent package stays portable and does not require `psutil`, PyTorch, or NVML;
collect those alongside the JSONL artifact when a hardware benchmark needs
them.

## Offline Evidence Evaluation

Gold relevance labels stay outside the online loop. Evaluate a completed
`RunTrace` offline with `EvidenceLabels` and `evaluate_trace`:

```python
from justatom.agentic.evaluation import EvidenceLabels, evaluate_trace

labels = EvidenceLabels(
    qrels={"doc-17": 2.0, "doc-29": 1.0},
    required_evidence_groups=(
        {"doc-17", "doc-17-duplicate"},
        {"doc-29"},
    ),
)

report = evaluate_trace(result.trace, labels, k=10, context_k=20)
```

Persisted artifacts round-trip back into the same immutable schema objects:

```python
from justatom.agentic.telemetry import iter_jsonl_traces

for trace in iter_jsonl_traces(".data/traces/agentic-rag.jsonl"):
    runtime_metrics = derive_run_metrics(trace)
    evidence_metrics = evaluate_trace(trace, labels, k=10, context_k=20)
```

The loader rejects malformed lines, unknown schema fields, and unsupported
schema versions with the file and line number instead of silently skipping
them.

Positive qrels have relevance greater than zero. Each required-evidence group
contains interchangeable documents for one evidence slot; a run completes the
target only after it has retrieved at least one document from every group.
The labels are evaluator-only data: do not put qrels, positive IDs, required
groups, or any derived hint into the planner prompt, request metadata, filters,
or online stopping rule.

### Static vs Agentic Context Benchmark

Compare static and adaptive retrieval under the same maximum number of ranked
slots. For agentic per-hop depth `K` and maximum retrieval-call count `H`,
define the budget as `B = H * K`.

Use the following paired configuration:

| Variant | Retrieval configuration | Context/evaluation depth |
| --- | --- | --- |
| Static | one `/searching` call with `top_k=B` | evaluate the returned top `B` |
| Agentic context | `objective=context`, `top_k=K`, `max_retrieval_calls=H` | `max_context_documents=B`; call `evaluate_trace(..., k=K, context_k=B)` |

The initial original-query retrieval is one of the agent's `H` calls; it is
not a free extra hop. Keep the corpus/index revision, retriever mode, filters,
embedding model, character budget, and query set identical. Make
`max_steps`, `max_llm_calls`, and the wall-clock limit large enough that they do
not become an unintended tighter retrieval cap, then report any guard exits.

This equalizes the maximum retrieval-slot opportunity, not necessarily the
work actually spent: a context agent may select `stop` before `H` calls.
Report `retrieval_requested_slot_count`,
`retrieval_slot_budget_utilization`, retrieval/LLM calls, and latency alongside
quality. This supports both a quality-under-cap comparison and
quality-versus-actual-work curves without treating an early stop as if it had
spent all `B` slots. Keep `max_context_documents=B` and a common
`max_context_chars`; apply the same document truncation, deduplication, and
character cap when constructing the static context so clipping does not
silently favor either variant.

The report evaluates each retrieval hop, cumulative ranked slots, the
final context, and the final cited documents independently. It includes
precision, recall, hit rate, reciprocal rank, nDCG, first-hit position, and
evidence-completion position with their explicit denominators. These are
evidence-retrieval and citation-selection metrics. They do not
claim that the generated answer is correct, faithful, or sufficiently cited;
answer-quality evaluation needs a separate labelled or human-reviewed
protocol.

For the `context` objective, `final_context` is the primary output-quality
view. `final_citations` and `final_citation_set` are intentionally empty
because that mode does not select citations or generate an answer. The report
copies the trace `objective`, so downstream analysis can reject accidental
mixtures of answer-producing and context-only runs.

nDCG uses the same linear graded-qrel gain as the repository's retrieval
benchmarks. Gains are divided by their maximum finite grade before DCG is
summed; the shared scale cancels in the nDCG ratio and prevents finite qrels
from overflowing.

Per-hop ranking metrics use `evaluation_depth=k`; final-context metrics use
`evaluation_depth=context_k`. When `context_k` is omitted it defaults to `k`
for answer-mode traces and to the recorded `max_context_documents` for
context-mode traces. Missing result slots are zero-filled for standard
precision/nDCG semantics. Cumulative metrics use the spent retrieval
depth `hop * k` and preserve repeated document slots, so duplicates cannot
artificially improve a later document's reciprocal rank or nDCG. Every
retrieval call with a payload consumes one hop: a non-OK call contributes no
observed documents but still spends `k` cumulative slots.
`retrieval_hop_count`, `successful_hop_count`, and `failed_hop_count` make that
distinction explicit, and each per-hop and cumulative entry includes its call
`status` and `successful` flag. Evidence completion is deduplicated separately.
Final citations expose two views. `final_citations` preserves the first `k`
emitted occurrences: duplicates consume slots, receive zero gain after their
first occurrence, and `selection_precision` divides by observed occurrences.
`final_citation_set` deduplicates the full emitted list before taking `k` and
reports set-selection precision separately. Occurrence, unique, and duplicate
counts make the distinction explicit. Every section reports both
`evaluation_depth` and `observed_depth`.

Evidence evaluation uses recorded document IDs, so it works with all three
text-capture policies. Text-based answer evaluation is different: it requires
the answer from the runtime result or a suitably protected `full` trace.

Archive the raw JSONL trace alongside the immutable qrels, evaluation config,
secret-stripped resolved service config, exact planner prompt, corpus/index and
model revisions or artifact digests, code revision, dependency lock or
container digest, and a short hardware/runtime manifest. The fingerprints
verify equality but cannot reconstruct missing inputs. With those artifacts,
derived reports can be recomputed when a metric definition changes without
rerunning the online agent.
