# Habr IR Query Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select 5,000 Habr articles and prepare, submit, and collect strict OpenAI Batch requests for 10,000 grounded Russian IR query candidates.

**Architecture:** Immutable passages and retrieval indexes remain the source of truth. Focused modules select two high-quality targets per article, attach hybrid collision context, build strict `/v1/responses` JSONL, and persist idempotent Batch state. Remote work starts with 100 requests; scaling requires measured pilot integrity.

**Tech Stack:** Python 3.12, Polars/Parquet, NumPy, bm25s, OpenAI Python SDK 1.109.1, Responses API, Batch API, pytest.

## Global Constraints

- Corpus fingerprint: `bb6ad903b82c337a61cce2b1cd5bf5dd7e3303b6b3263258979372c00e40c3c9`.
- Generator: `gpt-5.6-terra`, reasoning effort `low`, endpoint `/v1/responses`.
- Strict Structured Outputs and `store=false`; no free-form parsing fallback.
- Seed `42`; 4,000/500/500 article-safe train/dev/test split.
- Two distinct passages and intents per article; prefer different sections.
- At most 1,000 requests and 100 MB per JSONL shard.
- Read `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`; never persist or log credentials.
- Atomic, checksummed, fingerprinted, resumable stage outputs.
- Submit a 100-request pilot before preparing all 10,000 requests.

---

### Task 1: Quality Scoring and Balanced Target Scheduling

**Files:**
- Create: `justatom/tooling/ir_dataset/targets.py`
- Create: `tests/test_ir_targets.py`
- Modify: `justatom/tooling/ir_dataset/__init__.py`

**Interfaces:**
- Produces `TargetSelectionConfig`, `PassageQuality`, `score_passage_quality`, and `select_target_slots`.
- Writes `targets.parquet`, exactly two slots per selected article.

- [ ] **Step 1: Write failing quality and scheduling tests**

```python
def test_quality_rejects_code_blob_but_keeps_explanatory_prose():
    assert score_passage_quality(prose_row()).eligible
    assert not score_passage_quality(code_blob_row()).eligible

def test_selection_is_article_safe_and_assigns_two_intents():
    selected = select_target_slots(target_frame(), TargetSelectionConfig(article_count=3))
    assert selected.group_by("article_id").len()["len"].min() == 2
    assert selected.group_by("article_id").agg(pl.col("split").n_unique())["split"].max() == 1
    assert selected.group_by("article_id").agg(pl.col("requested_intent").n_unique())["requested_intent"].min() == 2
```

- [ ] **Step 2: Run `python -m pytest tests/test_ir_targets.py -q` and verify RED**

- [ ] **Step 3: Implement explicit quality features**

Compute alphabetic-character ratio, code-symbol ratio, long-token ratio, line
statistics, repeated-character runs, and token length. Reject `token_count < 80`,
`alpha_ratio < 0.55`, `symbol_ratio > 0.18`, `long_token_ratio > 0.05`, and
repeated-character runs over 80. Preserve scores and reason codes.

- [ ] **Step 4: Implement topic, split, passage, and intent scheduling**

Allocate primary-flow quotas by square-root article frequency with a 30% cap.
Down-weight frequent hubs by inverse square-root frequency. Stable-sort with
seeded SHA-256. Prefer two non-empty distinct sections. Assign article-safe
splits and the exact intent proportions from the design spec, with distinct
intents for the two article slots.

- [ ] **Step 5: Verify GREEN and commit `feat: select balanced Habr IR targets`**

---

### Task 2: Target-specific Hybrid Collision Context

**Files:**
- Create: `justatom/tooling/ir_dataset/generation_context.py`
- Create: `tests/test_ir_generation_context.py`

**Interfaces:**
- Consumes targets, passages, `BM25Index`, and `DenseIndex`.
- Writes `generation_context.parquet` with three non-target passages per slot.

- [ ] **Step 1: Write a failing test asserting self-exclusion, three unique neighbors, and structural priority**
- [ ] **Step 2: Run the focused test and verify RED**
- [ ] **Step 3: Retrieve BM25 top-20 and dense top-20 from stored target embeddings**
- [ ] **Step 4: RRF-union candidates, force adjacent/same-article passages, remove self and exact-content duplicates**
- [ ] **Step 5: Select adjacent sibling, strongest non-sibling, then strongest remaining candidate; retain raw ranks/scores**
- [ ] **Step 6: Verify GREEN and commit `feat: build Habr generation collision context`**

---

### Task 3: Strict Generator Requests and Deterministic Gates

**Files:**
- Create: `justatom/tooling/ir_dataset/generation.py`
- Create: `tests/test_ir_generation.py`
- Modify: `configs/datasets/habr-ir.yaml`

**Interfaces:**
- Produces `GENERATOR_SCHEMA`, `GeneratorResult`, `build_generator_request`, and `validate_generator_result`.
- Stable ID: `gen-{sha256(article_id, passage_id, intent, prompt_hash, attempt)}`.

- [ ] **Step 1: Write failing request/schema/custom-ID/gate tests**

```python
def test_request_uses_responses_and_strict_schema():
    request = build_generator_request(slot(), context(), config())
    assert request["url"] == "/v1/responses"
    assert request["body"]["text"]["format"]["strict"] is True
    assert request["body"]["store"] is False

def test_gates_require_exact_evidence():
    assert "evidence_not_substring" in validate_generator_result(bad_evidence(), slot()).reason_codes
```

- [ ] **Step 2: Run focused tests and verify RED**
- [ ] **Step 3: Implement the Russian prompt and strict schema verbatim from the design spec**
- [ ] **Step 4: Build Responses bodies with model, low reasoning, structured `input`, strict `text.format`, and `store=false`**
- [ ] **Step 5: Gate schema, usable, exact evidence, 5-30 words, banned phrases, intent equality, query duplicates, target identity, and copied spans over eight tokens**
- [ ] **Step 6: Verify GREEN and commit `feat: prepare strict Habr query generation`**

---

### Task 4: Resumable Batch Files, Submission, and Collection

**Files:**
- Create: `justatom/tooling/ir_dataset/batch.py`
- Create: `tests/test_ir_batch.py`
- Modify: `justatom/api/ir_dataset.py`

**Interfaces:**
- Adds CLI stages `select-targets`, `prepare-generation`, `submit-generation`, `generation-status`, and `collect-generation`.
- Persists request/output SHA-256, OpenAI file IDs, Batch IDs, status, and counts.

- [ ] **Step 1: Write failing shard and fake-client idempotency tests**

```python
def test_shards_respect_caps(tmp_path):
    shards = write_batch_shards(requests(1001), tmp_path, max_requests=1000, max_bytes=100_000_000)
    assert [item.request_count for item in shards] == [1000, 1]

def test_submit_is_idempotent(tmp_path):
    first = submit_pending_shards(state(), FakeOpenAI(), tmp_path)
    assert submit_pending_shards(first, FakeOpenAI(), tmp_path) == first
```

- [ ] **Step 2: Run focused tests and verify RED**
- [ ] **Step 3: Write fsynced atomic JSONL shards and reject duplicate custom IDs**
- [ ] **Step 4: Fingerprint targets, context, prompt, schema, and model; refuse checksum mismatch reuse**
- [ ] **Step 5: Upload with `files.create(..., purpose="batch")` and submit `/v1/responses` with a `24h` window**
- [ ] **Step 6: Retrieve status, download output/error files, match unique custom IDs, and parse only HTTP 200 response bodies**
- [ ] **Step 7: Verify GREEN and commit `feat: add resumable Habr generation batches`**

---

### Task 5: 100-request Remote Pilot

**Files:**
- Generate ignored artifacts under `.tmp_runs/datasets/habr-ir/generation-v1/`.

- [ ] **Step 1: Run `python -m pytest -m 'not integration' -q`**
- [ ] **Step 2: Materialize 50 articles, 100 slots, and three contexts per slot**
- [ ] **Step 3: Assert no article leakage, distinct targets/intents, immutable source IDs, and no credentials in files**
- [ ] **Step 4: Submit exactly one 100-request pilot shard and persist file ID, Batch ID, checksum, and status**
- [ ] **Step 5: Do not block for 24 hours; report the command/state used for status and collection**
- [ ] **Step 6: After collection, report schema success, usable rate, deterministic-gate pass, intent agreement, evidence validity, duplicates, and token usage**
- [ ] **Step 7: Prepare all 10,000 requests only if at least 70% are usable and 60% pass deterministic gates; otherwise version and rerun the pilot**
- [ ] **Step 8: Run fresh tests, `git diff --check`, and `git status --short`; never commit `.tmp_runs` or secrets**
