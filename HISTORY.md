# HISTORY

Last updated: 2026-03-10

## Purpose
This file is a handoff log for future sessions (human or LLM). It summarizes what was changed, why it was changed, how to validate it, and what still needs attention.

## High-Level Timeline
1. Audited repository for multimodal retrieval readiness and local llama.cpp embedding flow.
2. Migrated notebook-oriented abstractions into runtime code.
3. Built and iterated smoke diagnostics for OpenAI-compatible and native multimodal embeddings.
4. Added benchmark tooling for Flickr8k cross-modal metric-space checks.
5. Fixed benchmark methodology issues (notably text embedding extraction/pooling behavior and protocol correctness).
6. Added embedding backend configuration plumbing (prefix logic, defaults, max_seq_len, env-sanitized loading).
7. Added standalone diagnostics utility and Makefile target for embedding checks.
8. Introduced and later removed a temporary master/runtime config layer during the config migration.
9. Extended eval CLI to read defaults from config and support dotted CLI overrides.
10. Removed duplicated legacy config sources and moved remaining defaults to scenario configs or in-code compatibility defaults.
11. Refactored training orchestration into class-based jobs and generalized train APIs away from gamma-specific naming.
12. Introduced streaming training-row preparation with lazy iterators, reservoir sampling, and backend-aware eager/lazy handling.
13. Added follow-up inline documentation in `justatom/api/train.py` to make dataset-vs-batch-vs-row-stream semantics explicit.

## Detailed Change Log

### 1) Runtime and API Embedding Stack
- Added and/or stabilized abstractions in `justatom/running/llm.py` for task/wrapper migration from notebook logic.
- Kept exports in `justatom/api/__init__.py` aligned with runtime placement.
- Iterated snapshot/log persistence behavior in async paths.

Key outcome:
- Core abstractions moved out of notebooks and into production-oriented runtime modules.

### 2) Embedding Smoke and Diagnostics Scripts
- Created/updated `scripts/smoke_embed_qwen_vl.py` for multimodal probing against llama.cpp server.
- Added official-style native payload handling (`/embeddings` with `prompt_string` + `multimodal_data`).
- Hardened probe ordering and diagnostics to reduce false positives from text fallback behavior.

Key outcome:
- Reliable operator script for quick pass/fail checks and endpoint mode identification.

### 3) Flickr8k Metric-Space Benchmarking
- Created and improved `scripts/benchmark_flickr8k_metric_space.py`.
- Added support for multi-caption protocol and full-run artifact export (`.npz`, metrics json).
- Corrected methodology to avoid misleading near-random results caused by extraction/pooling mismatch.

Key outcome:
- End-to-end benchmark execution and saved artifacts for reproducible evaluation.

### 4) Embedding Config and Client Behavior
- Added env-aware builtins loader behavior in `justatom/configuring/builtins.py`.
- Added/updated `justatom/builtins/configs/embeddings.yaml` defaults.
- Added `justatom/builtins/prompts/embedding_prefix_guide.txt`.
- Extended `justatom/running/embeddings/openai_compatible.py` with:
  - query/passage prefix strategy
  - default request knobs (pooling, encoding_format)
  - `max_seq_len` handling
- Updated factory/wiring in:
  - `justatom/running/embeddings/__init__.py`
  - `justatom/running/service.py`

Key outcome:
- Prefix-aware OpenAI-compatible embeddings with centralized defaults and runtime override hooks.

### 5) Embedding Doctor Utility
- Added `scripts/embedding_doctor.py` as a standalone (dependency-light) diagnostics utility.
- Utility checks:
  - config resolution and unresolved placeholder reporting
  - server reachability (`GET /v1/models`)
  - OpenAI-compatible text embedding probes
  - native multimodal probe (`POST /embeddings`)
- Added Makefile target:
  - `check-embeddings`

Key outcome:
- One-command diagnostics for local embedding stack health.

### 6) Config Architecture Refactor
- Scenario configs became the primary source for `train` and `eval` flows.
- Legacy `Config` compatibility was reduced to in-code defaults in `justatom/configuring/prime.py`.
- Temporary master/runtime catalog files used during migration were removed after consumers were migrated away.

Key outcome:
- No runtime dependency on the removed master/runtime config layer.

### 7) Eval CLI Defaults + Dotted Override Support
- Updated `justatom/api/eval.py` to:
  - load structured defaults from scenario config
  - keep backward compatibility with existing legacy flags
  - support dotted CLI overrides, e.g.:
    - `--model.name="..."`
    - `--search.pipeline=keywords`
    - `--search.top_k=7`
    - `--metrics.top_k="['HitRate','mrr']"`

Key outcome:
- Structured defaults in config with ergonomic runtime overrides from CLI.

### 8) Training Orchestration Cleanup
- Extracted class-based training orchestration into `justatom/running/trainer_jobs.py`.
- Generalized training entrypoints and helpers to avoid gamma-specific naming in runtime APIs.
- Kept the public train flow compatible while moving orchestration details out of the API layer.

Key outcome:
- Train flow is easier to extend and reason about because job orchestration is separated from CLI/API glue.

### 9) Streaming Training Data Preparation
- Refactored `justatom/api/train.py` so training examples can be prepared as a stream instead of materializing the full dataset up front.
- Added/cleaned up helpers around the train-data path:
  - `_frame_batches_from_source(...)`
  - `_iterate_from_frame_batches(...)`
  - `iterate_training_rows(...)`
  - `_reservoir_sample_rows(...)`
  - `sample_training_rows(...)`
  - `prepare_training_data(...)`
- Introduced a lazy row iterator + bounded reservoir sample flow:
  - iterate rows lazily from the source
  - keep only a fixed-size in-memory sample for fitting
  - preserve compatibility for downstream consumers that still expect materialized sampled rows
- Important backend behavior discovered and preserved:
  - eager `pl.DataFrame` sources are wrapped as `[source]` so downstream code can always consume `Iterable[pl.DataFrame]`
  - lazy `pl.LazyFrame` sources use `collect_batches(maintain_order=True)`
  - non-polars sources return `None` from `_frame_batches_from_source(...)`, which intentionally triggers the adapter fallback branch

Key outcome:
- Training data prep now supports true streaming for supported backends while keeping backward-compatible sampled output.

### 10) Dataset Backend Findings Relevant To Train Flow
- Reviewed `justatom/storing/dataset.py` to confirm actual iterator contracts.
- Verified current behavior:
  - `JSONDataset.iterator(lazy=True)` still returns eager `pl.DataFrame`
  - `JSONLinesDataset.iterator(lazy=True)` can return `pl.LazyFrame`
  - `PARQUETDataset.iterator(...)` and `CSVDataset.iterator(...)` can use polars eager/lazy paths
  - `JUSTATOMDataset.iterator(lazy=True)` currently resolves through the JSON backend and therefore behaves eagerly
  - `HFDataset.iterator(...)` returns Hugging Face dataset objects, not polars frames
- Consequence for `justatom/api/train.py`:
  - `_frame_batches_from_source(...)` returning `None` is expected and correct for non-polars backends, especially `hf://...`

Key outcome:
- The `None` return from `_frame_batches_from_source(...)` is a deliberate control-flow signal, not an error condition.

### 11) Train.py Readability / Handoff Comments
- Added focused inline comments in `justatom/api/train.py` to disambiguate the following concepts:
  - whole dataset source
  - one materialized DataFrame batch
  - lazy row iterator
  - bounded sampled subset kept in memory
- Clarified the reason for wrapping eager frames as `[source]`:
  - normalize eager and lazy paths into the same `for batch in frame_batches` contract
- Clarified fallback semantics near the non-polars branch.
- Removed a misleading narrow type annotation on `source` because actual backends are broader than `pl.DataFrame | pl.LazyFrame`.

Key outcome:
- Future readers should be able to understand the train-data flow directly from `justatom/api/train.py` without re-deriving backend behavior from scratch.

## Validation and Regression Checks

### Config checks
- Validate scenario configs and focused config-loader tests.

### Embedding diagnostics
- `make check-embeddings`
- Expected (when server is up):
  - `/v1/models` reachable
  - text embedding probes succeed
  - native multimodal probe succeeds

### Tests (focused)
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate justatom`
- `python -m pytest -q tests/test_retriever_shape_and_factory.py tests/test_eval_metrics.py tests/test_eval_data_normalization.py`
- Latest known status in-session: passing.

### Tests and checks from the train-data refactor
- Full suite previously passed after the larger refactor:
  - `53 passed, 6 warnings`
- Targeted integration coverage exists in `tests/test_eval_streaming_integration.py`.
- Dev verification scripts used during the lazy/eager investigation:
  - `scripts/dev/check_lazy_json_justatom.py`
  - `scripts/dev/check_lazy_jsonl_from_justatom.py`
- What those checks established:
  - `.json` requests asking for lazy iteration still fall back to eager frame loading
  - `.jsonl` requests can use a truly lazy path
- After the later comment-only edit to `justatom/api/train.py`, diagnostics for that file were checked and no errors were reported.

## Operational Notes
- Some scripts require `PYTHONPATH=.` when run directly from shell.
- Local environment was primarily tested in conda env `justatom`.
- llama.cpp behavior may vary by launch flags (`--embeddings`, pooling mode, batch/token constraints).

## Known Caveats
- Multimodal stability can still depend heavily on model build and llama-server launch configuration.
- Train-data laziness is backend-dependent. "lazy=True" at the API level does not guarantee a lazy backend implementation.
- The `justatom` named dataset path currently behaves eagerly because it resolves through the JSON backend.
- Hugging Face datasets follow the adapter fallback path in `justatom/api/train.py`; they do not use the polars batch fast path.

## Current Train-Flow Mental Model
1. Call `iterate_training_rows(...)` when you want a lazy stream of normalized training rows.
2. If the backend returns polars frames:
  - eager `DataFrame` becomes `[source]` and is treated as one batch
  - lazy `LazyFrame` yields multiple materialized batches via `collect_batches(...)`
3. If the backend is not polars:
  - `_frame_batches_from_source(...)` returns `None`
  - code falls back to `DatasetRecordAdapter.from_source(...)`
4. `sample_training_rows(...)` consumes that lazy stream and applies reservoir sampling.
5. `prepare_training_data(...)` materializes only the sampled subset needed by the existing fit/train consumers.

## Suggested Next Steps
1. Add a short README section documenting config layering and dotted override examples for `justatom.api.eval`.
2. Add unit tests for `_parse_args` dotted override behavior in `justatom/api/eval.py`.
3. Add a small unit test that documents `_frame_batches_from_source(...) -> None` for a mocked non-polars/HF-like source.
4. Consider adding optional local override file support (for developer-only changes) if needed later.
