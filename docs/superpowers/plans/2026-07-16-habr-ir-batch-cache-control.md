# Habr IR Batch Cache Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate the next disjoint 3,000 Habr IR requests with GPT-5.6 Terra's implicit prompt cache disabled.

**Architecture:** Extend `GeneratorConfig` with a backwards-compatible cache mode. Request construction maps `explicit` to the Responses API `prompt_cache_options` payload, while generation fingerprints include the mode so incompatible artifacts cannot be reused.

**Tech Stack:** Python 3.12, dataclasses, OpenAI Batch/Responses API, pytest, YAML, Parquet.

## Global Constraints

- Existing configs default to `prompt_cache_mode: auto` behavior.
- `explicit` disables implicit caching and defines no cache breakpoint.
- The new cohort must not overlap either prior 1k or prior 3k targets.
- Submission must contain three shards of exactly 1,000 unique requests.

---

### Task 1: Configurable prompt-cache mode

**Files:**
- Modify: `justatom/tooling/ir_dataset/generation.py:121-154`
- Modify: `justatom/tooling/ir_dataset/generation.py:325-368`
- Modify: `justatom/tooling/ir_dataset/batch.py:188-215`
- Test: `tests/test_ir_generation.py`
- Test: `tests/test_ir_batch.py`

**Interfaces:**
- Consumes: `GeneratorConfig` mappings loaded by `justatom.api.ir_dataset.load_ir_dataset_config`.
- Produces: `GeneratorConfig.prompt_cache_mode: Literal["auto", "explicit"]` and conditional `body.prompt_cache_options`.

- [ ] **Step 1: Write failing request and validation tests**

```python
def test_explicit_prompt_cache_mode_disables_implicit_breakpoint():
    request = build_generator_request(slot(), context(), config(prompt_cache_mode="explicit"))
    assert request["body"]["prompt_cache_options"] == {"mode": "explicit"}


def test_auto_prompt_cache_mode_preserves_api_default():
    request = build_generator_request(slot(), context(), config())
    assert "prompt_cache_options" not in request["body"]


def test_prompt_cache_mode_rejects_unknown_values():
    with pytest.raises(ValueError, match="prompt_cache_mode"):
        GeneratorConfig(prompt_cache_mode="disabled")
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q tests/test_ir_generation.py`

Expected: failures because `GeneratorConfig` does not accept `prompt_cache_mode`.

- [ ] **Step 3: Implement config validation and request mapping**

```python
class GeneratorConfig:
    prompt_cache_mode: str = "auto"

    def __post_init__(self) -> None:
        if self.prompt_cache_mode not in {"auto", "explicit"}:
            raise ValueError("generation.prompt_cache_mode must be one of: auto, explicit")


body = {...}
if active_config.prompt_cache_mode == "explicit":
    body["prompt_cache_options"] = {"mode": "explicit"}
```

- [ ] **Step 4: Include the mode in `_generation_fingerprint`**

```python
payload = {
    # existing fields
    "prompt_cache_mode": config.prompt_cache_mode,
}
```

Add a batch test that prepares identical targets with `auto` and `explicit` in separate roots and asserts their `generation_fingerprint` values differ.

- [ ] **Step 5: Run focused and full tests**

Run: `pytest -q tests/test_ir_generation.py tests/test_ir_batch.py tests/test_ir_release.py`

Expected: all focused tests pass.

Run: `pytest -q`

Expected: the full suite passes.

- [ ] **Step 6: Commit**

```bash
git add justatom/tooling/ir_dataset/generation.py justatom/tooling/ir_dataset/batch.py tests/test_ir_generation.py tests/test_ir_batch.py
git commit -m "feat: control Habr Batch prompt caching"
```

### Task 2: Next disjoint 3k cohort

**Files:**
- Create: `configs/datasets/habr-ir-next-3k-v2.yaml`

**Interfaces:**
- Consumes: `target_selection.exclude_target_roots` and `generation.prompt_cache_mode`.
- Produces: generation root `generation-next-3k-v2` and release root `next-3k-release-v2`.

- [ ] **Step 1: Create the cohort config**

Copy the pinned source, chunking, retrieval, and preparation values from `configs/datasets/habr-ir-next-3k.yaml`, then set:

```yaml
target_selection:
  article_count: 1500
  seed: 42
  max_flow_share: 0.30
  exclude_target_roots:
    - .tmp_runs/datasets/habr-ir/generation-1k-canary-v1
    - .tmp_runs/datasets/habr-ir/generation-next-3k-v1

generation:
  model: gpt-5.6-terra
  reasoning_effort: low
  prompt_cache_mode: explicit
  max_requests_per_shard: 1000
  max_shard_bytes: 100000000
  max_batch_attempts: 2
  scale_authorized: true

output:
  generation_root: .tmp_runs/datasets/habr-ir/generation-next-3k-v2
  release_root: .tmp_runs/datasets/habr-ir/next-3k-release-v2
```

- [ ] **Step 2: Select targets and prepare generation requests**

Run:

```bash
python -m justatom.api.ir_dataset select-targets --config configs/datasets/habr-ir-next-3k-v2.yaml
python -m justatom.api.ir_dataset prepare-generation --config configs/datasets/habr-ir-next-3k-v2.yaml
```

Expected: 3,000 targets from 1,500 articles and three request shards.

- [ ] **Step 3: Validate cohort and payload invariants**

Use Parquet/JSON structured readers to assert:

```text
new target rows = 3000
new target articles = 1500
article overlap with either prior cohort = 0
passage overlap with either prior cohort = 0
shard request counts = [1000, 1000, 1000]
all request bodies contain prompt_cache_options == {"mode": "explicit"}
all custom_ids are unique
```

- [ ] **Step 4: Commit config**

```bash
git add configs/datasets/habr-ir-next-3k-v2.yaml
git commit -m "chore: add next Habr IR 3k cohort"
```

- [ ] **Step 5: Submit and poll once**

Run:

```bash
python -m justatom.api.ir_dataset submit-generation --config configs/datasets/habr-ir-next-3k-v2.yaml
python -m justatom.api.ir_dataset generation-status --config configs/datasets/habr-ir-next-3k-v2.yaml
```

Expected: exactly three submitted Batch IDs with no duplicate uploads.
