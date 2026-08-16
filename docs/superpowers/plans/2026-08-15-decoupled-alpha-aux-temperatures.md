# Decoupled Alpha Auxiliary Temperatures Implementation Plan

> **For Codex:** Execute this plan task by task with test-driven development. Do not alter the memory-bank path or the canonical method presets while the new temperatures are still experimental.

**Goal:** Separate the alpha confidence-target and SimCSE temperatures from the retrieval temperature while preserving legacy behavior exactly when the new options are unset.

**Architecture:** `TrainConfig` owns the two optional values, `TrainingModule` forwards the alpha target temperature with each objective call, and `ContrastiveObjective` selects either its retrieval kernel or a fixed non-learnable SimCSE kernel. The gate target remains detached, the gate BCE updates only the gate head, and the weighted SimCSE term updates only encoder representations.

**Tech Stack:** Python 3.10-3.13, PyTorch, pytest, dataclasses, Lightning training module, YAML/dotted CLI configuration.

---

### Task 1: Add the configuration contract

**Files:**
- Modify: `tests/test_training_config.py`
- Modify: `justatom/training/config.py`

**Step 1: Write failing configuration tests**

Add tests that prove:

```python
config = parse_train_config(
    {
        "method": "atom_gate",
        "objective": {"simcse_temperature": 0.2},
        "alpha_gate": {"target_temperature": 0.3},
    }
)
assert config.objective.simcse_temperature == pytest.approx(0.2)
assert config.alpha_gate.target_temperature == pytest.approx(0.3)
assert parse_train_config(train_config_to_dict(config)) == config
```

Parameterize both fields over `0.0`, `-0.1`, `nan`, `inf`, and `-inf` and assert a `ValueError`. Also assert that omitted fields resolve to `None`, and that fixed SimCSE values outside the kernel range `[1e-3, 1.0]` are rejected instead of silently clamped.

**Step 2: Run the tests to verify RED**

Run:

```bash
conda run -n justatom pytest -q tests/test_training_config.py
```

Expected: failures because both dataclass fields are unknown.

**Step 3: Implement the typed fields and validation**

Add:

```python
class ObjectiveConfig:
    simcse_temperature: float | None = None

class AlphaGateConfig:
    target_temperature: float | None = None
```

Validate each non-null value as finite and strictly positive. Restrict the fixed SimCSE value to the effective `ContrastiveLoss` range `[1e-3, 1.0]`. Keep defaults at `None`; do not change method presets.

**Step 4: Run the tests to verify GREEN**

Run:

```bash
conda run -n justatom pytest -q tests/test_training_config.py
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add tests/test_training_config.py justatom/training/config.py
git commit -m "feat(training): configure independent alpha temperatures"
```

### Task 2: Implement the objective mathematics and telemetry

**Files:**
- Modify: `tests/test_training_objective.py`
- Modify: `justatom/training/objective.py`

**Step 1: Write failing objective tests**

Add focused tests for these contracts:

1. `simcse_temperature=None` and `alpha_target_temperature=None` reproduce the legacy formulas using `objective.kernel.tau`.
2. A fixed target temperature changes `alpha_target` and the BCE term but leaves `main_per_row` unchanged.
3. A fixed SimCSE temperature changes `simcse_per_row` but leaves `main_per_row` unchanged.
4. The fixed SimCSE kernel has no learnable temperature parameter.
5. Gate BCE produces gradients only for `alpha_logits`; weighted SimCSE produces gradients for `queries` and `query_alt`, not `alpha_logits`.
6. Metrics report the raw and weighted auxiliary, auxiliary-to-main ratio, and the three effective temperatures.

Use a non-symmetric normalized tensor so the two temperatures produce observably different softmax values. Compute expectations directly with `F.cross_entropy` and `F.binary_cross_entropy_with_logits`.

**Step 2: Run the tests to verify RED**

Run:

```bash
conda run -n justatom pytest -q tests/test_training_objective.py
```

Expected: failures for the missing input field, auxiliary kernel, and telemetry.

**Step 3: Implement independent temperature selection**

Extend `ObjectiveInputs` with:

```python
alpha_target_temperature: float | None = None
```

Build a fixed `ContrastiveLoss` only when `ObjectiveConfig.simcse_temperature` is non-null:

```python
self.simcse_kernel = ContrastiveLoss(
    temperature=config.simcse_temperature,
    reduction="none",
    learnable_temperature=False,
    decoupled=config.decoupled,
) if config.simcse_temperature is not None else None
```

Use `self.simcse_kernel or self.kernel` for `L_sc`. For the target, use a detached tensor containing the configured value, otherwise the detached live retrieval temperature. Preserve the current retrieval kernel and main loss unchanged.

Record:

```text
temperature/simcse
temperature/alpha_target
loss/alpha_aux_weighted
loss/alpha_aux_to_main_ratio
```

Keep `loss/alpha_aux` as raw mean SimCSE. Compute the ratio from detached means with a small denominator clamp.
Emit each new metric only when its corresponding alpha or SimCSE path is active, and preserve the denominator sign in DCL ablations.

**Step 4: Run the tests to verify GREEN**

Run:

```bash
conda run -n justatom pytest -q tests/test_training_objective.py
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add tests/test_training_objective.py justatom/training/objective.py
git commit -m "feat(training): decouple alpha auxiliary temperatures"
```

### Task 3: Wire the module and document the public configuration

**Files:**
- Modify: `tests/test_training_module.py`
- Modify: `tests/test_training_job.py`
- Modify: `justatom/training/module.py`
- Modify: `docs/training.md`

**Step 1: Write failing plumbing and compatibility tests**

Add tests that:

- construct an atom-gate module with both temperatures set and assert the emitted temperature and weighted-loss telemetry;
- restore a historical config dictionary with both fields absent and assert both defaults are `None`;
- serialize a job manifest and assert both resolved values are present when configured.

**Step 2: Run the tests to verify RED**

Run:

```bash
conda run -n justatom pytest -q tests/test_training_module.py tests/test_training_job.py
```

Expected: at least the module telemetry test fails because the target temperature is not forwarded.

**Step 3: Forward the target temperature**

Pass:

```python
alpha_target_temperature=self.config.alpha_gate.target_temperature
```

into `ObjectiveInputs`. Do not change checkpoint schema: dataclass defaults provide backward compatibility.

**Step 4: Document configuration and formulas**

In `docs/training.md`, document the two optional YAML fields, their dotted CLI names, the exact `None` fallback, the autograd boundaries, and the four new telemetry fields. Mark `0.2` as an experiment value, not a canonical default.

**Step 5: Run focused tests to verify GREEN**

Run:

```bash
conda run -n justatom pytest -q tests/test_training_module.py tests/test_training_job.py
```

Expected: all tests pass.

**Step 6: Commit**

```bash
git add tests/test_training_module.py tests/test_training_job.py justatom/training/module.py docs/training.md
git commit -m "docs(training): expose alpha auxiliary temperatures"
```

### Task 4: Verify compatibility and run the mMARCO mechanism screen

**Files:**
- Create under ignored run root: `.tmp_runs/mmarco_qwen_alpha_temp_<timestamp>/`
- Reuse: `.tmp_runs/mmarco_qwen_alpha_20260815_164000/runs/coupled/encoder`
- Reuse: `.tmp_runs/mmarco_ru_clean_eval/dev1000_corpus10000_seed44.parquet`

**Step 1: Run focused and full offline tests**

Run:

```bash
conda run -n justatom pytest -q \
  tests/test_training_config.py \
  tests/test_training_objective.py \
  tests/test_training_module.py \
  tests/test_training_job.py
conda run -n justatom pytest -q
```

Expected: all tests pass with no network dependency.

**Step 2: Run the seed-44 alpha-temperature screen**

Reuse the exact clean mMARCO split and baseline from the prior seed-44 run. Train only the alpha variant with:

```yaml
objective:
  temperature: 0.05
  simcse_temperature: 0.2
alpha_gate:
  target_temperature: 0.2
```

Keep Qwen3-Embedding-0.6B, LoRA rank 16/alpha 32/dropout 0.05/RS-LoRA/all-linear, 6000 train pairs, one epoch, and seed 44 unchanged.

**Step 3: Evaluate the mechanism gate**

Aggregate training telemetry and require:

```text
mean(alpha_aux_weight) >= 0.10
mean(loss/alpha_aux) >= 1e-3
mean(loss/alpha_aux_weighted) / mean(loss/main) >= 0.01
```

If this fails, stop expansion and record the negative result.

**Step 4: Evaluate retrieval against the saved baseline**

On the same 1000-query / 10000-document screen, compare `HitRate@1` and `MRR@10`. Require both to be no worse than baseline and at least one to improve before running seed 45.

**Step 5: Record results**

Write `RESULTS.md` in the run root with the resolved config, exact artifact paths, mechanism telemetry, retrieval metrics, deltas, and the gate decision. Never combine this run with a memory-bank variant unless the alpha-only mechanism first passes independently.

**Step 6: Final verification commit**

Commit only tracked implementation/docs/test changes. Keep run artifacts under `.tmp_runs/` ignored.
