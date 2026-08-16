# Gradient-Safe Query Auxiliary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an experimental `atom_gate` controller that applies its difficulty-weighted SimCSE gradient only when it is compatible with the coupled InfoNCE gradient and caps its norm relative to the retrieval gradient.

**Architecture:** `ContrastiveObjective` exposes retrieval, encoder-auxiliary, head, and memory losses separately. A pure gradient-control module computes parameter-space cosine, ReLU compatibility, and retrieval-relative norm scaling; `ContrastiveTrainingModule` uses manual optimization only for `observe` and `safe` modes while preserving canonical automatic optimization. The new behavior remains an ablation until matched runs pass.

**Tech Stack:** Python 3.10-3.13, PyTorch, PyTorch Lightning, dataclasses, pytest, YAML/dotted CLI configuration, shell benchmark wrappers.

## Global Constraints

- Keep canonical `vanilla`, `atom_gate`, and `atomic` presets unchanged.
- Permit the new controller only for `method: atom_gate` with `experiment.role: ablation`.
- Keep the memory bank disabled in the first diagnostic and matched comparison.
- Calculate compatibility over actual trainable parameter gradients, not representation-space proxies.
- Preserve alpha-head detachment and add its BCE gradient unchanged.
- Apply control per contrastive microbatch before manual gradient accumulation.
- Use ordinary PyTorch operations supported on CPU, MPS, and CUDA; add no dependency.
- Claim first-order training-direction safety only, never guaranteed evaluation improvement.

---

### Task 1: Add the Auxiliary Gradient Configuration Contract

**Files:**
- Modify: `justatom/training/config.py`
- Modify: `justatom/training/methods.py`
- Modify: `configs/train.yaml`
- Modify: `tests/test_training_config.py`
- Modify: `tests/test_training_methods.py`

**Interfaces:**
- Produces: `AuxiliaryGradientMode(str, Enum)` with `OFF`, `OBSERVE`, and `SAFE`.
- Produces: `AuxiliaryGradientConfig(mode, max_norm_ratio, eps)` at `TrainConfig.auxiliary_gradient`.
- Consumes: existing `ExperimentRole`, `TrainingMethod`, and dataclass overlay/serialization.

- [ ] **Step 1: Write failing configuration tests**

Add tests equivalent to:

```python
def test_auxiliary_gradient_config_round_trips():
    config = parse_train_config(
        {
            "method": "atom_gate",
            "experiment": {"role": "ablation"},
            "auxiliary_gradient": {
                "mode": "safe",
                "max_norm_ratio": 0.25,
                "eps": 1e-10,
            },
        }
    )
    assert config.auxiliary_gradient.mode is AuxiliaryGradientMode.SAFE
    assert config.auxiliary_gradient.max_norm_ratio == pytest.approx(0.25)
    assert parse_train_config(train_config_to_dict(config)) == config
```

Parameterize invalid modes, `max_norm_ratio` values `-0.1`, `nan`, and `inf`,
and `eps` values `0.0`, `-1.0`, `nan`, and `inf`. Add method-resolution tests
that reject `observe`/`safe` for canonical `atom_gate`, `vanilla`, and `atomic`,
and accept both modes for an `atom_gate` ablation.

- [ ] **Step 2: Run the tests to verify RED**

Run:

```bash
conda run -n justatom pytest -q tests/test_training_config.py tests/test_training_methods.py
```

Expected: failures because `auxiliary_gradient` and its enum do not exist.

- [ ] **Step 3: Implement the typed config and validation**

Add:

```python
class AuxiliaryGradientMode(str, Enum):
    OFF = "off"
    OBSERVE = "observe"
    SAFE = "safe"


@dataclass(frozen=True)
class AuxiliaryGradientConfig:
    mode: AuxiliaryGradientMode = AuxiliaryGradientMode.OFF
    max_norm_ratio: float = 0.25
    eps: float = 1e-12
```

Add `auxiliary_gradient: AuxiliaryGradientConfig` to `TrainConfig`, validate
the numeric constraints, and add `auxiliary_gradient: {}` to `configs/train.yaml`.
In `resolve_method`, require `ATOM_GATE` plus `ExperimentRole.ABLATION` whenever
the mode is not `OFF`. Do not change `canonical_method_config` defaults.

- [ ] **Step 4: Run the focused tests to verify GREEN**

Run:

```bash
conda run -n justatom pytest -q tests/test_training_config.py tests/test_training_methods.py
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit the configuration contract**

```bash
git add justatom/training/config.py justatom/training/methods.py configs/train.yaml tests/test_training_config.py tests/test_training_methods.py
git commit -m "feat(training): configure safe auxiliary gradients"
```

### Task 2: Implement the Pure Gradient Compatibility Controller

**Files:**
- Create: `justatom/training/auxiliary_gradient.py`
- Create: `tests/test_auxiliary_gradient.py`

**Interfaces:**
- Produces: `control_auxiliary_gradients(primary, auxiliary, *, mode, max_norm_ratio, eps) -> tuple[list[Tensor | None], AuxiliaryGradientStats]`.
- Produces telemetry through `AuxiliaryGradientStats.metrics()`.
- Consumes: `AuxiliaryGradientMode` and gradient lists aligned with optimizer parameters.

- [ ] **Step 1: Write failing mathematical unit tests**

Cover aligned, partially aligned, orthogonal, conflicting, zero-primary,
zero-auxiliary, mismatched-list, and non-finite cases. The central assertions
must include:

```python
primary = [torch.tensor([1.0, 0.0])]
auxiliary = [torch.tensor([2.0, 0.0])]
controlled, stats = control_auxiliary_gradients(
    primary,
    auxiliary,
    mode=AuxiliaryGradientMode.SAFE,
    max_norm_ratio=0.25,
    eps=1e-12,
)
torch.testing.assert_close(controlled[0], torch.tensor([0.25, 0.0]))
assert stats.cosine_scale == pytest.approx(1.0)
assert stats.total_scale == pytest.approx(0.125)

direction = primary[0] + controlled[0]
assert torch.dot(primary[0], direction) >= torch.dot(primary[0], primary[0])
assert controlled[0].norm() <= 0.25 * primary[0].norm()
```

For a negative or exactly zero dot product, assert that `safe` returns a zero
shared auxiliary gradient. For `observe`, assert that the original auxiliary
is returned exactly while statistics are still reported.

- [ ] **Step 2: Run the test to verify RED**

```bash
conda run -n justatom pytest -q tests/test_auxiliary_gradient.py
```

Expected: import failure for the missing module.

- [ ] **Step 3: Implement float32 statistics and scaling**

Implement the following calculation over the complete parameter-aligned lists:

```python
dot = sum((g.float() * h.float()).sum() for g, h in shared_pairs)
primary_norm = sqrt(sum(g.float().square().sum() for g in primary if g is not None))
auxiliary_norm = sqrt(sum(h.float().square().sum() for h in auxiliary if h is not None))
cosine = dot / clamp(primary_norm * auxiliary_norm, min=eps)

if mode is AuxiliaryGradientMode.OBSERVE:
    cosine_scale = 1.0
    norm_scale = 1.0
else:
    cosine_scale = clamp(cosine, min=0.0)
    candidate_norm = cosine_scale * auxiliary_norm
    norm_scale = min(1.0, max_norm_ratio * primary_norm / (candidate_norm + eps))
    if dot <= 0 or primary_norm <= eps or auxiliary_norm <= eps:
        cosine_scale = 0.0
        norm_scale = 0.0

total_scale = cosine_scale * norm_scale
controlled = [None if h is None else h.detach().clone() * total_scale for h in auxiliary]
```

Reject non-finite aggregate statistics before returning. Metrics use the exact
names from the design spec.

- [ ] **Step 4: Run the mathematical tests to verify GREEN**

```bash
conda run -n justatom pytest -q tests/test_auxiliary_gradient.py
```

Expected: all tests pass on the active platform.

- [ ] **Step 5: Commit the pure controller**

```bash
git add justatom/training/auxiliary_gradient.py tests/test_auxiliary_gradient.py
git commit -m "feat(training): control auxiliary gradient compatibility"
```

### Task 3: Expose an Exact Objective Decomposition

**Files:**
- Modify: `justatom/training/objective.py`
- Modify: `tests/test_training_objective.py`
- Modify: `tests/test_training_golden.py`

**Interfaces:**
- Produces: `ObjectiveOutput.retrieval_loss`, `auxiliary_loss`, and `head_loss` scalar tensors.
- Preserves: `primary_loss = retrieval_loss + auxiliary_loss + head_loss` and `loss = primary_loss + memory_loss` when memory exists.
- Consumes: existing detached alpha weighting and optional fixed SimCSE temperature.

- [ ] **Step 1: Write failing decomposition and gradient-isolation tests**

Add an atom-gate case asserting:

```python
expected_retrieval = output.main_per_row.mean()
expected_auxiliary = (
    0.1
    * (1.0 - torch.sigmoid(alpha_logits).detach())
    * output.simcse_per_row
).mean()
expected_head = 0.3 * output.alpha_supervision_per_row.mean()

torch.testing.assert_close(output.retrieval_loss, expected_retrieval)
torch.testing.assert_close(output.auxiliary_loss, expected_auxiliary)
torch.testing.assert_close(output.head_loss, expected_head)
torch.testing.assert_close(
    output.loss,
    output.retrieval_loss + output.auxiliary_loss + output.head_loss,
)
```

Use `torch.autograd.grad` to prove retrieval and auxiliary gradients reach
encoder representations, auxiliary does not reach `alpha_logits`, and head loss
reaches `alpha_logits` but not queries. Extend existing bank tests to preserve
the exact `loss = primary_loss + memory_loss` identity.

- [ ] **Step 2: Run the objective tests to verify RED**

```bash
conda run -n justatom pytest -q tests/test_training_objective.py tests/test_training_golden.py
```

Expected: missing `ObjectiveOutput` fields.

- [ ] **Step 3: Refactor composition without changing formulas**

Compute named scalar terms before their final sum:

```python
retrieval_loss = main.mean()
auxiliary_loss = weighted_encoder_auxiliary.mean()
head_loss = weighted_alpha_supervision.mean()
primary_loss = retrieval_loss + auxiliary_loss + head_loss
loss = primary_loss if memory_loss is None else primary_loss + memory_loss
```

Represent absent auxiliary and head terms as differentiable-device-compatible
zeros that do not invent optimizer gradients. Keep every existing metric and
per-row tensor contract intact.

- [ ] **Step 4: Run the objective tests to verify GREEN**

```bash
conda run -n justatom pytest -q tests/test_training_objective.py tests/test_training_golden.py
```

Expected: all tests pass and old golden formulas remain exact.

- [ ] **Step 5: Commit the decomposition**

```bash
git add justatom/training/objective.py tests/test_training_objective.py tests/test_training_golden.py
git commit -m "refactor(training): expose objective gradient terms"
```

### Task 4: Wire Manual Optimization and Gradient Accumulation

**Files:**
- Modify: `justatom/training/module.py`
- Modify: `justatom/training/job.py`
- Modify: `tests/test_training_module.py`
- Modify: `tests/test_training_job.py`

**Interfaces:**
- Consumes: `ObjectiveOutput` scalar decomposition and `control_auxiliary_gradients`.
- Produces: `_auxiliary_control_optimization_step(output, batch_idx) -> dict[str, float]`.
- Preserves: existing `_projected_optimization_step` for canonical bank-only `atomic`.

- [ ] **Step 1: Write failing module behavior tests**

Add tests that prove:

1. `off` keeps `automatic_optimization=True` for `atom_gate`.
2. `observe` and `safe` set `automatic_optimization=False`.
3. `build_lightning_trainer` uses `accumulate_grad_batches=1` for either manual path.
4. An aligned scalar toy objective applies the capped safe auxiliary gradient.
5. A conflicting toy objective applies only retrieval plus alpha-head gradients.
6. `observe` reproduces retrieval plus the unchanged auxiliary and head gradients.
7. Two microbatches accumulate their already-controlled gradients and step once.

Use the existing fake optimizer/manual-backward style in
`tests/test_training_module.py`; compare final `.grad` tensors directly instead
of checking only telemetry.

- [ ] **Step 2: Run module tests to verify RED**

```bash
conda run -n justatom pytest -q tests/test_training_module.py tests/test_training_job.py
```

Expected: manual mode and the new optimization step are absent.

- [ ] **Step 3: Implement the auxiliary-control optimization step**

Set manual mode when either the existing bank projection or the new auxiliary
mode is active:

```python
self.automatic_optimization = not (
    config.gradient_projection.enabled
    or config.auxiliary_gradient.mode is not AuxiliaryGradientMode.OFF
)
```

In `_auxiliary_control_optimization_step`:

```python
accumulated = capture_current_gradients()
zero_grad()
backward(retrieval_loss, retain_graph=True)
retrieval_gradients = capture_current_gradients()
zero_grad()
backward(auxiliary_loss)
auxiliary_gradients = capture_current_gradients()
zero_grad()
backward(head_loss)
head_gradients = capture_current_gradients()
zero_grad()

controlled, stats = control_auxiliary_gradients(
    retrieval_gradients,
    auxiliary_gradients,
    mode=config.auxiliary_gradient.mode,
    max_norm_ratio=config.auxiliary_gradient.max_norm_ratio,
    eps=config.auxiliary_gradient.eps,
)
parameter.grad = accumulated + retrieval + controlled + head
```

Guard zero/non-requires-grad scalar terms before backward. Clamp the learnable
retrieval temperature after each actual optimizer step exactly as existing paths
do. Branch explicitly between bank projection and auxiliary control in
`training_step`; method validation guarantees they cannot both be active.

Change `build_lightning_trainer` to select Lightning accumulation `1` whenever
either manual path is active; otherwise preserve configured accumulation.

- [ ] **Step 4: Run module tests to verify GREEN**

```bash
conda run -n justatom pytest -q tests/test_training_module.py tests/test_training_job.py
```

Expected: all manual-optimization, accumulation, and existing bank tests pass.

- [ ] **Step 5: Commit the training integration**

```bash
git add justatom/training/module.py justatom/training/job.py tests/test_training_module.py tests/test_training_job.py
git commit -m "feat(training): apply safe atom gate updates"
```

### Task 5: Record the Contract and Expose Reproducible Commands

**Files:**
- Modify: `justatom/training/job.py`
- Modify: `scripts/run_pipeline.sh`
- Modify: `scripts/run_benchmark.sh`
- Modify: `tests/test_training_job.py`
- Modify: `tests/test_benchmark_variants.py`
- Modify: `docs/training.md`
- Create: `configs/experiments/qwen3-06b-lora-alpha-gradient-safe.yaml`

**Interfaces:**
- Produces CLI flags: `--train-config`, `--aux-gradient-mode`, `--aux-gradient-max-norm-ratio`, and `--aux-gradient-eps`.
- Produces manifest fields `objective_contract.auxiliary_gradient` and `objective_contract.auxiliary_norm`.
- Consumes the dotted CLI keys under `auxiliary_gradient`.

- [ ] **Step 1: Write failing manifest and shell-forwarding tests**

Assert that an off run records:

```python
assert manifest.objective_contract["auxiliary_gradient"] == "off"
assert manifest.objective_contract["auxiliary_norm"] == "unbounded"
```

and a safe run records `cosine_safe` and `retrieval_relative`. Extend the shell
tests to invoke `run_benchmark.sh` with an experiment config and all three
gradient flags, then assert the generated
pipeline command contains:

```text
--train-config configs/experiments/qwen3-06b-lora-alpha-gradient-safe.yaml
--aux-gradient-mode safe
--aux-gradient-max-norm-ratio 0.25
--aux-gradient-eps 1e-12
```

- [ ] **Step 2: Run the tests to verify RED**

```bash
conda run -n justatom pytest -q tests/test_training_job.py tests/test_benchmark_variants.py
```

Expected: missing contract keys and unsupported shell options.

- [ ] **Step 3: Implement manifests, shell plumbing, config, and documentation**

Map the shell flags to:

```text
--auxiliary-gradient.mode
--auxiliary-gradient.max-norm-ratio
--auxiliary-gradient.eps
```

Make `run_pipeline.sh --train-config PATH` replace only the existing hard-coded
`--config configs/train.yaml` argument. Validate that the path exists before
starting a run. Make `run_benchmark.sh` forward the same path unchanged. All
existing pipeline defaults and explicit overrides continue to take precedence
over values supplied by the selected YAML.

Document the formula, first-order boundary, telemetry, and an `observe` command
in `docs/training.md`. Create the Qwen experiment config by reusing the existing
Qwen LoRA values and fixing:

```yaml
method: atom_gate
experiment:
  role: ablation
objective:
  decoupled: false
  temperature: 0.05
  simcse_temperature: 0.2
  simcse_dropout_weight: 0.03
alpha_gate:
  enabled: true
  supervision_weight: 0.3
  target_temperature: 0.2
auxiliary_gradient:
  mode: safe
  max_norm_ratio: 0.25
  eps: 1.0e-12
memory_bank:
  enabled: false
```

Copy model, LoRA, optimization, runtime, and artifact structure from the current
Qwen screening config rather than introducing new defaults.

- [ ] **Step 4: Run contract and script tests to verify GREEN**

```bash
conda run -n justatom pytest -q tests/test_training_job.py tests/test_benchmark_variants.py
bash -n scripts/run_pipeline.sh scripts/run_benchmark.sh
```

Expected: all tests pass and both shell scripts parse.

- [ ] **Step 5: Commit the public experiment surface**

```bash
git add justatom/training/job.py scripts/run_pipeline.sh scripts/run_benchmark.sh tests/test_training_job.py tests/test_benchmark_variants.py docs/training.md configs/experiments/qwen3-06b-lora-alpha-gradient-safe.yaml
git commit -m "docs(training): expose gradient-safe atom gate ablation"
```

### Task 6: Verify the Repository and Run the Preregistered Diagnostic

**Files:**
- Create after the run: `.tmp_runs/<run-id>/RESULTS.md` (ignored research artifact)
- Modify only if results justify it: `docs/research/atomic-experiments/README.md`

**Interfaces:**
- Consumes: the completed safe controller, CSV telemetry, fixed Qwen split, and existing evaluation pipeline.
- Produces: a diagnostic verdict before any multi-seed claim.

- [ ] **Step 1: Run the full relevant test suite**

```bash
conda run -n justatom pytest -q \
  tests/test_auxiliary_gradient.py \
  tests/test_training_config.py \
  tests/test_training_methods.py \
  tests/test_training_objective.py \
  tests/test_training_golden.py \
  tests/test_training_module.py \
  tests/test_training_job.py \
  tests/test_benchmark_variants.py \
  tests/test_train_cli.py \
  tests/test_scenario_configs.py
```

Expected: all tests pass.

- [ ] **Step 2: Run static and complete verification**

```bash
conda run -n justatom python -m compileall -q justatom
conda run -n justatom pytest -q
```

Expected: compilation succeeds and the full suite passes.

- [ ] **Step 3: Run one observe-mode Qwen diagnostic**

Use the same frozen `justatom` sample, seed, LoRA configuration, microbatch,
accumulation, temperatures, and epoch count as the two-seed alpha confirmation:

```bash
bash scripts/run_pipeline.sh \
  --train-config configs/experiments/qwen3-06b-lora-alpha-gradient-safe.yaml \
  --method atom_gate \
  --experiment-role ablation \
  --dataset-ids justatom \
  --model Qwen/Qwen3-Embedding-0.6B \
  --batch-size 8 \
  --grad-acc-steps 4 \
  --epochs 1 \
  --nsamples 3000 \
  --temperature 0.05 \
  --aux-gradient-mode observe \
  --aux-gradient-max-norm-ratio 0.25 \
  --wandb-mode disabled
```

Before accepting the run, inspect `run_manifest.yaml` and verify that the
selected YAML resolved Qwen LoRA plus `tau_target=0.2`, `tau_simcse=0.2`, and
`lambda_sc=0.03`. Stop the run analysis if any of those values differ.

- [ ] **Step 4: Write the diagnostic verdict**

From `batch_metrics.csv`, report mean, median, p05, and p95 for cosine and norm
ratio, plus compatibility rate. Break cosine down by alpha-target quartile. The
verdict must state one of:

```text
ACTIVE: incompatible or weakly compatible steps are frequent enough to test safe mode.
INERT: compatibility is almost always strongly positive; stop before a safe retraining run.
INVALID: non-finite or missing telemetry; fix instrumentation before drawing a conclusion.
```

- [ ] **Step 5: Run safe mode only when the diagnostic is ACTIVE**

Repeat the exact diagnostic command with:

```text
--aux-gradient-mode safe
```

Compare against the already matched vanilla and current atom-gate artifacts.
Do not add a second seed or mMARCO until the first safe run has finite telemetry
and does not regress both HR@1 and MRR@10.

- [ ] **Step 6: Commit only durable research documentation**

If the diagnostic or safe run changes the research decision:

```bash
git add docs/research/atomic-experiments/README.md
git commit -m "docs(research): record safe auxiliary diagnostic"
```

Do not commit model weights, `.tmp_runs`, generated indexes, or evaluation CSVs.
