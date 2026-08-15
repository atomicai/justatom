# Count-Normalized Memory Bank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make bank pressure independent of active candidate count, ramp it smoothly, and compare ordinary versus projected normalized-bank gradients.

**Architecture:** `ContrastiveMemoryBank` adds `log(lambda(t)) + log(N - 1) - log(K_i)` to each active bank logit after selection. Candidate admission log weights compose additively. `ContrastiveObjective` keeps `L_memory = L_augmented - L_main`, preserving exact ordinary and projected controls.

**Tech Stack:** Python 3.10-3.13, PyTorch, PyTorch Lightning, pytest, YAML, Bash, Weaviate evaluation.

## Global Constraints

- Execute after `2026-08-14-difficulty-supervised-alpha.md`.
- Follow `docs/superpowers/specs/2026-08-14-difficulty-alpha-normalized-bank-design.md`.
- Canonical `mass_ratio=0.5`; canonical `mass_ramp_steps=20` optimizer steps.
- Count `K_i` after identity masks and selection.
- Before warmup, and whenever effective ratio is zero, return a no-op selection.
- Rows with `K_i == 0` remain identical to no-bank rows.
- Keep masked logits and log weights finite on MPS.
- Keep adaptive mining and `m(q)` disabled in screening.
- Record contrastive microbatch separately from optimizer effective batch.

---

## File Map

| File | Responsibility |
| --- | --- |
| `justatom/training/config.py` | Mass schema and validation |
| `justatom/training/methods.py` | Canonical values |
| `justatom/training/memory_bank.py` | Ramp, normalization, telemetry |
| `justatom/training/objective.py` | Exact decomposition integration |
| `justatom/training/job.py` | Manifest provenance |
| `scripts/run_pipeline.sh` | Component overrides |
| `scripts/run_benchmark.sh` | Override passthrough |
| `configs/experiments/qwen3-06b-lora-vanilla-bank.yaml` | Control recipe |
| `tests/test_memory_bank.py` | Closed-form mass tests |
| `tests/test_training_objective.py` | Count invariance |
| `tests/test_training_module.py` | Manual optimization regression |
| `tests/test_training_config.py` | Schema validation |
| `tests/test_training_methods.py` | Canonical contract |
| `tests/test_training_job.py` | Manifest contract |
| `tests/test_benchmark_variants.py` | Shell forwarding |
| `tests/test_train_cli.py` | Recipe resolution |
| `docs/training.md` | Public formula |

### Task 1: Add memory-mass configuration

**Files:**
- Modify: `justatom/training/config.py`
- Modify: `justatom/training/methods.py`
- Modify: `configs/experiments/qwen3-06b-lora-vanilla-bank.yaml`
- Test: `tests/test_training_config.py`
- Test: `tests/test_training_methods.py`
- Test: `tests/test_train_cli.py`

**Interfaces:**
- Produces: `MemoryBankConfig.mass_ratio: float = 0.5`.
- Produces: `MemoryBankConfig.mass_ramp_steps: int = 20`.

- [ ] **Step 1: Write failing schema tests**

```python
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_memory_mass_ratio_is_finite_and_non_negative(value):
    with pytest.raises(ValueError, match=r"memory_bank\.mass_ratio"):
        parse_train_config({"method": "atomic", "memory_bank": {"mass_ratio": value}})


def test_memory_mass_ramp_is_positive():
    with pytest.raises(ValueError, match=r"memory_bank\.mass_ramp_steps"):
        parse_train_config({"method": "atomic", "memory_bank": {"mass_ramp_steps": 0}})
```

Extend canonical and recipe assertions:

```python
assert atomic.memory_bank.mass_ratio == pytest.approx(0.5)
assert atomic.memory_bank.mass_ramp_steps == 20
assert recipe.memory_bank.mass_ratio == pytest.approx(0.5)
assert recipe.memory_bank.mass_ramp_steps == 20
```

- [ ] **Step 2: Verify failure**

```bash
python -m pytest tests/test_training_config.py tests/test_training_methods.py tests/test_train_cli.py -q
```

Expected: FAIL on absent fields.

- [ ] **Step 3: Implement schema and validation**

Add to `MemoryBankConfig`:

```python
mass_ratio: float = 0.5
mass_ramp_steps: int = 20
```

Reuse the finite numeric validator introduced by the alpha plan and validate:

```python
_require_number(bank.mass_ratio, "memory_bank.mass_ratio", 0.0)
_require_int(bank.mass_ramp_steps, "memory_bank.mass_ramp_steps", 1)
```

Set both canonical values explicitly in `canonical_method_config`; add them to the Qwen bank YAML.

- [ ] **Step 4: Verify pass**

```bash
python -m pytest tests/test_training_config.py tests/test_training_methods.py tests/test_train_cli.py -q
ruff check justatom/training/config.py justatom/training/methods.py tests/test_training_config.py tests/test_training_methods.py tests/test_train_cli.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add justatom/training/config.py justatom/training/methods.py configs/experiments/qwen3-06b-lora-vanilla-bank.yaml tests/test_training_config.py tests/test_training_methods.py tests/test_train_cli.py
git commit -m "feat(training): configure normalized memory mass"
```

### Task 2: Normalize selected bank weights

**Files:**
- Modify: `justatom/training/memory_bank.py`
- Test: `tests/test_memory_bank.py`

**Interfaces:**
- Produces: `_mass_progress(step: int) -> float`.
- Produces: `_effective_mass_ratio(step: int) -> float`.
- Produces: `_normalized_log_weights(active, candidate_log_weights, step) -> (Tensor, metrics)`.

- [ ] **Step 1: Write failing ramp and weight tests**

```python
def test_memory_mass_ramps_after_warmup():
    bank = ContrastiveMemoryBank(
        MemoryBankConfig(
            enabled=True,
            size=32,
            warmup_steps=50,
            mass_ratio=0.5,
            mass_ramp_steps=20,
        )
    )
    assert bank._mass_progress(49) == pytest.approx(0.0)
    assert bank._mass_progress(50) == pytest.approx(0.05)
    assert bank._mass_progress(59) == pytest.approx(0.5)
    assert bank._mass_progress(69) == pytest.approx(1.0)


def test_n8_k12_candidate_weight_is_seven_over_twenty_four():
    bank = ContrastiveMemoryBank(
        MemoryBankConfig(
            enabled=True,
            size=32,
            mass_ratio=0.5,
            mass_ramp_steps=1,
        )
    )
    active = torch.ones(8, 12, dtype=torch.bool)
    log_weights, metrics = bank._normalized_log_weights(active, None, step=0)
    torch.testing.assert_close(log_weights.exp(), torch.full((8, 12), 7.0 / 24.0))
    assert metrics["memory/effective_mass_ratio"] == pytest.approx(0.5)


def test_candidate_weights_compose_and_empty_rows_stay_finite():
    bank = ContrastiveMemoryBank(
        MemoryBankConfig(enabled=True, size=8, mass_ratio=0.5, mass_ramp_steps=1)
    )
    active = torch.tensor([[True, True, False], [False, False, False]])
    candidate = torch.log(torch.tensor([[0.5, 0.25, 1.0], [1.0, 1.0, 1.0]]))
    log_weights, _ = bank._normalized_log_weights(active, candidate, step=0)
    torch.testing.assert_close(log_weights[0, :2].exp(), torch.tensor([0.125, 0.0625]))
    assert torch.isfinite(log_weights).all()
    assert log_weights[1].eq(0.0).all()


def test_normalization_rejects_single_row_contrastive_batch():
    bank = ContrastiveMemoryBank(
        MemoryBankConfig(enabled=True, size=8, mass_ratio=0.5, mass_ramp_steps=1)
    )
    with pytest.raises(ValueError, match="contrastive batch size >= 2"):
        bank._normalized_log_weights(torch.ones(1, 2, dtype=torch.bool), None, step=0)


def test_metric_columns_are_stable_across_mass_warmup():
    bank = ContrastiveMemoryBank(
        MemoryBankConfig(
            enabled=True,
            size=8,
            warmup_steps=1,
            mining="random",
            random_negatives=2,
            mass_ratio=0.5,
            mass_ramp_steps=1,
        )
    )
    vectors = F.normalize(torch.randn(4, 3), dim=-1)
    bank.enqueue(vectors, {"doc_key_id": torch.arange(4)})
    batch = {"doc_key_id": torch.tensor([10, 11])}
    before = bank.select(batch, query_vectors=vectors[:2], positive_vectors=vectors[:2], step=0)
    after = bank.select(batch, query_vectors=vectors[:2], positive_vectors=vectors[:2], step=1)
    assert set(before.metrics) == set(after.metrics)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_normalized_weights_are_finite_on_mps():
    bank = ContrastiveMemoryBank(
        MemoryBankConfig(enabled=True, size=8, mass_ratio=0.5, mass_ramp_steps=1)
    )
    active = torch.tensor([[True, False], [False, False]], device="mps")
    log_weights, _ = bank._normalized_log_weights(active, None, step=0)
    assert torch.isfinite(log_weights).all()


def test_zero_mass_ratio_returns_noop_selection():
    bank = ContrastiveMemoryBank(
        MemoryBankConfig(
            enabled=True,
            size=8,
            warmup_steps=0,
            mining="random",
            random_negatives=2,
            mass_ratio=0.0,
            mass_ramp_steps=1,
        )
    )
    vectors = F.normalize(torch.randn(4, 3), dim=-1)
    bank.enqueue(vectors, {"doc_key_id": torch.arange(4)})
    selection = bank.select(
        {"doc_key_id": torch.tensor([10, 11])},
        query_vectors=vectors[:2],
        positive_vectors=vectors[:2],
        step=0,
    )
    assert selection.embeddings is None
    assert selection.active_mask is None
    assert selection.log_weights is None
```

- [ ] **Step 2: Verify failure**

```bash
python -m pytest tests/test_memory_bank.py -q
```

Expected: FAIL on missing helpers.

- [ ] **Step 3: Implement ramp and finite log weights**

```python
def _mass_progress(self, step: int) -> float:
    if step < self.config.warmup_steps:
        return 0.0
    offset = int(step) - self.config.warmup_steps + 1
    return min(max(offset / float(self.config.mass_ramp_steps), 0.0), 1.0)


def _effective_mass_ratio(self, step: int) -> float:
    return float(self.config.mass_ratio) * self._mass_progress(step)
```

At the beginning of `select`, return `_noop_selection()` when the effective ratio is zero. After selection, calculate:

```python
batch_size = int(active.shape[0])
if batch_size < 2:
    raise ValueError("memory bank requires contrastive batch size >= 2")
counts = active.sum(dim=1)
safe_counts = counts.clamp_min(1).float()
ratio = self._effective_mass_ratio(step)
row_log_weight = math.log(ratio) + math.log(batch_size - 1) - safe_counts.log()
normalized = torch.where(
    active,
    row_log_weight.view(-1, 1).expand_as(active),
    torch.zeros(active.shape, device=active.device, dtype=row_log_weight.dtype),
)
if candidate_log_weights is not None:
    normalized = normalized + torch.where(
        active,
        candidate_log_weights.to(normalized),
        torch.zeros_like(normalized),
    )
```

Convert to query dtype before `MemorySelection`. Emit stable `memory/mass_ratio`, `memory/mass_ramp`, `memory/effective_mass_ratio`, `memory/active_count/*`, and `memory/candidate_mass_weight/*` metrics, including zero/NaN distribution fields in `_base_metrics` before warmup.

- [ ] **Step 4: Verify pass and finiteness**

```bash
python -m pytest tests/test_memory_bank.py -q
ruff check justatom/training/memory_bank.py tests/test_memory_bank.py
git diff --check
```

Expected: PASS; no `NaN` or infinity in returned tensors.

- [ ] **Step 5: Commit**

```bash
git add justatom/training/memory_bank.py tests/test_memory_bank.py
git commit -m "feat(training): normalize memory mass by candidate count"
```

### Task 3: Prove denominator invariance and decomposition

**Files:**
- Modify: `tests/test_training_objective.py`
- Modify: `tests/test_training_module.py`
- Modify only on exposed defect: `justatom/training/loss.py`, `justatom/training/objective.py`

**Interfaces:**
- Verifies: duplicate equal-score bank candidates do not change normalized loss.
- Verifies: `loss == primary_loss + memory_loss`.

- [ ] **Step 1: Add count-invariance test**

```python
def test_normalized_bank_loss_is_invariant_to_duplicate_count():
    objective = ContrastiveObjective(
        ObjectiveConfig(temperature=1.0, learnable_temperature=False, decoupled=False)
    )
    q = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    p = q.clone()
    vector = F.normalize(torch.tensor([[1.0, 1.0]]), dim=-1)

    def output_for(count):
        weight = 0.5 * (q.shape[0] - 1) / count
        memory = MemorySelection(
            embeddings=vector.repeat(count, 1),
            active_mask=torch.ones(2, count, dtype=torch.bool),
            log_weights=torch.full((2, count), math.log(weight)),
            collision_g=None,
            hard_weights=None,
            metrics={},
        )
        return objective(ObjectiveInputs(queries=q, positives=p, memory=memory))

    one, four = output_for(1), output_for(4)
    torch.testing.assert_close(one.loss, four.loss)
    torch.testing.assert_close(one.memory_per_row, four.memory_per_row)


def test_row_without_valid_bank_candidates_equals_main_loss():
    objective = ContrastiveObjective(
        ObjectiveConfig(temperature=1.0, learnable_temperature=False, decoupled=False)
    )
    q = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    p = q.clone()
    memory = MemorySelection(
        embeddings=F.normalize(torch.tensor([[1.0, 1.0]]), dim=-1),
        active_mask=torch.zeros(2, 1, dtype=torch.bool),
        log_weights=torch.zeros(2, 1),
        collision_g=None,
        hard_weights=None,
        metrics={},
    )
    plain = objective(ObjectiveInputs(queries=q, positives=p))
    empty = objective(ObjectiveInputs(queries=q, positives=p, memory=memory))
    torch.testing.assert_close(empty.loss, plain.loss)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_normalized_bank_objective_gradients_are_finite_on_mps():
    objective = ContrastiveObjective(
        ObjectiveConfig(temperature=0.05, learnable_temperature=True, decoupled=False)
    ).to("mps")
    q = F.normalize(torch.randn(2, 4, device="mps"), dim=-1).requires_grad_()
    p = F.normalize(torch.randn(2, 4, device="mps"), dim=-1).requires_grad_()
    memory = MemorySelection(
        embeddings=F.normalize(torch.randn(3, 4, device="mps"), dim=-1),
        active_mask=torch.tensor([[True, True, False], [True, False, True]], device="mps"),
        log_weights=torch.full((2, 3), math.log(0.25), device="mps"),
        collision_g=None,
        hard_weights=None,
        metrics={},
    )
    output = objective(ObjectiveInputs(queries=q, positives=p, memory=memory))
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert p.grad is not None and torch.isfinite(p.grad).all()
```

In existing bank tests, retain:

```python
torch.testing.assert_close(output.loss, output.primary_loss + output.memory_loss)
assert torch.isfinite(output.loss)
```

- [ ] **Step 2: Run objective and accumulation tests**

```bash
python -m pytest tests/test_training_objective.py tests/test_training_module.py::test_lightning_atomic_manual_optimization_steps_with_live_bank tests/test_training_module.py::test_lightning_atomic_handles_gradient_accumulation_and_final_partial_step -q
```

Expected: PASS. A failure indicates normalization was applied in the wrong logit space.

- [ ] **Step 3: Verify kernel ordering**

Required order:

```python
memory_logits = memory_logits / tau_scale
memory_logits = memory_logits + memory_soft_log_weight
```

Do not temperature-scale the logarithm of a multiplicative mass weight. Edit the kernel only if Step 2 proves the order differs.

- [ ] **Step 4: Run all bank/projection tests**

```bash
python -m pytest tests/test_memory_bank.py tests/test_training_objective.py tests/test_gradient_projection.py tests/test_training_module.py -q
```

Expected: PASS with unchanged projection equations.

- [ ] **Step 5: Commit invariant coverage**

```bash
git add tests/test_training_objective.py tests/test_training_module.py
git add -u justatom/training/loss.py justatom/training/objective.py
git commit -m "test(training): prove normalized memory decomposition"
```

### Task 4: Expose controls and provenance

**Files:**
- Modify: `scripts/run_pipeline.sh`
- Modify: `scripts/run_benchmark.sh`
- Modify: `justatom/training/job.py`
- Modify: `tests/test_benchmark_variants.py`
- Modify: `tests/test_training_job.py`
- Modify: `docs/training.md`

**Interfaces:**
- Produces flags: `--memory-bank-mass-ratio`, `--memory-bank-mass-ramp-steps`.
- Produces manifest value: `memory_mass=count_normalized` for enabled bank.
- Produces manifest `batch_contract` with microbatch, accumulation, and effective optimizer batch.

- [ ] **Step 1: Write failing shell/manifest tests**

Pass both flags through benchmark dry-run and assert:

```python
assert "--memory-bank-mass-ratio 0.5" in commands
assert "--memory-bank-mass-ramp-steps 20" in commands
assert loaded["objective_contract"]["memory_mass"] == "count_normalized"
assert loaded["resolved_config"]["memory_bank"]["mass_ratio"] == pytest.approx(0.5)
assert loaded["batch_contract"] == {
    "contrastive_microbatch": 8,
    "gradient_accumulation": 4,
    "optimizer_effective_batch": 32,
}
```

- [ ] **Step 2: Verify failure**

```bash
python -m pytest tests/test_benchmark_variants.py tests/test_training_job.py -q
```

Expected: FAIL on unknown flags and missing manifest key.

- [ ] **Step 3: Implement shell mapping and manifest key**

Map in `run_pipeline.sh`:

```bash
--memory-bank-mass-ratio) EXPLICIT_OVERRIDES+=(--memory-bank.mass-ratio "$2"); shift 2 ;;
--memory-bank-mass-ramp-steps) EXPLICIT_OVERRIDES+=(--memory-bank.mass-ramp-steps "$2"); shift 2 ;;
```

Add both flags to `run_benchmark.sh` passthrough. Add to `objective_contract`:

```python
contract["memory_mass"] = (
    "count_normalized" if config.memory_bank.enabled else "not_applicable"
)
```

Add `batch_contract: dict[str, int]` to `RunManifest`, include it in
`to_dict`, and populate it in `from_config`:

```python
batch_contract={
    "contrastive_microbatch": config.optimization.batch_size,
    "gradient_accumulation": config.optimization.grad_acc_steps,
    "optimizer_effective_batch": (
        config.optimization.batch_size * config.optimization.grad_acc_steps
    ),
},
```

- [ ] **Step 4: Document and verify**

Document:

```text
L_aug,i = -z_ii + log(
  exp(z_ii) + A_batch,i + lambda(t) (N - 1) / K_i A_bank,i
)
```

Run:

```bash
bash -n scripts/run_pipeline.sh
bash -n scripts/run_benchmark.sh
python -m pytest tests/test_benchmark_variants.py tests/test_training_job.py -q
ruff check justatom/training/job.py tests/test_training_job.py
git diff --check
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_pipeline.sh scripts/run_benchmark.sh justatom/training/job.py tests/test_benchmark_variants.py tests/test_training_job.py docs/training.md
git commit -m "docs(training): expose normalized memory contract"
```

### Task 5: Verify before experiments

**Files:**
- Verify only; repair preceding files only after an observed failure.

**Interfaces:**
- Produces: clean implementation commit.

- [ ] **Step 1: Run focused training tests**

```bash
python -m pytest tests/test_training_config.py tests/test_training_methods.py tests/test_memory_bank.py tests/test_training_objective.py tests/test_gradient_projection.py tests/test_training_module.py tests/test_training_job.py tests/test_train_cli.py tests/test_benchmark_variants.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full offline verification**

```bash
python -m pytest tests -m "not integration and not network" -q
ruff check justatom tests
bash -n scripts/run_pipeline.sh
bash -n scripts/run_benchmark.sh
git diff --check
```

Expected: PASS.

- [ ] **Step 3: Run focused MPS checks**

```bash
python -m pytest tests/test_memory_bank.py::test_normalized_weights_are_finite_on_mps tests/test_training_objective.py::test_normalized_bank_objective_gradients_are_finite_on_mps tests/test_training_module.py::test_lightning_atomic_manual_optimization_steps_with_live_bank -q
```

Run under `conda activate justatom`; expected: PASS, or both MPS-specific tests SKIP only when MPS is unavailable.

- [ ] **Step 4: Record commit and clean state**

```bash
git status --short
git rev-parse HEAD
```

Expected: clean worktree and one implementation hash.

- [ ] **Step 5: Commit only a verified repair**

If tracked files changed due to a verified defect, commit those exact files as `fix(training): complete normalized bank verification`. Otherwise create no commit.

### Task 6: Run matched Qwen screening

**Files:**
- Create locally: `.tmp_runs/difficulty_alpha_bank_${STAMP}/run_matrix.sh`
- Produce locally: `EXPERIMENT.md`, `RESULTS.md`, per-run manifests, metrics, encoders, and eval CSVs.

**Interfaces:**
- Produces: `coupled`, `difficulty_alpha`, `normalized_bank`, `normalized_bank_projected`.

- [ ] **Step 1: Record fixed conditions**

```bash
STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT=".tmp_runs/difficulty_alpha_bank_${STAMP}"
mkdir -p "$ROOT/runs" "$ROOT/eval"
git status --short
git rev-parse HEAD | tee "$ROOT/IMPLEMENTATION_COMMIT"
```

Create `EXPERIMENT.md` with these exact fixed conditions:

```markdown
# Difficulty Alpha and Normalized Bank Screening

- Model: `Qwen/Qwen3-Embedding-0.6B`
- Dataset: `justatom`
- Training pairs: `3000`
- Seed: `42`
- Epochs: `1`
- Contrastive microbatch/matrix: `8` / `8 x 8`
- Gradient accumulation: `4`
- Effective optimizer batch: `32`
- Temperature: `0.05`, learnable
- Adaptation: LoRA rank `16`, alpha `32`, dropout `0.05`, RS-LoRA, `all-linear`
- Bank: FIFO `512`, warmup `50`, random `12`, mass ratio `0.5`, ramp `20`
- Primary metrics: `HitRate@1`, `mrr@10`
```

- [ ] **Step 2: Build the local runner with exact variant overrides**

Create `run_matrix.sh` with this body and mark it executable:

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${ROOT:?ROOT must point to the screening directory}"
SEED="${SEED:-42}"
BASE_CONFIG="configs/experiments/qwen3-06b-lora-vanilla-bank.yaml"
QUERY_PREFIX=$'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'
STAMP_TOKEN="$(basename "$ROOT" | tr -cd '[:alnum:]')"

common=(
  --config "$BASE_CONFIG"
  --dataset.id justatom
  --dataset.limit 3000
  --experiment.seed "$SEED"
  --optimization.num-samples 3000
  --optimization.batch-size 8
  --optimization.grad-acc-steps 4
  --optimization.epochs 1
  --objective.temperature 0.05
  --objective.learnable-temperature true
  --telemetry.backend csv
)

bank=(
  --memory-bank.enabled true
  --memory-bank.size 512
  --memory-bank.warmup-steps 50
  --memory-bank.mining random
  --memory-bank.random-negatives 12
  --memory-bank.hard-negatives 0
  --memory-bank.mass-ratio 0.5
  --memory-bank.mass-ramp-steps 20
  --memory-bank.adaptive.enabled false
  --memory-bank.margin.mode off
  --memory-bank.margin.regularization-weight 0.0
)

train_variant() {
  local name="$1" method="$2" role="$3"
  shift 3
  python -m justatom.api.train \
    --method "$method" \
    "${common[@]}" \
    --experiment.role "$role" \
    --artifacts.save-dir "$ROOT/runs/$name" \
    --artifacts.collection-name "Justatom${STAMP_TOKEN}${name//_/}" \
    --telemetry.metrics-path "$ROOT/runs/$name/batch_metrics.csv" \
    "$@"
}

eval_variant() {
  local name="$1"
  python -m justatom.api.eval \
    --config configs/evaluate.yaml \
    --embedding-backend local \
    --embedding-model "$ROOT/runs/$name/encoder" \
    --query-prefix "$QUERY_PREFIX" \
    --document-prefix "" \
    --dataset.id justatom \
    --collection-name "JustatomEval${STAMP_TOKEN}${name//_/}" \
    --save-results-to-dir "$ROOT/eval/$name" \
    --flush-collection \
    --search-mode vector \
    --top-k 20 \
    --index-batch-size 4 \
    --search-batch-size 64 \
    --weaviate-url http://127.0.0.1:2211 \
    --weaviate-grpc-port 50051
}

train_variant coupled vanilla canonical \
  --alpha-gate.enabled false \
  --memory-bank.enabled false \
  --gradient-projection.enabled false \
  --objective.decoupled false \
  --objective.simcse-dropout-weight 0.0
eval_variant coupled

train_variant difficulty_alpha atom_gate canonical \
  --alpha-gate.enabled true \
  --alpha-gate.supervision-weight 0.3 \
  --objective.decoupled false \
  --objective.simcse-dropout-weight 0.1 \
  --memory-bank.enabled false \
  --gradient-projection.enabled false
eval_variant difficulty_alpha

train_variant normalized_bank vanilla ablation \
  --alpha-gate.enabled false \
  --objective.decoupled false \
  --objective.simcse-dropout-weight 0.0 \
  --gradient-projection.enabled false \
  "${bank[@]}"
eval_variant normalized_bank

train_variant normalized_bank_projected atomic canonical \
  --alpha-gate.enabled false \
  --objective.decoupled false \
  --objective.simcse-dropout-weight 0.0 \
  --gradient-projection.enabled true \
  --gradient-projection.memory-weight 1.0 \
  "${bank[@]}"
eval_variant normalized_bank_projected
```

- [ ] **Step 3: Validate preconditions**

```bash
bash -n "$ROOT/run_matrix.sh"
curl -fsS http://127.0.0.1:2211/v1/.well-known/ready >/dev/null
for v in coupled difficulty_alpha normalized_bank normalized_bank_projected; do test ! -e "$ROOT/runs/$v/run_manifest.yaml"; done
```

Expected: valid shell, live Weaviate, no reused runs.

- [ ] **Step 4: Execute and verify artifacts**

```bash
ROOT="$ROOT" "$ROOT/run_matrix.sh" 2>&1 | tee "$ROOT/matrix.log"
for v in coupled difficulty_alpha normalized_bank normalized_bank_projected; do
  test -f "$ROOT/runs/$v/run_manifest.yaml"
  test -f "$ROOT/runs/$v/batch_metrics.csv"
  test -d "$ROOT/runs/$v/encoder"
  find "$ROOT/eval/$v" -name '*.csv' -type f | grep -q .
done
```

Expected: all stages exit zero. Generate `RESULTS.md` by metric name:

```bash
python - "$ROOT" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
variants = ("coupled", "difficulty_alpha", "normalized_bank", "normalized_bank_projected")
wanted = ("HitRate@1", "HitRate@5", "HitRate@10", "mrr@10", "map@10", "ndcg@10")
values = {}
for variant in variants:
    paths = list((root / "eval" / variant).glob("*.csv"))
    if len(paths) != 1:
        raise SystemExit(f"expected one evaluation CSV for {variant}, got {paths}")
    with paths[0].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    metrics = {row[0]: float(row[1]) for row in rows if len(row) >= 2 and row[0] in wanted}
    missing = set(wanted) - set(metrics)
    if missing:
        raise SystemExit(f"missing metrics for {variant}: {sorted(missing)}")
    values[variant] = metrics

with (root / "RESULTS.md").open("w", encoding="utf-8") as handle:
    handle.write("# Screening Results\n\n")
    handle.write("| Variant | " + " | ".join(wanted) + " | delta HR@1 | delta MRR@10 |\n")
    handle.write("| --- | " + " | ".join("---:" for _ in wanted) + " | ---: | ---: |\n")
    baseline = values["coupled"]
    for variant in variants:
        row = values[variant]
        delta_hr = row["HitRate@1"] - baseline["HitRate@1"]
        delta_mrr = row["mrr@10"] - baseline["mrr@10"]
        rendered = " | ".join(f"{row[name]:.6f}" for name in wanted)
        handle.write(f"| `{variant}` | {rendered} | {delta_hr:+.6f} | {delta_mrr:+.6f} |\n")
PY
```

- [ ] **Step 5: Apply screening rule**

Compute `difficulty_alpha - coupled`, `normalized_bank - coupled`, `normalized_bank_projected - normalized_bank`, and `normalized_bank_projected - coupled`.

A mechanism passes only when HR@1 and MRR@10 are both at least baseline and one is higher. Summarize alpha target/error/auxiliary weight, bank effective ratio/count/memory loss, and projection conflict/cosine. Passing mechanisms repeat at seed 43; failing mechanisms remain negative ablations and are not combined.
