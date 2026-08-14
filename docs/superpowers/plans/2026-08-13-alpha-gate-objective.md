# Alpha-Gate Objective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all canonical methods use coupled InfoNCE and prevent the SimCSE auxiliary path from training `alpha(q)` toward the loss-disabling solution.

**Architecture:** Method resolution owns the shared coupled-InfoNCE contract. `ContrastiveObjective` uses live alpha for semantic/lexical pair supervision and `alpha.detach()` only for SimCSE weighting; `ContrastiveTrainingModule` publishes the resulting effective weight distribution. `RunManifest` records this static objective contract alongside the resolved configuration.

**Tech Stack:** Python 3.10-3.13, PyTorch, PyTorch Lightning, pytest, dataclasses, YAML.

## Global Constraints

- Public method names remain exactly `vanilla`, `atom_gate`, and `atomic`.
- All canonical methods resolve to `objective.decoupled: false`.
- DCL remains available only when `experiment.role: ablation` is explicit.
- Detach only the alpha value used to weight SimCSE; keep alpha live in pair mixing and entropy regularization.
- Introduce no new public hyperparameters, heads, budget losses, or changes to ATOMIC memory projection.
- Existing untracked research files and scripts are outside this change and must remain untouched.

---

### Task 1: Canonical Coupled-InfoNCE Contract

**Files:**
- Modify: `tests/test_training_methods.py`
- Modify: `justatom/training/methods.py`

**Interfaces:**
- Consumes: `canonical_method_config(method) -> TrainConfig` and `resolve_method(config) -> TrainConfig`.
- Produces: a canonical method invariant where `config.objective.decoupled is False`; DCL is accepted only for `ExperimentRole.ABLATION`.

- [x] **Step 1: Write failing method-contract tests**

Extend the existing structural profile test with:

```python
assert not vanilla.objective.decoupled
assert not gate.objective.decoupled
assert not atomic.objective.decoupled
```

Add a parameterized behavior test:

```python
@pytest.mark.parametrize("method", list(TrainingMethod))
def test_canonical_methods_reject_dcl_but_ablation_allows_it(method):
    config = canonical_method_config(method)
    dcl = replace(config, objective=replace(config.objective, decoupled=True))

    with pytest.raises(ValueError, match="coupled InfoNCE.*experiment.role=ablation"):
        resolve_method(dcl)

    ablation = replace(dcl, experiment=ExperimentConfig(role=ExperimentRole.ABLATION, seed=42))
    assert resolve_method(ablation).objective.decoupled
```

The production mutation caught here is either restoring DCL as a canonical default or allowing a canonical override to silently confound benchmark methods.

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```bash
conda run -n justatom pytest -q tests/test_training_methods.py
```

Expected: failure because canonical `vanilla` and `atom_gate` still resolve to DCL, and their canonical DCL configurations are not rejected.

- [x] **Step 3: Implement the shared method invariant**

In `canonical_method_config`, construct a coupled base objective and reuse it:

```python
objective = ObjectiveConfig(decoupled=False)
if method is TrainingMethod.VANILLA:
    return TrainConfig(method=method, objective=objective)

atom_gate_objective = replace(objective, simcse_dropout_weight=0.1)
```

Before method-specific structural validation in `resolve_method`, enforce:

```python
if role is ExperimentRole.CANONICAL and config.objective.decoupled:
    raise ValueError(
        f"canonical {method.value} requires coupled InfoNCE; "
        "use experiment.role=ablation for DCL"
    )
```

Remove the now-duplicated ATOMIC-only DCL rejection.

- [x] **Step 4: Run focused configuration tests and verify GREEN**

Run:

```bash
conda run -n justatom pytest -q \
  tests/test_training_methods.py \
  tests/test_training_config.py \
  tests/test_train_cli.py
```

Expected: all tests pass.

- [x] **Step 5: Commit the method contract**

```bash
git add tests/test_training_methods.py justatom/training/methods.py
git commit -m "fix(training): align canonical methods on InfoNCE"
```

---

### Task 2: Detached Alpha Auxiliary Control

**Files:**
- Modify: `tests/test_training_objective.py`
- Modify: `tests/test_training_module.py`
- Modify: `justatom/training/objective.py`
- Modify: `justatom/training/module.py`

**Interfaces:**
- Consumes: `ObjectiveInputs.alpha: Tensor[B]`, `ObjectiveInputs.query_alt`, and existing semantic/lexical pair score tensors.
- Produces: SimCSE coefficient `1 - alpha.detach()` and telemetry keys `alpha_aux_weight/{mean,std,min,p05,p50,p95,max}`.

- [x] **Step 1: Write the failing auxiliary-gradient test**

Replace the old alpha-gradient assertion in the augment-formula test with a focused contract:

```python
auxiliary_only = output.loss - output.main_per_row.mean()
auxiliary_only.backward()

assert alpha.grad is None
assert queries.grad is not None and float(queries.grad.abs().sum()) > 0.0
assert alternate_queries.grad is not None and float(alternate_queries.grad.abs().sum()) > 0.0
```

Retain the hand-derived forward assertion:

```python
expected = (
    output.main_per_row
    + (1.0 - alpha.detach()) * 0.1 * output.simcse_per_row
).mean()
torch.testing.assert_close(output.loss, expected)
```

Use two otherwise identical calls with alpha values `0.2` and `0.8` to assert that the detached control still changes the forward loss.

The mutation caught is removing `.detach()`, which would restore the shortcut gradient into the alpha head.

- [x] **Step 2: Write the failing pair-supervision gradient test**

Use literal semantic and lexical score pairs with opposite preferences:

```python
semantic = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
lexical = torch.tensor([[0.1, 0.8], [0.9, 0.2]])
alpha = torch.tensor([0.4, 0.6], requires_grad=True)
```

Pass both score matrices and `alpha_mix_weight=0.3`, backpropagate the full loss, and assert a finite, non-zero `alpha.grad`. This catches accidentally detaching alpha globally rather than only at the SimCSE coefficient.

- [x] **Step 3: Write the failing effective-weight telemetry test**

Build a real `ATOM_GATE` module, execute `compute_training_step`, and assert:

```python
assert output.metrics["alpha_aux_weight/mean"] == pytest.approx(
    1.0 - output.metrics["alpha/mean"]
)
assert output.metrics["alpha_aux_weight/min"] == pytest.approx(
    1.0 - output.metrics["alpha/max"]
)
assert output.metrics["alpha_aux_weight/max"] == pytest.approx(
    1.0 - output.metrics["alpha/min"]
)
```

The mutation caught is losing observability of the effective query-conditioned regularization strength.

- [x] **Step 4: Run the focused tests and verify RED**

Run:

```bash
conda run -n justatom pytest -q \
  tests/test_training_objective.py \
  tests/test_training_module.py
```

Expected: the auxiliary-gradient assertion fails because alpha currently receives a gradient, and the telemetry keys are absent.

- [x] **Step 5: Implement the local stop-gradient and telemetry**

In `ContrastiveObjective.forward`:

```python
auxiliary_weight = 1.0 - alpha.detach()
per_row = main + auxiliary_weight * auxiliary
```

Do not alter the live-alpha expressions used by `mixed_pair` or entropy.

In `ContrastiveTrainingModule.compute_training_step`, next to existing alpha telemetry:

```python
metrics.update(scalar_distribution("alpha", alpha))
metrics.update(scalar_distribution("alpha_aux_weight", 1.0 - alpha.detach()))
```

- [x] **Step 6: Run focused objective and module tests and verify GREEN**

Run:

```bash
conda run -n justatom pytest -q \
  tests/test_training_objective.py \
  tests/test_training_module.py \
  tests/test_training_golden.py \
  tests/test_alpha_gate.py
```

Expected: all tests pass. If a golden expectation encodes the old live-gate gradient, update it only after confirming its forward values remain unchanged.

- [x] **Step 7: Commit detached control**

```bash
git add \
  tests/test_training_objective.py \
  tests/test_training_module.py \
  justatom/training/objective.py \
  justatom/training/module.py
git commit -m "fix(training): detach alpha auxiliary control"
```

---

### Task 3: Reproducibility Metadata and User Documentation

**Files:**
- Modify: `tests/test_training_job.py`
- Modify: `justatom/training/job.py`
- Modify: `docs/training.md`

**Interfaces:**
- Consumes: resolved `TrainConfig.method` and `TrainConfig.objective.decoupled`.
- Produces: top-level `RunManifest.objective_contract` with `contrastive_kernel` and `alpha_aux_gradient`; documents the exact canonical equations.

- [x] **Step 1: Write the failing run-manifest contract test**

Extend the real YAML round-trip test with:

```python
assert loaded["objective_contract"] == {
    "contrastive_kernel": "coupled_infonce",
    "alpha_aux_gradient": "not_applicable",
}
```

Add an `ATOM_GATE` manifest assertion:

```python
gate = RunManifest.from_config(
    canonical_method_config(TrainingMethod.ATOM_GATE),
    git_commit="abc123",
    git_dirty=False,
)
assert gate.objective_contract["alpha_aux_gradient"] == "detached"
```

The production mutation caught is omitting the static detach policy from artifacts, which would make old and new runs indistinguishable without commit archaeology.

- [x] **Step 2: Run the manifest test and verify RED**

Run:

```bash
conda run -n justatom pytest -q tests/test_training_job.py::test_manifest_contains_resolved_method_seed_and_git_state
```

Expected: failure because `objective_contract` does not exist.

- [x] **Step 3: Implement objective-contract metadata**

Add a pure helper in `job.py`:

```python
def objective_contract(config: TrainConfig) -> dict[str, str]:
    return {
        "contrastive_kernel": (
            "decoupled_infonce" if config.objective.decoupled else "coupled_infonce"
        ),
        "alpha_aux_gradient": (
            "detached" if config.method is TrainingMethod.ATOM_GATE else "not_applicable"
        ),
    }
```

Store its result in `RunManifest.from_config` and emit it from `to_dict`. Keep schema version `1` because this is an additive field and no strict manifest reader exists.

- [x] **Step 4: Update the public training documentation**

In `docs/training.md`:

- describe one shared coupled InfoNCE row loss for all canonical methods;
- write the gate equation as `(1 - stop_gradient(alpha_i)) lambda_sc L_i_sc`;
- state that alpha remains live for semantic/lexical pair supervision;
- state that DCL is an ablation-only override;
- remove wording that claims only ATOMIC uses coupled InfoNCE.

- [x] **Step 5: Run focused tests and documentation build**

Run:

```bash
conda run -n justatom pytest -q tests/test_training_job.py tests/test_training_config.py
conda run -n justatom mkdocs build --strict
```

Expected: tests and strict documentation build pass.

- [x] **Step 6: Commit metadata and documentation**

```bash
git add tests/test_training_job.py justatom/training/job.py docs/training.md
git commit -m "docs(training): record alpha objective contract"
```

---

### Task 4: Full Verification and Draft PR

**Files:**
- Verify only; modify only when a failing repository-owned check exposes a regression caused by Tasks 1-3.

**Interfaces:**
- Consumes: all prior task commits.
- Produces: a verified branch and a draft pull request against `master`.

- [x] **Step 1: Run formatting and static checks configured by the repository**

Inspect `pyproject.toml`/CI and run the exact local equivalents. At minimum:

```bash
conda run -n justatom black --check justatom tests
conda run -n justatom isort --check-only justatom tests
```

- [x] **Step 2: Run the complete offline test suite**

```bash
conda run -n justatom pytest -q tests -m "not integration and not network"
```

Expected: all selected tests pass with no new warnings attributable to this change.

- [x] **Step 3: Inspect the final diff and branch hygiene**

```bash
git diff --check origin/master...HEAD
git diff --stat origin/master...HEAD
git status --short
```

Confirm only the spec, plan, production files, focused tests, and public training documentation are tracked in the branch. Do not add unrelated untracked files.

- [x] **Step 4: Push and open the draft PR**

```bash
git push -u origin fix/alpha-gate-objective
gh pr create --draft --base master --head fix/alpha-gate-objective \
  --title "fix(training): prevent alpha-gate collapse" \
  --body-file /tmp/justatom-alpha-gate-pr.md
```

The PR body must summarize the mathematical shortcut, the local stop-gradient, the shared coupled baseline, telemetry/manifest additions, test evidence, and the requirement to rerun matched benchmarks rather than pooling historical metrics.
