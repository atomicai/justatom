# Difficulty-Supervised Alpha Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace semantic/lexical alpha mixing with a detached query head supervised by coupled-InfoNCE positive confidence.

**Architecture:** Coupled InfoNCE remains the encoder objective. `QueryAlphaGate` consumes `queries.detach()`, learns detached positive probability with BCE, and supplies detached `1 - alpha` to SimCSE. Old lexical lookup, pair loss, entropy, and warmup plumbing are removed.

**Tech Stack:** Python 3.10-3.13, PyTorch, PyTorch Lightning, dataclasses, pytest, Polars, YAML.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-14-difficulty-alpha-normalized-bank-design.md`.
- Canonical methods retain coupled InfoNCE.
- Add no dependency.
- BCE updates the head but not the encoder; weighted SimCSE updates the encoder but not the head.
- Remove retired alpha aliases rather than silently accepting them.
- Migrate schema-v1 research checkpoints explicitly.
- Keep CPU, CUDA, and MPS tensors finite.

---

## File Map

| File | Responsibility |
| --- | --- |
| `justatom/training/config.py` | Alpha schema and validation |
| `justatom/training/methods.py` | Canonical defaults |
| `justatom/training/objective.py` | Confidence, BCE, loss composition |
| `justatom/training/module.py` | Detached head input, telemetry, migration |
| `justatom/training/telemetry.py` | Retrieval metrics bucketed by detached target confidence |
| `justatom/training/data.py` | Rows without lexical lookup return value |
| `justatom/training/job.py` | Two-value loader factory and provenance |
| `tests/test_training_*.py` | Closed-form, gradient, schema, and integration coverage |
| `tests/test_train_data_preparation.py` | Simplified data contract |
| `docs/training.md` | Public formula |

### Task 1: Replace the alpha schema

**Files:**
- Modify: `justatom/training/config.py`
- Modify: `justatom/training/methods.py`
- Test: `tests/test_training_config.py`
- Test: `tests/test_training_methods.py`

**Interfaces:**
- Produces: `AlphaGateConfig.supervision_weight: float = 0.3`.
- Removes: `pairwise_margin`, `mix_weight`, `mix_weight_warmup_steps`, `entropy_weight`.

- [ ] **Step 1: Write failing schema tests**

```python
def test_alpha_supervision_schema_round_trips():
    config = parse_train_config(
        {"method": "atom_gate", "alpha_gate": {"supervision_weight": 0.4}}
    )
    payload = train_config_to_dict(config)
    assert config.alpha_gate.supervision_weight == pytest.approx(0.4)
    assert "mix_weight" not in payload["alpha_gate"]
    assert "pairwise_margin" not in payload["objective"]
    assert parse_train_config(payload) == config


@pytest.mark.parametrize("field", ["mix_weight", "mix_weight_warmup_steps", "entropy_weight"])
def test_retired_alpha_fields_are_rejected(field):
    with pytest.raises(ValueError, match=rf"unknown configuration field: alpha_gate\.{field}"):
        parse_train_config({"method": "atom_gate", "alpha_gate": {field: 0}})


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_alpha_supervision_weight_is_finite_and_non_negative(value):
    with pytest.raises(ValueError, match=r"alpha_gate\.supervision_weight"):
        parse_train_config(
            {"method": "atom_gate", "alpha_gate": {"supervision_weight": value}}
        )
```

Extend `test_canonical_profiles_have_exact_structural_components`:

```python
assert gate.alpha_gate.supervision_weight == pytest.approx(0.3)
assert not hasattr(gate.alpha_gate, "mix_weight")
assert not hasattr(gate.objective, "pairwise_margin")
```

- [ ] **Step 2: Verify failure**

```bash
python -m pytest tests/test_training_config.py tests/test_training_methods.py -q
```

Expected: FAIL because `supervision_weight` is absent.

- [ ] **Step 3: Implement minimal schema**

```python
@dataclass(frozen=True)
class ObjectiveConfig:
    temperature: float = 0.05
    learnable_temperature: bool = True
    decoupled: bool = True
    simcse_dropout_weight: float = 0.0
    soft_fn_attract_weight: float = 0.0
    soft_fn_topk: int = 1


@dataclass(frozen=True)
class AlphaGateConfig:
    enabled: bool = False
    supervision_weight: float = 0.3
    head: AlphaHeadConfig = field(default_factory=AlphaHeadConfig)
```

Make the shared numeric validator reject non-finite values, then validate the
loss coefficient without an artificial upper bound:

```python
if not math.isfinite(float(value)):
    raise ValueError(f"{path} must be finite")

_require_number(gate.supervision_weight, "alpha_gate.supervision_weight", 0.0)
```

In `canonical_method_config`, use
`AlphaGateConfig(enabled=True, supervision_weight=0.3)`.

- [ ] **Step 4: Verify pass and lint**

```bash
python -m pytest tests/test_training_config.py tests/test_training_methods.py -q
ruff check justatom/training/config.py justatom/training/methods.py tests/test_training_config.py tests/test_training_methods.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add justatom/training/config.py justatom/training/methods.py tests/test_training_config.py tests/test_training_methods.py
git commit -m "refactor(training): define alpha confidence supervision"
```

### Task 2: Implement confidence BCE

**Files:**
- Modify: `justatom/training/objective.py`
- Test: `tests/test_training_objective.py`

**Interfaces:**
- Produces: `ObjectiveInputs.alpha_supervision_weight: float`.
- Produces: `ObjectiveOutput.alpha_target` and `alpha_supervision_per_row`.
- Produces metric: `loss/alpha_supervision`.

- [ ] **Step 1: Replace pair and entropy tests with closed-form tests**

```python
def test_atom_gate_uses_detached_positive_confidence():
    objective = ContrastiveObjective(
        ObjectiveConfig(temperature=1.0, learnable_temperature=False, decoupled=False)
    )
    q = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1).requires_grad_()
    p = F.normalize(torch.tensor([[1.0, 0.0], [0.6, 0.8]]), dim=-1).requires_grad_()
    alpha = torch.tensor([0.4, 0.7], requires_grad=True)
    output = objective(
        ObjectiveInputs(
            queries=q,
            positives=p,
            alpha=alpha,
            alpha_supervision_weight=0.3,
        )
    )
    target = torch.softmax(q.detach() @ p.detach().T, dim=-1).diagonal()
    bce = F.binary_cross_entropy(alpha, target, reduction="none")
    torch.testing.assert_close(output.alpha_target, target)
    torch.testing.assert_close(output.alpha_supervision_per_row, bce)
    torch.testing.assert_close(output.loss, output.main_per_row.mean() + 0.3 * bce.mean())


def test_alpha_bce_does_not_update_retrieval_embeddings():
    objective = ContrastiveObjective(
        ObjectiveConfig(temperature=1.0, learnable_temperature=False, decoupled=False)
    )
    q = F.normalize(torch.randn(3, 4), dim=-1).requires_grad_()
    p = F.normalize(torch.randn(3, 4), dim=-1).requires_grad_()
    alpha = torch.tensor([0.2, 0.5, 0.8], requires_grad=True)
    output = objective(
        ObjectiveInputs(queries=q, positives=p, alpha=alpha, alpha_supervision_weight=1.0)
    )
    output.alpha_supervision_per_row.mean().backward()
    assert alpha.grad is not None and float(alpha.grad.abs().sum()) > 0.0
    assert q.grad is None and p.grad is None


def test_alpha_bce_is_minimized_at_soft_target():
    target = torch.tensor([0.2, 0.8])
    at_target = F.binary_cross_entropy(target, target)
    away_from_target = F.binary_cross_entropy(torch.tensor([0.4, 0.6]), target)
    assert at_target < away_from_target


def test_hard_row_receives_more_simcse_pressure_than_easy_row():
    alpha = torch.tensor([0.2, 0.8])
    simcse = torch.tensor([2.0, 2.0])
    weighted = (1.0 - alpha.detach()) * simcse
    torch.testing.assert_close(weighted, torch.tensor([1.6, 0.4]))
```

Keep the SimCSE detach test with `alpha_supervision_weight=0.0`; assert `alpha.grad is None`.

- [ ] **Step 2: Verify failure**

```bash
python -m pytest tests/test_training_objective.py -q
```

Expected: FAIL on missing inputs and outputs.

- [ ] **Step 3: Implement BCE and remove old branches**

Import `torch.nn.functional as F`. Replace retired input fields with `alpha_supervision_weight`. Add nullable target and supervision tensors to `ObjectiveOutput`.

```python
confidence_logits = inputs.queries.detach() @ inputs.positives.detach().T
confidence_logits = confidence_logits / self.kernel.tau.detach()
alpha_target = torch.softmax(confidence_logits, dim=-1).diagonal()
alpha_supervision = F.binary_cross_entropy(alpha, alpha_target, reduction="none")
```

Compose:

```python
per_row = main + auxiliary
if inputs.alpha is not None:
    auxiliary_weight = 1.0 - inputs.alpha.view(-1).detach()
    per_row = main + auxiliary_weight * auxiliary
    per_row = per_row + inputs.alpha_supervision_weight * alpha_supervision
```

Delete semantic/lexical and entropy blocks. Emit `loss/alpha_supervision`; return target and per-row BCE.

- [ ] **Step 4: Verify pass**

```bash
python -m pytest tests/test_training_objective.py -q
ruff check justatom/training/objective.py tests/test_training_objective.py
```

Expected: PASS with finite gradients.

- [ ] **Step 5: Commit**

```bash
git add justatom/training/objective.py tests/test_training_objective.py
git commit -m "feat(training): supervise alpha from retrieval confidence"
```

### Task 3: Detach head input and remove lexical plumbing

**Files:**
- Modify: `justatom/training/module.py`
- Modify: `justatom/training/telemetry.py`
- Modify: `justatom/training/data.py`
- Modify: `justatom/training/job.py`
- Modify: `tests/test_training_module.py`
- Modify: `tests/test_train_data_preparation.py`
- Modify: `tests/test_training_job.py`

**Interfaces:**
- Produces: `sample_training_rows(...) -> list[dict[str, Any]]`.
- Produces: `prepare_training_data(...) -> tuple[pl.DataFrame, list[dict[str, Any]]]`.
- Produces: loader factory `(loader, processor)`.
- Produces: `retrieval_metrics_by_confidence(scores, confidence) -> dict[str, float]`.

- [ ] **Step 1: Write failing module and factory tests**

```python
def test_atom_gate_bce_updates_only_head_parameters():
    module = ContrastiveTrainingModule.build(
        TinyEncoder(), canonical_method_config(TrainingMethod.ATOM_GATE)
    )
    output = module.compute_training_step(tiny_batch(), step=0)
    output.alpha_supervision_per_row.mean().backward()
    assert all(parameter.grad is None for parameter in module.encoder.parameters())
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in module.alpha_gate.parameters()
    )


def test_atom_gate_reports_calibration():
    module = ContrastiveTrainingModule.build(
        TinyEncoder(), canonical_method_config(TrainingMethod.ATOM_GATE)
    )
    output = module.compute_training_step(tiny_batch(), step=0)
    assert 0.0 <= output.metrics["alpha_target/mean"] <= 1.0
    assert output.metrics["alpha/absolute_error_mean"] >= 0.0


def test_retrieval_metrics_are_bucketed_by_target_confidence():
    scores = torch.tensor([[2.0, 0.0], [2.0, 1.0]])
    confidence = torch.tensor([0.2, 0.8])
    metrics = retrieval_metrics_by_confidence(scores, confidence)
    assert metrics["alpha_target_bucket/low/count"] == 1.0
    assert metrics["alpha_target_bucket/low/hit_rate_at_1"] == 1.0
    assert metrics["alpha_target_bucket/high/count"] == 1.0
    assert metrics["alpha_target_bucket/high/hit_rate_at_1"] == 0.0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_atom_gate_two_steps_stay_finite_on_mps():
    module = ContrastiveTrainingModule.build(
        TinyEncoder().to("mps"), canonical_method_config(TrainingMethod.ATOM_GATE)
    ).to("mps")
    for _ in range(2):
        batch = {key: value.to("mps") for key, value in tiny_batch().items()}
        output = module.compute_training_step(batch, step=0)
        output.loss.backward()
        assert torch.isfinite(output.loss)
        assert output.alpha_target is not None and torch.isfinite(output.alpha_target).all()
        module.zero_grad(set_to_none=True)
```

Update every fake `ObjectiveOutput` with `alpha_target=None` and `alpha_supervision_per_row=None`. Change data tests to unpack two values and job factories to return `("loader", "processor")`.

- [ ] **Step 2: Verify failure**

```bash
python -m pytest tests/test_training_module.py tests/test_train_data_preparation.py tests/test_training_job.py -q
```

Expected: FAIL on live head input and three-value factories.

- [ ] **Step 3: Simplify production flow**

In `compute_training_step`:

```python
alpha = None if self.alpha_gate is None else self.alpha_gate(queries.detach())
```

Pass `alpha_supervision_weight=self.config.alpha_gate.supervision_weight`. Record `scalar_distribution("alpha_target", output.alpha_target)` and mean absolute alpha error.

Remove `lexical_lookup`, `_last_negative_indices`, `_negative_indices`, `_semantic_pair_scores`, `_lexical_pair_scores`, `_remove_text_prefix`, and `effective_alpha_mix_weight` from the module.

Add fixed confidence buckets to `telemetry.py`. Empty buckets retain the same
keys and use `count=0`, `hit_rate_at_1=NaN`, and `mrr=NaN`, so CSV columns do
not change between microbatches:

```python
@torch.no_grad()
def retrieval_metrics_by_confidence(
    scores: torch.Tensor,
    confidence: torch.Tensor,
) -> dict[str, float]:
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"scores must be square, got {tuple(scores.shape)}")
    confidence = confidence.detach().reshape(-1).to(scores.device)
    if confidence.shape[0] != scores.shape[0]:
        raise ValueError("confidence batch dimension must match scores")
    targets = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1)
    ranks = (torch.argsort(scores, dim=1, descending=True) == targets).nonzero(as_tuple=False)[:, 1] + 1
    result: dict[str, float] = {}
    buckets = (
        ("low", 0.0, 1.0 / 3.0, False),
        ("medium", 1.0 / 3.0, 2.0 / 3.0, False),
        ("high", 2.0 / 3.0, 1.0, True),
    )
    for name, lower, upper, include_upper in buckets:
        mask = (confidence >= lower) & ((confidence <= upper) if include_upper else (confidence < upper))
        prefix = f"alpha_target_bucket/{name}"
        result[f"{prefix}/count"] = float(mask.sum().item())
        result[f"{prefix}/hit_rate_at_1"] = (
            float((ranks[mask] <= 1).float().mean().item()) if mask.any() else float("nan")
        )
        result[f"{prefix}/mrr"] = (
            float((1.0 / ranks[mask].float()).mean().item()) if mask.any() else float("nan")
        )
    return result
```

Change data and job boundaries:

```python
def sample_training_rows(...) -> list[dict[str, Any]]:
    return _reservoir_sample_rows(iterate_training_rows(...), int(num_samples), seed=seed)


def prepare_training_data(**kwargs) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    sampled = sample_training_rows(**kwargs)
    return pl.from_dicts(sampled) if sampled else pl.DataFrame(...), sampled


def prepare_training_data_from_config(config: TrainConfig) -> list[dict[str, Any]]:
    rows = sample_training_rows(...)
    return rebalance_rows_by_content(rows, config.optimization.batch_size)
```

`TrainingJob` must unpack `(loader, processor)` and call `module_factory(encoder, config)`.

- [ ] **Step 4: Verify cleanup**

```bash
python -m pytest tests/test_training_module.py tests/test_train_data_preparation.py tests/test_training_job.py -q
rg -n "lexical_lookup|alpha_mix_weight|mix_weight_warmup|_lexical_pair_scores|_semantic_pair_scores" justatom/training tests
ruff check justatom/training tests/test_training_module.py tests/test_train_data_preparation.py tests/test_training_job.py
```

Expected: tests pass and `rg` returns no retired production alpha references.

- [ ] **Step 5: Commit**

```bash
git add justatom/training/module.py justatom/training/telemetry.py justatom/training/data.py justatom/training/job.py tests/test_training_module.py tests/test_train_data_preparation.py tests/test_training_job.py
git commit -m "refactor(training): isolate alpha head data flow"
```

### Task 4: Migrate checkpoints and record provenance

**Files:**
- Modify: `justatom/training/module.py`
- Modify: `justatom/training/job.py`
- Modify: `tests/test_training_module.py`
- Modify: `tests/test_training_job.py`
- Modify: `docs/training.md`

**Interfaces:**
- Produces: checkpoint schema `2`, loader support for schemas `1` and `2`.
- Produces manifest keys `alpha_target` and `alpha_head_input_gradient`.

- [ ] **Step 1: Write failing migration and manifest tests**

Build a schema-v1 payload containing `mix_weight`, `mix_weight_warmup_steps`, `entropy_weight`, and `pairwise_margin`. Assert load maps weight to `supervision_weight`, marks role `ablation`, and loads the unchanged head state strictly.

```python
assert manifest.objective_contract == {
    "contrastive_kernel": "coupled_infonce",
    "alpha_aux_gradient": "detached",
    "alpha_target": "detached_positive_softmax_confidence",
    "alpha_head_input_gradient": "detached",
}
```

- [ ] **Step 2: Verify failure**

```bash
python -m pytest tests/test_training_module.py::test_load_schema_v1_checkpoint_migrates_historical_canonical_dcl_to_ablation tests/test_training_job.py -q
```

Expected: FAIL on retired schema keys and missing manifest fields.

- [ ] **Step 3: Implement migration and manifest contract**

Save schema `2` and accept `{1, 2}` on load. For schema 1, copy nested
dictionaries and migrate without mutating the loaded payload:

```python
objective = dict(resolved_config.get("objective", {}))
objective.pop("pairwise_margin", None)
resolved_config["objective"] = objective

alpha_gate = dict(resolved_config.get("alpha_gate", {}))
alpha_gate["supervision_weight"] = alpha_gate.pop("mix_weight", 0.3)
alpha_gate.pop("mix_weight_warmup_steps", None)
alpha_gate.pop("entropy_weight", None)
resolved_config["alpha_gate"] = alpha_gate

experiment = dict(resolved_config.get("experiment", {}))
if resolved_config.get("method") == "atom_gate" or objective.get("decoupled") is True:
    experiment["role"] = "ablation"
resolved_config["experiment"] = experiment
```

This preserves canonical coupled `vanilla`/`atomic` checkpoints, while old
atom-gate semantics and historical DCL checkpoints remain explicitly labeled
as ablations.

For atom gate, extend `objective_contract` with:

```python
{
    "alpha_target": "detached_positive_softmax_confidence",
    "alpha_head_input_gradient": "detached",
}
```

- [ ] **Step 4: Document and verify**

Document:

```text
t_i = stop_gradient(softmax(S / tau)_ii)
alpha_i = sigmoid(MLP(stop_gradient(q_i)))
L_i = L_InfoNCE,i
    + (1 - stop_gradient(alpha_i)) lambda_sc L_SimCSE,i
    + lambda_alpha BCE(alpha_i, t_i)
```

Run:

```bash
python -m pytest tests/test_training_module.py tests/test_training_job.py -q
ruff check justatom/training/module.py justatom/training/job.py tests/test_training_module.py tests/test_training_job.py
git diff --check
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add justatom/training/module.py justatom/training/job.py tests/test_training_module.py tests/test_training_job.py docs/training.md
git commit -m "docs(training): record confidence-supervised alpha contract"
```

### Task 5: Verify alpha end to end

**Files:**
- Verify only; repair the preceding files only when a command exposes a defect.

**Interfaces:**
- Produces: clean prerequisite for the normalized-bank plan.

- [ ] **Step 1: Run focused tests**

```bash
python -m pytest tests/test_training_config.py tests/test_training_methods.py tests/test_training_objective.py tests/test_training_module.py tests/test_train_data_preparation.py tests/test_training_job.py tests/test_train_cli.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the offline suite**

```bash
python -m pytest tests -m "not integration and not network" -q
ruff check justatom tests
git diff --check
```

Expected: PASS.

- [ ] **Step 3: Verify retired symbols are absent**

```bash
rg -n "pairwise_margin|mix_weight|mix_weight_warmup_steps|entropy_weight|loss/lexical_mix|lexical_lookup" justatom configs tests docs/training.md
```

Expected: no production alpha matches; migration tests may contain legacy field names.

- [ ] **Step 4: Run the two-step MPS test**

```bash
python -m pytest tests/test_training_module.py::test_atom_gate_two_steps_stay_finite_on_mps -q
```

Run under `conda activate justatom`; expected: PASS, or SKIP only when MPS is unavailable.

- [ ] **Step 5: Commit only a verified repair**

```bash
git status --short
```

If a repair changed tracked files, stage those exact files and commit `fix(training): complete alpha verification`. Otherwise create no commit.
