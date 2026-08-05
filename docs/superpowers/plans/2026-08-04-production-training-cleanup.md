# Production Training Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the accumulated gamma-era training stack with one typed, reproducible training path exposing only `vanilla`, `atom_gate`, and `atomic`, while preserving the current validated tensors and gradients.

**Architecture:** Parse YAML and CLI overrides once into a strict `TrainConfig`, resolve one of three method profiles, and pass the immutable resolved config through a single `TrainingJob` into one compositional Lightning module. The module delegates query control, contrastive objective, memory-bank selection, and telemetry to focused components; legacy trainer/job inheritance is deleted only after deterministic parity tests pass.

**Tech Stack:** Python 3.10+, PyTorch, PyTorch Lightning, dataclasses/enums, Polars, YAML scenario loader, pytest, Bash pipeline scripts.

## Global Constraints

- The root `justatom` package is the only source of truth; do not edit `justatom-rc`.
- Public method names are exactly `vanilla`, `atom_gate`, and `atomic`.
- Canonical `atom_gate` is query-only `alpha(q)` with no memory bank.
- Canonical `atomic` is `alpha(q)` plus adaptive dynamic memory bank plus `m(q)`.
- Structural controls require `experiment.role: ablation` and must be recorded in the manifest.
- Preserve current canonical numerical behavior before deleting any old implementation.
- CPU float32 is the golden numerical oracle; MPS is used only for smoke tests.
- Do not add a new configuration or validation dependency.
- Mathematical behavior must not be controlled by environment variables.
- Keep deployable encoder artifacts separate from research checkpoints.
- Do not modify unrelated retrieval, storage, service, evaluation, dissertation, or `justatom-rc` code.
- Work with the existing dirty tree; never revert unrelated user changes.

---

## File Map

### New production modules

- `justatom/training/config.py`: typed configuration, strict parsing, value validation.
- `justatom/training/methods.py`: canonical defaults and method invariants.
- `justatom/training/alpha_gate.py`: query-only `QueryAlphaGate`.
- `justatom/training/objective.py`: contrastive, SimCSE, lexical auxiliary, and margin regularization composition.
- `justatom/training/telemetry.py`: distribution, retrieval, gradient, and device-transfer metrics.
- `justatom/training/sampling.py`: safe in-batch negative selection and lexical overlap helpers.
- `justatom/training/module.py`: the sole Lightning training module.
- `justatom/training/data.py`: training-row preparation and loader construction.
- `justatom/training/job.py`: run orchestration, manifests, checkpoints, and deployable artifacts.

### Existing modules retained and narrowed

- `justatom/training/loss.py`: retain the canonical contrastive kernel and unrelated externally referenced losses; remove retired training-only branches after reference checks.
- `justatom/training/memory_bank.py`: retain FIFO/mining behavior and return typed selections.
- `justatom/training/diagnostics.py`: retain embedding geometry calculations.
- `justatom/running/encoders.py`: retain generic encoder runners; remove `GammaHybridRunner`.
- `justatom/api/train.py`: become a thin config/dispatch entry point.
- `configs/train.yaml`: expose the canonical schema.
- `justatom/builtins/configs/train.default.yaml`: packaged mirror of the canonical schema.
- `scripts/run_pipeline.sh`: forward `--method` and explicit overrides; do not resolve method defaults.
- `scripts/run_benchmark.sh`: benchmark only the three canonical variants.

### Legacy modules removed after parity

- `justatom/running/trainer.py`.
- `justatom/running/trainer_jobs.py`.
- `justatom/api/tune.py` if the final repository-wide reference check confirms that it has no supported entry point.
- Retired gamma/query-pair/diagonal/tau tests and scripts.

### Test layout

- `tests/test_training_config.py`.
- `tests/test_training_methods.py`.
- `tests/test_training_golden.py`.
- `tests/test_alpha_gate.py`.
- `tests/test_memory_bank.py`.
- `tests/test_training_objective.py`.
- `tests/test_training_module.py`.
- `tests/test_training_job.py`.
- Existing dataset, scenario, benchmark, Qwen, and evaluation tests updated in place.

---

### Task 1: Stabilize The Current ATOMIC Baseline

**Files:**
- Modify: none during the audit step
- Test: `tests/test_soft_contrastive_loss.py`
- Test: `tests/test_train_data_preparation.py`
- Test: `tests/test_benchmark_variants.py`
- Test: `tests/test_scenario_configs.py`

**Interfaces:**
- Consumes: the current dirty implementation on `feature/qc-ch-margin-head`.
- Produces: a tested baseline commit from which an isolated cleanup worktree can be created.

- [ ] **Step 1: Record the dirty-tree inventory without modifying it**

Run:

```bash
git status --short --branch
git diff --stat
git diff --check
```

Expected: `git diff --check` exits `0`; the inventory includes the current alpha, bank, margin-head, Qwen, config, and benchmark work.

- [ ] **Step 2: Run the current contract tests in the project environment**

Run:

```bash
conda run -n justatom pytest \
  tests/test_soft_contrastive_loss.py \
  tests/test_train_data_preparation.py \
  tests/test_benchmark_variants.py \
  tests/test_scenario_configs.py \
  tests/test_training_prefixes.py \
  tests/test_qwen3_embedding_model.py -q
```

Expected: all selected tests pass. Any pre-existing failure is diagnosed and fixed before proceeding; do not establish a failing oracle.

- [ ] **Step 3: Stage only the current research implementation and its tests**

Run:

```bash
git add \
  configs/train.yaml \
  justatom/api/eval.py \
  justatom/api/train.py \
  justatom/builtins/configs/train.default.yaml \
  justatom/modeling/mask.py \
  justatom/modeling/prime.py \
  justatom/processing/prime.py \
  justatom/running/encoders.py \
  justatom/running/trainer.py \
  justatom/running/trainer_jobs.py \
  justatom/storing/dataset.py \
  justatom/tooling/dataset.py \
  justatom/training/diagnostics.py \
  justatom/training/loss.py \
  justatom/training/memory_bank.py \
  scripts/run_pipeline.sh \
  scripts/run_benchmark.sh \
  tests/test_benchmark_variants.py \
  tests/test_eval_data_normalization.py \
  tests/test_qwen3_embedding_model.py \
  tests/test_scenario_configs.py \
  tests/test_soft_contrastive_loss.py \
  tests/test_train_data_preparation.py \
  tests/test_training_prefixes.py
```

Review:

```bash
git diff --cached --stat
git diff --cached --check
```

Expected: no dissertation build products, `.tmp_runs`, `weights`, `tmp`, `justatom-rc`, `.env`, or unrelated documents are staged.

- [ ] **Step 4: Commit the tested oracle**

```bash
git commit -m "feat: checkpoint atomic research baseline"
```

Expected: one baseline commit containing the implementation that generated the existing experiment evidence.

- [ ] **Step 5: Create an isolated cleanup worktree**

Use `superpowers:using-git-worktrees`, then create a branch named `refactor/production-training` from the baseline commit. All subsequent tasks run in that worktree.

---

### Task 2: Freeze Golden Mathematical Contracts

**Files:**
- Create: `tests/test_training_golden.py`
- Modify: `tests/test_soft_contrastive_loss.py`

**Interfaces:**
- Consumes: `ContrastiveLoss`, `ContrastiveMemoryBank`, `GammaHybridRunner`, and `_BaseGammaLightningTrainer._compute_contrastive_loss` from the baseline.
- Produces: deterministic CPU float32 assertions that the new objective, alpha head, bank selection, and gradients must satisfy.

- [ ] **Step 1: Add deterministic tensor fixtures**

Create these fixtures at the top of `tests/test_training_golden.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def golden_embeddings() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = F.normalize(
        torch.tensor(
            [[1.0, 0.2, 0.0], [0.1, 1.0, 0.2], [0.0, 0.2, 1.0]],
            dtype=torch.float32,
        ),
        dim=-1,
    ).requires_grad_()
    p = F.normalize(
        torch.tensor(
            [[0.9, 0.1, 0.0], [0.2, 0.9, 0.1], [0.1, 0.1, 0.9]],
            dtype=torch.float32,
        ),
        dim=-1,
    ).requires_grad_()
    bank = F.normalize(
        torch.tensor(
            [[0.8, 0.2, 0.0], [0.0, 0.8, 0.2], [0.2, 0.0, 0.8], [-1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    return q, p, bank


def golden_ids() -> dict[str, torch.Tensor]:
    return {
        "doc_key_id": torch.tensor([101, 102, 103]),
        "content_key_id": torch.tensor([201, 202, 203]),
        "query_key_id": torch.tensor([301, 302, 303]),
    }
```

- [ ] **Step 2: Add the vanilla closed-form contract**

```python
def test_golden_vanilla_matches_decoupled_closed_form():
    from justatom.training.loss import ContrastiveLoss

    q, p, _ = golden_embeddings()
    loss_fn = ContrastiveLoss(
        temperature=0.05,
        learnable_temperature=False,
        decoupled=True,
        reduction="none",
    )
    actual = loss_fn.info_nce(q, p)

    sim = F.normalize(q, dim=-1) @ F.normalize(p, dim=-1).T / 0.05
    eye = torch.eye(sim.shape[0], dtype=torch.bool)
    expected = -sim.diagonal() + torch.logsumexp(sim.masked_fill(eye, -1e9), dim=-1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    actual.mean().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert p.grad is not None and torch.isfinite(p.grad).all()
```

- [ ] **Step 3: Add the alpha augment contract**

```python
def test_golden_atom_gate_uses_query_alpha_for_auxiliary_pressure():
    from justatom.training.loss import ContrastiveLoss

    q, p, _ = golden_embeddings()
    q_alt = F.normalize(q.detach() + 0.03, dim=-1).requires_grad_()
    alpha_logits = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    alpha = torch.sigmoid(alpha_logits)
    loss_fn = ContrastiveLoss(temperature=0.05, learnable_temperature=False, decoupled=True)

    main = loss_fn.info_nce(q, p, reduction="none")
    simcse = loss_fn.simcse_term(q, q_alt, reduction="none")
    actual = (main + (1.0 - alpha) * 0.1 * simcse).mean()
    expected = torch.mean(main + (1.0 - torch.sigmoid(alpha_logits)) * 0.1 * simcse)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    actual.backward()
    assert alpha_logits.grad is not None
    assert torch.isfinite(alpha_logits.grad).all()
    assert float(alpha_logits.grad.abs().sum()) > 0.0
```

- [ ] **Step 4: Add the ATOMIC denominator and margin-gradient contract**

```python
def test_golden_atomic_adds_weighted_bank_logits_and_live_margin_gradient():
    from justatom.training.loss import ContrastiveLoss

    q, p, bank = golden_embeddings()
    margin = torch.full((q.shape[0],), 0.05, requires_grad=True)
    hard_weight = torch.tensor([0.25, 0.5, 1.0])
    mask = torch.ones(q.shape[0], bank.shape[0], dtype=torch.bool)
    log_weights = torch.log(hard_weight).view(-1, 1).expand(-1, bank.shape[0])
    loss_fn = ContrastiveLoss(
        temperature=0.05,
        learnable_temperature=False,
        decoupled=True,
        reduction="none",
    )

    actual = loss_fn.info_nce(
        q,
        p,
        memory_negatives=bank,
        memory_negative_mask=mask,
        memory_log_weights=log_weights,
        memory_margin=margin,
        memory_soft_beta=0.05,
    )

    qn, pn, bn = F.normalize(q, dim=-1), F.normalize(p, dim=-1), F.normalize(bank, dim=-1)
    current = qn @ pn.T / 0.05
    positive = (qn * pn).sum(dim=-1, keepdim=True)
    bank_cos = qn @ bn.T
    admission = torch.sigmoid((positive - margin.view(-1, 1) - bank_cos) / 0.05)
    bank_logits = bank_cos / 0.05 + log_weights + torch.log(admission.clamp_min(1e-8))
    negatives = torch.cat(
        [current.masked_fill(torch.eye(q.shape[0], dtype=torch.bool), -1e9), bank_logits],
        dim=1,
    )
    expected = -current.diagonal() + torch.logsumexp(negatives, dim=-1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    actual.mean().backward()
    assert margin.grad is not None and torch.isfinite(margin.grad).all()
    assert float(margin.grad.abs().sum()) > 0.0
```

- [ ] **Step 5: Run the golden tests**

Run:

```bash
conda run -n justatom pytest tests/test_training_golden.py -q
```

Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add tests/test_training_golden.py tests/test_soft_contrastive_loss.py
git commit -m "test: freeze training mathematics"
```

---

### Task 3: Add Strict Typed Configuration

**Files:**
- Create: `justatom/training/config.py`
- Create: `tests/test_training_config.py`

**Interfaces:**
- Consumes: nested mappings returned by `load_scenario_config`.
- Produces: `parse_train_config(raw: Mapping[str, Any]) -> TrainConfig`, `train_config_to_dict(config: TrainConfig) -> dict[str, Any]`, and immutable config dataclasses.

- [ ] **Step 1: Write failing parsing and rejection tests**

```python
import pytest

from justatom.training.config import ExperimentRole, TrainingMethod, parse_train_config


def test_parse_train_config_builds_typed_atomic_config():
    config = parse_train_config(
        {
            "method": "atomic",
            "experiment": {"role": "canonical", "seed": 42},
            "model": {"name_or_path": "intfloat/multilingual-e5-small"},
            "dataset": {"name_or_path": "justatom/mmarco-ru-selected"},
        }
    )
    assert config.method is TrainingMethod.ATOMIC
    assert config.experiment.role is ExperimentRole.CANONICAL
    assert config.experiment.seed == 42


def test_parse_train_config_rejects_unknown_nested_field():
    with pytest.raises(ValueError, match=r"memory_bank\.mystery"):
        parse_train_config(
            {
                "method": "atomic",
                "memory_bank": {"mystery": 7},
            }
        )


def test_parse_train_config_rejects_invalid_beta_before_model_load():
    with pytest.raises(ValueError, match=r"memory_bank\.adaptive\.collision_beta"):
        parse_train_config(
            {
                "method": "atomic",
                "memory_bank": {"adaptive": {"collision_beta": 0.0}},
            }
        )
```

- [ ] **Step 2: Run tests and confirm the module is missing**

Run:

```bash
conda run -n justatom pytest tests/test_training_config.py -q
```

Expected: collection fails with `ModuleNotFoundError: justatom.training.config`.

- [ ] **Step 3: Implement enums and frozen configuration records**

Implement these public types in `justatom/training/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Mapping, TypeVar


class TrainingMethod(str, Enum):
    VANILLA = "vanilla"
    ATOM_GATE = "atom_gate"
    ATOMIC = "atomic"


class ExperimentRole(str, Enum):
    CANONICAL = "canonical"
    ABLATION = "ablation"


class MarginMode(str, Enum):
    OFF = "off"
    CONSTANT = "constant"
    QUERY = "query"


@dataclass(frozen=True)
class ExperimentConfig:
    role: ExperimentRole = ExperimentRole.CANONICAL
    seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str = "intfloat/multilingual-e5-small"
    query_prefix: str = "query:"
    content_prefix: str = "passage:"
    max_query_seq_len: int | None = None
    max_seq_len: int = 512


@dataclass(frozen=True)
class DatasetConfig:
    id: str | None = None
    name_or_path: str | None = None
    labels_field: str = "queries"
    content_field: str = "content"
    split: str | None = None
    limit: int | None = None
    chunk_id_col: str | None = None
    keywords_col: str | None = "keywords_or_phrases"
    keywords_nested_col: str | None = None
    explanation_nested_col: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FilterConfig:
    fields: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class OptimizationConfig:
    optimizer: str = "adamw"
    lr_encoder: float = 2e-5
    lr_heads: float = 1e-2
    weight_decay: float = 0.01
    batch_size: int = 32
    grad_acc_steps: int = 1
    epochs: int = 1
    num_samples: int = 100


@dataclass(frozen=True)
class ObjectiveConfig:
    temperature: float = 0.05
    learnable_temperature: bool = True
    decoupled: bool = True
    pairwise_margin: float = 0.5
    simcse_dropout_weight: float = 0.0
    soft_fn_attract_weight: float = 0.0
    soft_fn_topk: int = 1


@dataclass(frozen=True)
class AlphaHeadConfig:
    layers: int = 1
    hidden_dim: int | None = None
    dropout: float = 0.0
    activation: str = "gelu"


@dataclass(frozen=True)
class AlphaGateConfig:
    enabled: bool = False
    mix_weight: float = 0.3
    mix_weight_warmup_steps: int = 0
    entropy_weight: float = 0.0
    head: AlphaHeadConfig = field(default_factory=AlphaHeadConfig)


@dataclass(frozen=True)
class AdaptiveBankConfig:
    enabled: bool = False
    collision_threshold: float = 0.0
    collision_beta: float = 0.05


@dataclass(frozen=True)
class MarginConfig:
    mode: MarginMode = MarginMode.OFF
    base: float = 0.05
    scale: float = 0.02
    minimum: float = 0.0
    maximum: float = 0.15
    admission_beta: float = 0.05
    regularization_weight: float = 0.0


@dataclass(frozen=True)
class MemoryBankConfig:
    enabled: bool = False
    size: int = 0
    warmup_steps: int = 0
    mining: str = "all"
    hard_negatives: int = 0
    random_negatives: int = 0
    hard_warmup_steps: int = 0
    hard_ramp_steps: int = 1
    adaptive: AdaptiveBankConfig = field(default_factory=AdaptiveBankConfig)
    margin: MarginConfig = field(default_factory=MarginConfig)


@dataclass(frozen=True)
class TelemetryConfig:
    backend: str = "csv"
    metrics_path: str | None = None
    wandb_project: str = "justatom"
    wandb_run_name: str | None = None


@dataclass(frozen=True)
class ArtifactConfig:
    save_dir: str | None = None
    collection_name: str | None = None
    collection_tag: str | None = None
    save_research_checkpoint: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    accelerator: str = "auto"
    devices: str | int = "auto"


@dataclass(frozen=True)
class TrainConfig:
    method: TrainingMethod
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    alpha_gate: AlphaGateConfig = field(default_factory=AlphaGateConfig)
    memory_bank: MemoryBankConfig = field(default_factory=MemoryBankConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
```

Implement strict recursive parsing with an explicit parser table. Do not rely on `dataclass(**mapping)` for nested values because enum conversion and dotted-path errors must be deterministic:

```python
def _reject_unknown(path: str, raw: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        dotted = f"{path}.{unknown[0]}" if path else unknown[0]
        raise ValueError(f"Unknown configuration field: {dotted}")


def parse_train_config(raw: Mapping[str, Any]) -> TrainConfig:
    _reject_unknown(
        "",
        raw,
        {
            "method", "experiment", "model", "dataset", "filters", "optimization",
            "objective", "alpha_gate", "memory_bank", "telemetry",
            "artifacts", "runtime",
        },
    )
    if "method" not in raw:
        raise ValueError("Missing required configuration field: method")
    method = TrainingMethod(str(raw["method"]))
    # Local import avoids a module-import cycle: methods.py defines profiles
    # using the dataclasses above, while parsing starts from the selected profile.
    from justatom.training.methods import canonical_method_config

    defaults = canonical_method_config(method)
    config = TrainConfig(
        method=method,
        experiment=_parse_experiment(raw.get("experiment", {}), defaults.experiment),
        model=_parse_model(raw.get("model", {}), defaults.model),
        dataset=_parse_dataset(raw.get("dataset", {}), defaults.dataset),
        filters=_parse_filters(raw.get("filters", {}), defaults.filters),
        optimization=_parse_optimization(raw.get("optimization", {}), defaults.optimization),
        objective=_parse_objective(raw.get("objective", {}), defaults.objective),
        alpha_gate=_parse_alpha_gate(raw.get("alpha_gate", {}), defaults.alpha_gate),
        memory_bank=_parse_memory_bank(raw.get("memory_bank", {}), defaults.memory_bank),
        telemetry=_parse_telemetry(raw.get("telemetry", {}), defaults.telemetry),
        artifacts=_parse_artifacts(raw.get("artifacts", {}), defaults.artifacts),
        runtime=_parse_runtime(raw.get("runtime", {}), defaults.runtime),
    )
    validate_train_config(config)
    return config
```

The `_parse_*` helpers must each call `_reject_unknown` with their exact allowed keys, copy missing values from the supplied typed defaults, construct the corresponding dataclass, and convert enum values. Nested helpers receive the selected profile's nested defaults in the same way. `_parse_dataset` accepts the operational preset keys (`id`, `name_or_path`, field names, split, limit, and chunk ID) plus the established metadata keys (`display_name`, `source`, `upstream_source`, `manifest_path`, `selection`, `train`, `eval`, and `corpus`); it stores the latter under `DatasetConfig.metadata`. Any other preset key is an error. `validate_train_config` checks all positive sizes, probabilities, beta values, bounds, dropout, and optimizer values and reports the full dotted field name.

Add deterministic serialization used by manifests and checkpoints:

```python
from dataclasses import asdict


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def train_config_to_dict(config: TrainConfig) -> dict[str, Any]:
    return _plain(asdict(config))
```

- [ ] **Step 4: Run configuration tests**

Run:

```bash
conda run -n justatom pytest tests/test_training_config.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add justatom/training/config.py tests/test_training_config.py
git commit -m "feat: add typed training configuration"
```

---

### Task 4: Resolve The Three Method Profiles

**Files:**
- Create: `justatom/training/methods.py`
- Create: `tests/test_training_methods.py`

**Interfaces:**
- Consumes: `TrainConfig` from Task 3.
- Produces: `resolve_method(config: TrainConfig) -> TrainConfig`, `canonical_method_config(method: TrainingMethod) -> TrainConfig`, and method invariant errors.

- [ ] **Step 1: Write failing canonical-profile tests**

```python
from dataclasses import replace

import pytest

from justatom.training.config import (
    ExperimentConfig,
    ExperimentRole,
    MarginMode,
    TrainingMethod,
)
from justatom.training.methods import canonical_method_config, resolve_method


def test_canonical_profiles_have_exact_structural_components():
    vanilla = canonical_method_config(TrainingMethod.VANILLA)
    gate = canonical_method_config(TrainingMethod.ATOM_GATE)
    atomic = canonical_method_config(TrainingMethod.ATOMIC)

    assert not vanilla.alpha_gate.enabled and not vanilla.memory_bank.enabled
    assert gate.alpha_gate.enabled and not gate.memory_bank.enabled
    assert atomic.alpha_gate.enabled and atomic.memory_bank.enabled
    assert atomic.memory_bank.adaptive.enabled
    assert atomic.memory_bank.margin.mode is MarginMode.QUERY


def test_canonical_atom_gate_rejects_bank():
    gate = canonical_method_config(TrainingMethod.ATOM_GATE)
    invalid = replace(gate, memory_bank=canonical_method_config(TrainingMethod.ATOMIC).memory_bank)
    with pytest.raises(ValueError, match="atom_gate.*memory_bank"):
        resolve_method(invalid)


def test_atomic_fixed_margin_requires_ablation_role():
    atomic = canonical_method_config(TrainingMethod.ATOMIC)
    constant = replace(
        atomic,
        memory_bank=replace(
            atomic.memory_bank,
            margin=replace(atomic.memory_bank.margin, mode=MarginMode.CONSTANT),
        ),
    )
    with pytest.raises(ValueError, match="experiment.role"):
        resolve_method(constant)

    ablation = replace(
        constant,
        experiment=ExperimentConfig(role=ExperimentRole.ABLATION, seed=42),
    )
    assert resolve_method(ablation).memory_bank.margin.mode is MarginMode.CONSTANT
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
conda run -n justatom pytest tests/test_training_methods.py -q
```

Expected: collection fails because `justatom.training.methods` does not exist.

- [ ] **Step 3: Implement explicit canonical profiles**

Use `dataclasses.replace`; do not encode method defaults in shell or API layers:

```python
from __future__ import annotations

from dataclasses import replace

from justatom.training.config import (
    AdaptiveBankConfig,
    AlphaGateConfig,
    MarginConfig,
    MarginMode,
    MemoryBankConfig,
    ObjectiveConfig,
    TrainConfig,
    TrainingMethod,
)


def canonical_method_config(method: TrainingMethod) -> TrainConfig:
    base = TrainConfig(method=method)
    if method is TrainingMethod.VANILLA:
        return replace(
            base,
            objective=replace(base.objective, simcse_dropout_weight=0.0),
            alpha_gate=AlphaGateConfig(enabled=False),
            memory_bank=MemoryBankConfig(enabled=False, size=0),
        )
    gate = replace(
        base,
        objective=replace(base.objective, simcse_dropout_weight=0.1),
        alpha_gate=AlphaGateConfig(enabled=True, mix_weight=0.3),
        memory_bank=MemoryBankConfig(enabled=False, size=0),
    )
    if method is TrainingMethod.ATOM_GATE:
        return gate
    return replace(
        gate,
        memory_bank=MemoryBankConfig(
            enabled=True,
            size=512,
            warmup_steps=50,
            mining="mixed",
            hard_negatives=4,
            random_negatives=12,
            hard_warmup_steps=120,
            hard_ramp_steps=200,
            adaptive=AdaptiveBankConfig(
                enabled=True,
                collision_threshold=0.0,
                collision_beta=0.05,
            ),
            margin=MarginConfig(
                mode=MarginMode.QUERY,
                base=0.05,
                scale=0.02,
                minimum=0.0,
                maximum=0.15,
                admission_beta=0.05,
                regularization_weight=50.0,
            ),
        ),
    )
```

Implement `resolve_method` so canonical runs enforce the exact structural boundary. Ablation runs may change structural toggles, but still reject impossible combinations such as query margin without a bank or alpha enabled for `vanilla`.

- [ ] **Step 4: Run tests**

```bash
conda run -n justatom pytest tests/test_training_config.py tests/test_training_methods.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add justatom/training/methods.py tests/test_training_methods.py
git commit -m "feat: define canonical training methods"
```

---

### Task 5: Extract Query-Only Alpha Gate

**Files:**
- Create: `justatom/training/alpha_gate.py`
- Create: `tests/test_alpha_gate.py`

**Interfaces:**
- Consumes: query embeddings shaped `[B, D]` and `AlphaGateConfig`.
- Produces: `QueryAlphaGate.forward(queries) -> Tensor[B]`, `parameters_for_optimizer()`, and serializable metadata.

- [ ] **Step 1: Write failing alpha tests**

```python
import torch

from justatom.training.alpha_gate import QueryAlphaGate
from justatom.training.config import AlphaGateConfig, AlphaHeadConfig


def test_query_alpha_gate_returns_one_value_per_query():
    gate = QueryAlphaGate(
        embedding_dim=4,
        config=AlphaGateConfig(
            enabled=True,
            head=AlphaHeadConfig(layers=1, hidden_dim=8, dropout=0.0, activation="gelu"),
        ),
    )
    queries = torch.randn(3, 4, requires_grad=True)
    alpha = gate(queries)
    assert alpha.shape == (3,)
    assert torch.all((alpha > 0.0) & (alpha < 1.0))


def test_query_alpha_gate_backpropagates_from_per_query_weights():
    torch.manual_seed(0)
    gate = QueryAlphaGate(embedding_dim=4, config=AlphaGateConfig(enabled=True))
    queries = torch.randn(3, 4, requires_grad=True)
    loss = ((1.0 - gate(queries)) * torch.tensor([1.0, 2.0, 3.0])).mean()
    loss.backward()
    assert queries.grad is not None and torch.isfinite(queries.grad).all()
    assert all(parameter.grad is not None for parameter in gate.parameters())


def test_query_alpha_gate_rejects_document_argument_by_signature():
    gate = QueryAlphaGate(embedding_dim=4, config=AlphaGateConfig(enabled=True))
    queries = torch.randn(2, 4)
    documents = torch.randn(2, 4)
    try:
        gate(queries, documents)
    except TypeError:
        pass
    else:
        raise AssertionError("QueryAlphaGate must accept query embeddings only")
```

- [ ] **Step 2: Run and verify failure**

```bash
conda run -n justatom pytest tests/test_alpha_gate.py -q
```

Expected: collection fails because the module is missing.

- [ ] **Step 3: Implement the focused head**

```python
from __future__ import annotations

import torch
from torch import nn

from justatom.training.config import AlphaGateConfig


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


class QueryAlphaGate(nn.Module):
    def __init__(self, embedding_dim: int, config: AlphaGateConfig):
        super().__init__()
        if not config.enabled:
            raise ValueError("QueryAlphaGate requires alpha_gate.enabled=true")
        hidden_dim = config.head.hidden_dim or max(32, min(256, embedding_dim // 2))
        activation = _ACTIVATIONS[config.head.activation]
        layers: list[nn.Module] = []
        current_dim = embedding_dim
        for _ in range(config.head.layers):
            layers.extend([nn.Linear(current_dim, hidden_dim), activation()])
            if config.head.dropout > 0.0:
                layers.append(nn.Dropout(config.head.dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)
        self.config = config

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        if queries.ndim != 2:
            raise ValueError(f"queries must have shape [batch, dim], got {tuple(queries.shape)}")
        return torch.sigmoid(self.network(queries)).squeeze(-1)

    def metadata(self) -> dict[str, object]:
        return {
            "input": "query",
            "layers": self.config.head.layers,
            "hidden_dim": self.config.head.hidden_dim,
            "dropout": self.config.head.dropout,
            "activation": self.config.head.activation,
        }
```

- [ ] **Step 4: Run alpha and golden tests**

```bash
conda run -n justatom pytest tests/test_alpha_gate.py tests/test_training_golden.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add justatom/training/alpha_gate.py tests/test_alpha_gate.py
git commit -m "feat: extract query alpha gate"
```

---

### Task 6: Narrow The Memory Bank And Add Query Margin

**Files:**
- Modify: `justatom/training/memory_bank.py`
- Create: `tests/test_memory_bank.py`
- Modify: `tests/test_soft_contrastive_loss.py`

**Interfaces:**
- Consumes: detached positive document vectors, current query/positive vectors, key IDs, `MemoryBankConfig`.
- Produces: `MemorySelection`, `QueryMarginHead`, and `ContrastiveMemoryBank.select(...)`.

- [ ] **Step 1: Write typed selection and margin tests**

```python
import torch
import torch.nn.functional as F

from justatom.training.config import MarginConfig, MarginMode, MemoryBankConfig
from justatom.training.memory_bank import ContrastiveMemoryBank, MemorySelection, QueryMarginHead


def test_query_margin_starts_at_base_and_has_live_gradient():
    head = QueryMarginHead(
        embedding_dim=4,
        config=MarginConfig(
            mode=MarginMode.QUERY,
            base=0.05,
            scale=0.02,
            minimum=0.0,
            maximum=0.15,
            regularization_weight=50.0,
        ),
    )
    queries = F.normalize(torch.randn(3, 4), dim=-1).requires_grad_()
    raw, margin = head(queries)
    torch.testing.assert_close(raw, torch.full((3,), 0.05), atol=1e-7, rtol=0.0)
    torch.testing.assert_close(margin, torch.full((3,), 0.05), atol=1e-7, rtol=0.0)
    (margin.sum() + raw.sum()).backward()
    assert queries.grad is not None


def test_memory_bank_returns_noop_selection_before_warmup():
    bank = ContrastiveMemoryBank(
        MemoryBankConfig(enabled=True, size=8, warmup_steps=2)
    )
    selection = bank.select(
        batch={"doc_key_id": torch.tensor([1, 2])},
        query_vectors=F.normalize(torch.eye(2), dim=-1),
        positive_vectors=F.normalize(torch.eye(2), dim=-1),
        step=0,
    )
    assert isinstance(selection, MemorySelection)
    assert selection.embeddings is None
    assert selection.active_mask is None


def test_memory_bank_enqueue_always_detaches_graph():
    bank = ContrastiveMemoryBank(MemoryBankConfig(enabled=True, size=4))
    vectors = torch.randn(2, 3, requires_grad=True)
    bank.enqueue(vectors, {"doc_key_id": torch.tensor([1, 2])})
    assert bank.embeddings is not None
    assert not bank.embeddings.requires_grad
    assert bank.embeddings.grad_fn is None
```

- [ ] **Step 2: Run and verify failure**

```bash
conda run -n justatom pytest tests/test_memory_bank.py -q
```

Expected: tests fail because the constructor and typed result are not implemented.

- [ ] **Step 3: Introduce the typed result and query margin**

Add:

```python
from dataclasses import dataclass

from torch import nn

from justatom.training.config import MarginConfig, MarginMode, MemoryBankConfig


@dataclass(frozen=True)
class MemorySelection:
    embeddings: torch.Tensor | None
    active_mask: torch.Tensor | None
    log_weights: torch.Tensor | None
    collision_g: torch.Tensor | None
    hard_weights: torch.Tensor | None
    metrics: dict[str, float | torch.Tensor | str]


class QueryMarginHead(nn.Module):
    def __init__(self, embedding_dim: int, config: MarginConfig):
        super().__init__()
        if config.mode is not MarginMode.QUERY:
            raise ValueError("QueryMarginHead requires margin.mode=query")
        hidden_dim = max(32, min(256, embedding_dim // 2))
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.config = config

    def forward(self, queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.config.scale * torch.tanh(self.network(queries)).squeeze(-1)
        raw = self.config.base + delta
        return raw, raw.clamp(self.config.minimum, self.config.maximum)
```

Change `ContrastiveMemoryBank.__init__` to accept one `MemoryBankConfig`. Rename `get` to `select`, preserve current FIFO, collision, hard/random mining, and soft row-weight formulas, and return `MemorySelection`. Keep query and positive embeddings detached inside bank selection; only `QueryMarginHead` and objective admission use a live graph.

- [ ] **Step 4: Move old bank tests to the focused file and run parity**

Move every `test_memory_bank_*` test from `tests/test_soft_contrastive_loss.py` into `tests/test_memory_bank.py`, adapting only constructor/result access. Do not change expected masks, schedules, metrics, or formulas.

Run:

```bash
conda run -n justatom pytest \
  tests/test_memory_bank.py \
  tests/test_training_golden.py \
  tests/test_soft_contrastive_loss.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add justatom/training/memory_bank.py tests/test_memory_bank.py tests/test_soft_contrastive_loss.py
git commit -m "refactor: isolate adaptive memory bank"
```

---

### Task 7: Extract The Canonical Objective

**Files:**
- Create: `justatom/training/objective.py`
- Create: `tests/test_training_objective.py`
- Modify: `justatom/training/loss.py`
- Modify: `tests/test_training_golden.py`

**Interfaces:**
- Consumes: `ObjectiveInputs` with live embeddings, optional alpha, optional memory selection, optional margin, and lexical pair scores.
- Produces: `ObjectiveOutput(loss, main_per_row, metrics)`.

- [ ] **Step 1: Write failing objective composition tests**

```python
import torch
import torch.nn.functional as F

from justatom.training.config import AlphaGateConfig, MarginConfig, MarginMode, ObjectiveConfig
from justatom.training.memory_bank import MemorySelection
from justatom.training.objective import ContrastiveObjective, ObjectiveInputs


def test_objective_vanilla_has_no_auxiliary_components():
    objective = ContrastiveObjective(ObjectiveConfig(temperature=1.0, learnable_temperature=False))
    q = F.normalize(torch.eye(3), dim=-1).requires_grad_()
    p = q.detach().clone().requires_grad_()
    output = objective(ObjectiveInputs(queries=q, positives=p))
    assert output.loss.ndim == 0
    assert output.metrics["loss/alpha_aux"] == 0.0
    assert output.metrics["loss/memory_margin_regularization"] == 0.0


def test_objective_atom_gate_uses_augment_formula():
    objective = ContrastiveObjective(
        ObjectiveConfig(
            temperature=1.0,
            learnable_temperature=False,
            simcse_dropout_weight=0.1,
        )
    )
    q = F.normalize(torch.randn(3, 4), dim=-1).requires_grad_()
    p = F.normalize(torch.randn(3, 4), dim=-1).requires_grad_()
    q_alt = F.normalize(torch.randn(3, 4), dim=-1).requires_grad_()
    alpha = torch.tensor([0.2, 0.5, 0.8], requires_grad=True)
    output = objective(ObjectiveInputs(queries=q, positives=p, query_alt=q_alt, alpha=alpha))
    expected = (output.main_per_row + (1.0 - alpha) * 0.1 * output.simcse_per_row).mean()
    torch.testing.assert_close(output.loss, expected)


def test_objective_regularizes_raw_margin_to_constant_base():
    config = MarginConfig(
        mode=MarginMode.QUERY,
        base=0.05,
        scale=0.02,
        regularization_weight=50.0,
    )
    objective = ContrastiveObjective(ObjectiveConfig(temperature=1.0, learnable_temperature=False))
    q = F.normalize(torch.randn(3, 4), dim=-1)
    p = F.normalize(torch.randn(3, 4), dim=-1)
    raw = torch.tensor([0.04, 0.05, 0.07], requires_grad=True)
    output = objective(
        ObjectiveInputs(queries=q, positives=p, margin=raw.clamp(0.0, 0.15), raw_margin=raw),
        margin_config=config,
    )
    expected_regularization = 50.0 * (raw - 0.05).pow(2).mean()
    torch.testing.assert_close(
        output.metrics["loss/memory_margin_regularization_tensor"],
        expected_regularization,
    )
```

- [ ] **Step 2: Run and verify failure**

```bash
conda run -n justatom pytest tests/test_training_objective.py -q
```

Expected: collection fails because the module is missing.

- [ ] **Step 3: Implement typed objective input and output**

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from justatom.training.config import MarginConfig, ObjectiveConfig
from justatom.training.loss import ContrastiveLoss
from justatom.training.memory_bank import MemorySelection


@dataclass(frozen=True)
class ObjectiveInputs:
    queries: torch.Tensor
    positives: torch.Tensor
    query_alt: torch.Tensor | None = None
    alpha: torch.Tensor | None = None
    memory: MemorySelection | None = None
    margin: torch.Tensor | None = None
    raw_margin: torch.Tensor | None = None
    semantic_pair_scores: torch.Tensor | None = None
    lexical_pair_scores: torch.Tensor | None = None
    alpha_mix_weight: float = 0.0
    alpha_entropy_weight: float = 0.0


@dataclass(frozen=True)
class ObjectiveOutput:
    loss: torch.Tensor
    main_per_row: torch.Tensor
    simcse_per_row: torch.Tensor | None
    metrics: dict[str, float | torch.Tensor]


class ContrastiveObjective(nn.Module):
    def __init__(self, config: ObjectiveConfig):
        super().__init__()
        self.config = config
        self.kernel = ContrastiveLoss(
            temperature=config.temperature,
            reduction="none",
            learnable_temperature=config.learnable_temperature,
            decoupled=config.decoupled,
        )

    def forward(
        self,
        inputs: ObjectiveInputs,
        *,
        margin_config: MarginConfig | None = None,
    ) -> ObjectiveOutput:
        memory = inputs.memory
        main = self.kernel.info_nce(
            inputs.queries,
            inputs.positives,
            reduction="none",
            memory_negatives=None if memory is None else memory.embeddings,
            memory_negative_mask=None if memory is None else memory.active_mask,
            memory_log_weights=None if memory is None else memory.log_weights,
            memory_margin=inputs.margin,
            memory_soft_beta=None if margin_config is None else margin_config.admission_beta,
        )
        simcse = None
        per_row = main
        if inputs.query_alt is not None and self.config.simcse_dropout_weight > 0.0:
            simcse = self.kernel.simcse_term(inputs.queries, inputs.query_alt, reduction="none")
            weighted = self.config.simcse_dropout_weight * simcse
            per_row = per_row + weighted if inputs.alpha is None else per_row + (1.0 - inputs.alpha) * weighted
        loss = per_row.mean()
        metrics: dict[str, float | torch.Tensor] = {
            "loss/main": main.detach().mean(),
            "loss/alpha_aux": 0.0 if simcse is None else simcse.detach().mean(),
            "loss/memory_margin_regularization": 0.0,
        }
        if inputs.alpha is not None and inputs.semantic_pair_scores is not None and inputs.lexical_pair_scores is not None:
            alpha = inputs.alpha.view(-1, 1)
            mixed_pair = alpha * inputs.semantic_pair_scores + (1.0 - alpha) * inputs.lexical_pair_scores
            positive_distance = 1.0 - mixed_pair[:, 0]
            negative_distance = 1.0 - mixed_pair[:, 1]
            positive_loss = 0.5 * positive_distance.pow(2)
            negative_loss = 0.5 * torch.relu(self.config.pairwise_margin - negative_distance).pow(2)
            mix_loss = (positive_loss + negative_loss).mean()
            loss = loss + inputs.alpha_mix_weight * mix_loss
            metrics["loss/lexical_mix"] = mix_loss.detach()
            alpha_safe = inputs.alpha.clamp(1e-6, 1.0 - 1e-6)
            entropy = -(
                alpha_safe * alpha_safe.log()
                + (1.0 - alpha_safe) * (1.0 - alpha_safe).log()
            ).mean()
            entropy_bonus = inputs.alpha_entropy_weight * entropy
            loss = loss - entropy_bonus
            metrics["loss/alpha_entropy_bonus"] = entropy_bonus.detach()
        if margin_config is not None and inputs.raw_margin is not None:
            regularization = margin_config.regularization_weight * (
                inputs.raw_margin - margin_config.base
            ).pow(2).mean()
            loss = loss + regularization
            metrics["loss/memory_margin_regularization"] = regularization.detach()
            metrics["loss/memory_margin_regularization_tensor"] = regularization
        return ObjectiveOutput(loss=loss, main_per_row=main, simcse_per_row=simcse, metrics=metrics)
```

Keep the existing alpha mix-weight warmup semantics exactly in the training module before constructing `ObjectiveInputs`. Preserve soft false-negative attraction only as an explicit `ObjectiveConfig` ablation field; canonical defaults keep it at `0.0`.

- [ ] **Step 4: Remove query-temperature input from the contrastive kernel**

Delete `tau_per_query` from `ContrastiveLoss.info_nce`. Always use `self.tau`. Update the docstring and tests so query-conditional temperature has no active or compatibility path.

- [ ] **Step 5: Run objective and golden tests**

```bash
conda run -n justatom pytest \
  tests/test_training_objective.py \
  tests/test_training_golden.py \
  tests/test_soft_contrastive_loss.py -q
```

Expected: all tests pass, including live gradient checks for encoder, alpha, temperature, and margin.

- [ ] **Step 6: Commit**

```bash
git add \
  justatom/training/objective.py \
  justatom/training/loss.py \
  tests/test_training_objective.py \
  tests/test_training_golden.py
git commit -m "refactor: isolate atomic objective"
```

---

### Task 8: Extract Sampling And Telemetry

**Files:**
- Create: `justatom/training/sampling.py`
- Create: `justatom/training/telemetry.py`
- Modify: `tests/test_safe_negative_sampling.py`
- Create: `tests/test_training_telemetry.py`
- Modify: `justatom/training/diagnostics.py`

**Interfaces:**
- Consumes: batch IDs/text, embeddings, losses, component gradients, and bank metrics.
- Produces: `sample_safe_negative_indices(...)`, `scalar_distribution(...)`, `batch_retrieval_metrics(...)`, `grad_norm(...)`, and `resolve_metric_tensors(...)`.

- [ ] **Step 1: Redirect sampling tests to the future module and verify failure**

Change:

```python
from justatom.running.trainer import _sample_safe_negative_indices
```

to:

```python
from justatom.training.sampling import sample_safe_negative_indices
```

and update calls to the public name.

Run:

```bash
conda run -n justatom pytest tests/test_safe_negative_sampling.py -q
```

Expected: collection fails because `justatom.training.sampling` is missing.

- [ ] **Step 2: Move sampling without changing behavior**

Move `_sample_negative_derangement`, `_sample_safe_negative_indices`, and `_inverse_idf_recall` from `justatom/running/trainer.py` to `justatom/training/sampling.py`. Rename them without leading underscores and preserve all key-collision and lexical-overlap behavior.

Run:

```bash
conda run -n justatom pytest tests/test_safe_negative_sampling.py -q
```

Expected: all sampling tests pass.

- [ ] **Step 3: Write telemetry tests**

```python
import math

import torch

from justatom.training.telemetry import batch_retrieval_metrics, grad_norm, scalar_distribution


def test_scalar_distribution_has_stable_quantiles():
    metrics = scalar_distribution("alpha", torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert metrics["alpha/mean"] == 0.5
    assert metrics["alpha/p50"] == 0.5
    assert metrics["alpha/min"] == 0.0
    assert metrics["alpha/max"] == 1.0


def test_batch_retrieval_metrics_uses_diagonal_as_positive():
    scores = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
    metrics = batch_retrieval_metrics(scores)
    assert metrics["batch/hit_rate_at_1"] == 1.0
    assert metrics["batch/mrr"] == 1.0


def test_grad_norm_returns_zero_without_gradients():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    assert grad_norm([parameter]) == 0.0
```

- [ ] **Step 4: Implement telemetry helpers and remove duplicates**

Move `_scalar_distribution_metrics`, `_batch_retrieval_metrics`, `_grad_norm`, and `_resolve_metric_tensors` from the old trainer into `justatom/training/telemetry.py`. Move memory-bank distribution helpers there as well, using stable slash-separated names. Keep `embedding_geometry_metrics` in `diagnostics.py` and make it return tensor scalars until telemetry detaches them.

- [ ] **Step 5: Run tests**

```bash
conda run -n justatom pytest \
  tests/test_safe_negative_sampling.py \
  tests/test_training_telemetry.py \
  tests/test_memory_bank.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add \
  justatom/training/sampling.py \
  justatom/training/telemetry.py \
  justatom/training/diagnostics.py \
  tests/test_safe_negative_sampling.py \
  tests/test_training_telemetry.py \
  tests/test_memory_bank.py
git commit -m "refactor: separate sampling and telemetry"
```

---

### Task 9: Build One Compositional Lightning Module

**Files:**
- Create: `justatom/training/module.py`
- Modify: `justatom/running/encoders.py`
- Create: `tests/test_training_module.py`

**Interfaces:**
- Consumes: encoder runner, `TrainConfig`, `ContrastiveObjective`, optional `QueryAlphaGate`, optional `ContrastiveMemoryBank`, optional `QueryMarginHead`, lexical lookup.
- Produces: `ContrastiveTrainingModule` with one training path and explicit artifact state.

- [ ] **Step 1: Write failing construction and optimizer tests**

```python
import torch
from torch import nn

from justatom.training.alpha_gate import QueryAlphaGate
from justatom.training.methods import canonical_method_config
from justatom.training.module import ContrastiveTrainingModule
from justatom.training.config import TrainingMethod


class TinyEncoder(nn.Module):
    output_dims = 4

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def encode_pair(self, batch):
        return (
            torch.nn.functional.normalize(self.projection(batch["queries"]), dim=-1),
            torch.nn.functional.normalize(self.projection(batch["documents"]), dim=-1),
        )

    def encode_queries(self, batch):
        return torch.nn.functional.normalize(self.projection(batch["queries"]), dim=-1)


def test_module_constructs_only_components_required_by_method():
    vanilla = ContrastiveTrainingModule.build(TinyEncoder(), canonical_method_config(TrainingMethod.VANILLA))
    gate = ContrastiveTrainingModule.build(TinyEncoder(), canonical_method_config(TrainingMethod.ATOM_GATE))
    atomic = ContrastiveTrainingModule.build(TinyEncoder(), canonical_method_config(TrainingMethod.ATOMIC))
    assert vanilla.alpha_gate is None and vanilla.memory_bank is None and vanilla.margin_head is None
    assert isinstance(gate.alpha_gate, QueryAlphaGate) and gate.memory_bank is None
    assert atomic.alpha_gate is not None and atomic.memory_bank is not None and atomic.margin_head is not None


def test_optimizer_contains_encoder_temperature_alpha_and_margin_parameters_once():
    module = ContrastiveTrainingModule.build(
        TinyEncoder(),
        canonical_method_config(TrainingMethod.ATOMIC),
    )
    optimizer = module.configure_optimizers()
    ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    expected = [id(parameter) for parameter in module.parameters() if parameter.requires_grad]
    assert sorted(ids) == sorted(expected)
    assert len(ids) == len(set(ids))
```

- [ ] **Step 2: Write the enqueue-order test**

```python
from justatom.training.objective import ObjectiveOutput


def _tiny_batch():
    return {
        "queries": torch.eye(2, 4),
        "documents": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        "doc_key_id": torch.tensor([1, 2]),
        "content_key_id": torch.tensor([11, 12]),
        "query_key_id": torch.tensor([21, 22]),
    }


def _finite_output():
    loss = torch.tensor(1.0, requires_grad=True)
    return ObjectiveOutput(
        loss=loss,
        main_per_row=torch.ones(2),
        simcse_per_row=None,
        metrics={},
    )


def test_atomic_enqueues_documents_after_objective(monkeypatch):
    module = ContrastiveTrainingModule.build(
        TinyEncoder(),
        canonical_method_config(TrainingMethod.ATOMIC),
    )
    events: list[str] = []
    monkeypatch.setattr(module, "_encode_dropout_query_view", lambda batch: module.encoder.encode_queries(batch))
    monkeypatch.setattr(module, "_semantic_pair_scores", lambda q, p, batch: torch.ones(2, 2))
    monkeypatch.setattr(module, "_lexical_pair_scores", lambda batch: torch.ones(2, 2))
    monkeypatch.setattr(module.objective, "forward", lambda *args, **kwargs: events.append("loss") or _finite_output())
    monkeypatch.setattr(module.memory_bank, "enqueue", lambda *args, **kwargs: events.append("enqueue"))
    module.compute_training_step(_tiny_batch(), step=0)
    assert events == ["loss", "enqueue"]
```

- [ ] **Step 3: Run and verify failure**

```bash
conda run -n justatom pytest tests/test_training_module.py -q
```

Expected: collection fails because the module is missing.

- [ ] **Step 4: Implement component construction and the training-step order**

Add focused embedding methods to `EncoderRunner` while retaining its existing generic `forward` for retrieval callers:

```python
def encode_pair(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    model_batch = {
        key: value
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "pos_input_ids", "pos_attention_mask", "group_ids"}
    }
    outputs = self(model_batch, norm=True)
    if len(outputs) != 2:
        raise RuntimeError(f"Expected query/document embeddings, got {len(outputs)} outputs")
    return outputs[0], outputs[1]


def encode_queries(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    query_batch = {key: batch[key] for key in ("input_ids", "attention_mask") if key in batch}
    outputs = self(query_batch, norm=True)
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one query embedding tensor, got {len(outputs)} outputs")
    return outputs[0]
```

`ContrastiveTrainingModule.build` must be the only place that turns a resolved method config into optional train-time components. `compute_training_step` must perform this exact order:

```python
q, p = self.encoder.encode_pair(batch)
alpha = None if self.alpha_gate is None else self.alpha_gate(q)
query_alt = self._encode_dropout_query_view(batch) if self._needs_simcse else None
selection = None if self.memory_bank is None else self.memory_bank.select(
    batch=batch,
    query_vectors=q,
    positive_vectors=p,
    step=step,
)
raw_margin, margin = (None, None) if self.margin_head is None else self.margin_head(q)
output = self.objective(
    ObjectiveInputs(
        queries=q,
        positives=p,
        query_alt=query_alt,
        alpha=alpha,
        memory=selection,
        raw_margin=raw_margin,
        margin=margin,
        semantic_pair_scores=self._semantic_pair_scores(q, p, batch),
        lexical_pair_scores=self._lexical_pair_scores(batch),
        alpha_mix_weight=self._effective_alpha_mix_weight(step),
        alpha_entropy_weight=self.config.alpha_gate.entropy_weight,
    ),
    margin_config=self.config.memory_bank.margin if self.memory_bank is not None else None,
)
if not torch.isfinite(output.loss.detach()):
    raise RuntimeError(f"Non-finite loss at step={step}: {output.loss.detach().cpu().item()}")
if self.memory_bank is not None:
    self.memory_bank.enqueue(p, batch)
return output
```

Use the baseline accumulation and alpha warmup formulas verbatim:

```python
def _effective_alpha_mix_weight(self, step: int) -> float:
    target = float(self.config.alpha_gate.mix_weight)
    warmup = int(self.config.alpha_gate.mix_weight_warmup_steps)
    if warmup <= 0:
        return target
    progress = min(max(float(step), 0.0) / float(warmup), 1.0)
    return target * progress


def _adjust_loss_for_accumulation(self, loss: torch.Tensor) -> torch.Tensor:
    return loss / max(int(self.config.optimization.grad_acc_steps), 1)


def _is_accumulation_start(self, batch_idx: int) -> bool:
    return batch_idx % max(int(self.config.optimization.grad_acc_steps), 1) == 0


def _should_step_optimizer(self, batch_idx: int) -> bool:
    steps = max(int(self.config.optimization.grad_acc_steps), 1)
    return ((batch_idx + 1) % steps == 0) or bool(getattr(self.trainer, "is_last_batch", False))
```

After every optimizer step call `self.objective.kernel.clamp_temperature_()`. Build optimizer groups with each parameter ID present exactly once:

```python
def configure_optimizers(self):
    seen: set[int] = set()

    def unique(parameters):
        selected = []
        for parameter in parameters:
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                selected.append(parameter)
        return selected

    groups = []
    encoder_parameters = unique(self.encoder.parameters())
    if encoder_parameters:
        groups.append(
            {
                "params": encoder_parameters,
                "lr": self.config.optimization.lr_encoder,
                "weight_decay": self.config.optimization.weight_decay,
            }
        )
    head_modules = [module for module in (self.alpha_gate, self.margin_head) if module is not None]
    head_parameters = unique(parameter for module in head_modules for parameter in module.parameters())
    if head_parameters:
        groups.append({"params": head_parameters, "lr": self.config.optimization.lr_heads, "weight_decay": 0.0})
    objective_parameters = unique(self.objective.parameters())
    if objective_parameters:
        groups.append(
            {"params": objective_parameters, "lr": self.config.optimization.lr_encoder, "weight_decay": 0.0}
        )
    return torch.optim.AdamW(groups)
```

Add distinct artifact methods:

```python
def save_deployable_encoder(self, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    self.encoder.save(str(destination))
    return destination


def save_research_checkpoint(self, destination: Path) -> Path | None:
    if not self.config.artifacts.save_research_checkpoint:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    optimizer_states = [optimizer.state_dict() for optimizer in getattr(self.trainer, "optimizers", [])]
    torch.save(
        {
            "schema_version": 1,
            "resolved_config": train_config_to_dict(self.config),
            "state_dict": self.state_dict(),
            "optimizer_states": optimizer_states,
            "epoch": int(self.current_epoch),
            "global_step": int(self.global_step),
        },
        temporary,
    )
    temporary.replace(destination)
    return destination
```

Load it through a strict classmethod:

```python
@classmethod
def load_research_checkpoint(
    cls,
    path: Path,
    *,
    encoder: nn.Module,
    lexical_lookup: dict[str, str] | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple["ContrastiveTrainingModule", list[dict[str, object]]]:
    payload = torch.load(path, map_location=map_location)
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported research checkpoint schema: {payload.get('schema_version')!r}")
    config = resolve_method(parse_train_config(payload["resolved_config"]))
    module = cls.build(encoder, config, lexical_lookup=lexical_lookup)
    module.load_state_dict(payload["state_dict"], strict=True)
    return module, list(payload.get("optimizer_states", []))
```

The deployable save path never serializes alpha or margin heads.

- [ ] **Step 5: Port focused trainer tests**

Move the existing tests for optimizer inclusion, accumulation boundaries, non-finite loss, margin regularization, alpha gradients, and memory enqueue into `tests/test_training_module.py`. Delete tests that assert gamma-only or encoder-only class selection.

- [ ] **Step 6: Run module and mathematical tests**

```bash
conda run -n justatom pytest \
  tests/test_training_module.py \
  tests/test_training_objective.py \
  tests/test_training_golden.py \
  tests/test_alpha_gate.py \
  tests/test_memory_bank.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add justatom/training/module.py justatom/running/encoders.py tests/test_training_module.py tests/test_soft_contrastive_loss.py
git commit -m "feat: add compositional training module"
```

---

### Task 10: Move Data Preparation And Add Reproducible Training Job

**Files:**
- Create: `justatom/training/data.py`
- Create: `justatom/training/job.py`
- Create: `tests/test_training_job.py`
- Modify: `tests/test_train_data_preparation.py`
- Modify: `justatom/tooling/collections.py`

**Interfaces:**
- Consumes: resolved `TrainConfig`, existing dataset adapters/processors, generic encoder loading, `ContrastiveTrainingModule`.
- Produces: `TrainingJob.run() -> TrainingResult`, `run_manifest.yaml`, deployable encoder directory, research checkpoint.

- [ ] **Step 1: Move data helpers under tests**

Move these functions from `justatom/api/train.py` into `justatom/training/data.py` without behavior changes:

```text
iterate_training_rows
rebalance_rows_by_content
count_batches_with_duplicate_content
sample_training_rows
prepare_training_data
```

Update imports in `tests/test_train_data_preparation.py` to use `justatom.training.data`.

Run:

```bash
conda run -n justatom pytest tests/test_train_data_preparation.py -q
```

Expected: all retained data tests pass; class-selection tests are removed because those classes are intentionally retired.

- [ ] **Step 2: Write manifest and artifact tests**

```python
from pathlib import Path

import yaml

from justatom.training.config import TrainingMethod
from justatom.training.job import RunManifest, write_run_manifest
from justatom.training.methods import canonical_method_config


def test_manifest_contains_resolved_method_seed_and_git_state(tmp_path: Path):
    config = canonical_method_config(TrainingMethod.ATOMIC)
    manifest = RunManifest.from_config(config, git_commit="abc123", git_dirty=True)
    path = write_run_manifest(tmp_path, manifest)
    loaded = yaml.safe_load(path.read_text())
    assert loaded["schema_version"] == 1
    assert loaded["method"] == "atomic"
    assert loaded["experiment"]["seed"] == 42
    assert loaded["git_commit"] == "abc123"
    assert loaded["git_dirty"] is True
    assert loaded["resolved_config"]["memory_bank"]["margin"]["mode"] == "query"


def test_artifact_directories_are_distinct(tmp_path: Path):
    from justatom.training.job import artifact_paths

    paths = artifact_paths(tmp_path)
    assert paths.deployable_encoder == tmp_path / "encoder"
    assert paths.research_checkpoint == tmp_path / "research" / "checkpoint.pt"
    assert paths.manifest == tmp_path / "run_manifest.yaml"
```

- [ ] **Step 3: Implement manifest serialization**

Use `dataclasses.asdict`, converting enums to `.value` recursively. Capture git state with `subprocess.run(["git", ...], check=False, text=True, capture_output=True)` and do not fail training when Git metadata is unavailable.

Write YAML atomically:

```python
def write_run_manifest(run_dir: Path, manifest: RunManifest) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    destination = run_dir / "run_manifest.yaml"
    temporary = destination.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(manifest.to_dict(), sort_keys=False), encoding="utf-8")
    temporary.replace(destination)
    return destination
```

- [ ] **Step 4: Implement explicit artifact, loader, encoder, and Trainer factories**

Add these concrete boundaries:

```python
from dataclasses import dataclass
from pathlib import Path

import pytorch_lightning as L

from justatom.modeling.mask import ILanguageModel
from justatom.processing import ITokenizer, igniset
from justatom.processing.loader import NamedDataLoader
from justatom.processing.prime import TrainWithContrastiveProcessor
from justatom.running.encoders import EncoderRunner
from justatom.tooling.collections import resolve_artifact_dirname


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    deployable_encoder: Path
    research_checkpoint: Path
    manifest: Path


def artifact_paths(root: Path) -> ArtifactPaths:
    return ArtifactPaths(
        root=root,
        deployable_encoder=root / "encoder",
        research_checkpoint=root / "research" / "checkpoint.pt",
        manifest=root / "run_manifest.yaml",
    )


def resolve_run_dir(config: TrainConfig) -> Path:
    if config.artifacts.save_dir:
        return Path(config.artifacts.save_dir)
    dirname = resolve_artifact_dirname(config.artifacts.collection_name)
    return Path.cwd() / "weights" / dirname


def build_training_loader(config: TrainConfig):
    rows, lexical_lookup = prepare_training_data_from_config(config)
    tokenizer = ITokenizer.from_pretrained(config.model.name_or_path)
    processor = TrainWithContrastiveProcessor(
        tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
        max_query_seq_len=config.model.max_query_seq_len,
        queries_field="queries",
        queries_prefix=config.model.query_prefix,
        pos_queries_prefix=config.model.content_prefix,
    )
    dataset, tensor_names = igniset(
        rows,
        processor=processor,
        batch_size=config.optimization.batch_size,
        streaming=True,
    )
    loader = NamedDataLoader(
        dataset=dataset,
        tensor_names=tensor_names,
        batch_size=config.optimization.batch_size,
    )
    return loader, processor, lexical_lookup


def prepare_training_data_from_config(config: TrainConfig):
    _, rows, lexical_lookup = prepare_training_data(
        dataset_name_or_path=config.dataset.name_or_path,
        num_samples=config.optimization.num_samples,
        content_field=config.dataset.content_field,
        labels_field=config.dataset.labels_field,
        split=config.dataset.split,
        limit=config.dataset.limit,
        chunk_id_col=config.dataset.chunk_id_col,
        keywords_or_phrases_field=config.dataset.keywords_col,
        keywords_nested_col=config.dataset.keywords_nested_col,
        explanation_nested_col=config.dataset.explanation_nested_col,
        filters=config.filters.fields,
    )
    return rows, lexical_lookup


def resolve_torch_device(runtime: RuntimeConfig) -> str:
    if runtime.accelerator != "auto":
        return runtime.accelerator
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_encoder(config: TrainConfig, processor) -> EncoderRunner:
    language_model = ILanguageModel.load(model_name_or_path=config.model.name_or_path)
    return EncoderRunner(
        model=language_model,
        processor=processor,
        prediction_heads=[],
        device=resolve_torch_device(config.runtime),
    )


def build_lightning_trainer(config: TrainConfig):
    return L.Trainer(
        max_epochs=config.optimization.epochs,
        accelerator=config.runtime.accelerator,
        devices=config.runtime.devices,
        logger=build_training_logger(config),
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_model_summary=False,
    )


def build_training_logger(config: TrainConfig):
    if config.telemetry.backend == "csv":
        return False
    if config.telemetry.backend == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        return WandbLogger(
            project=config.telemetry.wandb_project,
            name=config.telemetry.wandb_run_name,
        )
    raise ValueError(f"Unsupported telemetry.backend={config.telemetry.backend!r}")
```

These factories preserve current row selection and CUDA, then MPS, then CPU device preference. CSV batch metrics remain owned by `TelemetryConfig` and `training/module.py`; Lightning's own logger is disabled for that backend.

- [ ] **Step 5: Implement the single training job**

`TrainingJob` receives a resolved config and injectable loader/model factories for unit tests:

```python
@dataclass
class TrainingResult:
    run_dir: Path
    encoder_dir: Path
    research_checkpoint: Path | None
    metrics_path: Path | None


class TrainingJob:
    def __init__(self, config: TrainConfig, *, loader_factory=build_training_loader, encoder_factory=load_encoder):
        self.config = config
        self.loader_factory = loader_factory
        self.encoder_factory = encoder_factory

    def run(self) -> TrainingResult:
        config = resolve_method(self.config)
        paths = artifact_paths(resolve_run_dir(config))
        write_run_manifest(paths.root, RunManifest.capture(config))
        write_collection_metadata_from_config(paths.root, config)
        loader, processor, lexical_lookup = self.loader_factory(config)
        encoder = self.encoder_factory(config, processor)
        module = ContrastiveTrainingModule.build(encoder, config, lexical_lookup=lexical_lookup)
        trainer = build_lightning_trainer(config)
        trainer.fit(module, train_dataloaders=loader)
        module.save_deployable_encoder(paths.deployable_encoder)
        checkpoint = module.save_research_checkpoint(paths.research_checkpoint)
        return TrainingResult(paths.root, paths.deployable_encoder, checkpoint, module.metrics_path)
```

The deployable encoder excludes alpha and margin heads. The research checkpoint stores module state, resolved config, schema version, epoch/global step, and optimizer state when resume support is enabled.

Implement collection metadata through the existing helpers:

```python
def write_collection_metadata_from_config(run_dir: Path, config: TrainConfig) -> None:
    if config.artifacts.collection_name is None:
        return
    metadata = build_collection_metadata(
        collection_name=config.artifacts.collection_name,
        collection_tag=config.artifacts.collection_tag,
        model_name_or_path=config.model.name_or_path,
        dataset_name_or_path=config.dataset.name_or_path,
        save_dir=run_dir,
        payload=train_config_to_dict(config),
    )
    write_collection_metadata(save_dir=run_dir, metadata=metadata)
```

- [ ] **Step 6: Add an injected tiny-job test**

Use a two-batch CPU loader and `TinyEncoder`; monkeypatch the Lightning trainer with a fake that records `fit`. Assert the manifest is written before `fit`, then assert both artifact paths are produced after fit.

- [ ] **Step 7: Run tests**

```bash
conda run -n justatom pytest \
  tests/test_training_job.py \
  tests/test_train_data_preparation.py \
  tests/test_training_module.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add \
  justatom/training/data.py \
  justatom/training/job.py \
  justatom/tooling/collections.py \
  tests/test_training_job.py \
  tests/test_train_data_preparation.py
git commit -m "feat: add reproducible training job"
```

---

### Task 11: Switch Config, API, Pipeline, And Benchmark

**Files:**
- Modify: `justatom/api/train.py`
- Modify: `configs/train.yaml`
- Modify: `justatom/builtins/configs/train.default.yaml`
- Modify: `scripts/run_pipeline.sh`
- Modify: `scripts/run_benchmark.sh`
- Modify: `tests/test_scenario_configs.py`
- Modify: `tests/test_benchmark_variants.py`
- Create: `tests/test_train_cli.py`

**Interfaces:**
- Consumes: scenario mapping plus dotted CLI overrides.
- Produces: thin `run(...)`, `--method`, canonical benchmark commands, no recipe aliases.

- [ ] **Step 1: Write failing CLI contract tests**

```python
import pytest

from justatom.api import train


def test_train_cli_accepts_only_method_and_dotted_overrides():
    parsed = train._parse_args(
        [
            "--config", "configs/train.yaml",
            "--method", "atomic",
            "--memory-bank.size", "256",
        ]
    )
    assert parsed["overrides"]["method"] == "atomic"
    assert parsed["overrides"]["memory_bank"]["size"] == 256


@pytest.mark.parametrize("retired", ["atom", "e_alpha_gate", "atom_gate_bank", "atom_gate_dynamic"])
def test_train_cli_rejects_retired_method_aliases(retired):
    with pytest.raises(SystemExit):
        train._parse_args(["--method", retired])
```

- [ ] **Step 2: Replace YAML with the typed schema**

Both default files must contain the same shared keys and a canonical neutral method. Method-controlled sections stay empty so selecting `--method atom_gate` or `--method atomic` cannot be overwritten by vanilla values from the common file:

```yaml
method: vanilla

experiment:
  role: canonical
  seed: 42

model:
  name_or_path: intfloat/multilingual-e5-small
  query_prefix: "query:"
  content_prefix: "passage:"
  max_query_seq_len: null
  max_seq_len: 512

dataset:
  id: null
  name_or_path: null
  labels_field: queries
  content_field: content
  split: null
  limit: null
  chunk_id_col: null
  keywords_col: keywords_or_phrases
  keywords_nested_col: null
  explanation_nested_col: null

filters:
  fields: null

optimization:
  optimizer: adamw
  lr_encoder: 0.00002
  lr_heads: 0.01
  weight_decay: 0.01
  batch_size: 32
  grad_acc_steps: 1
  epochs: 1
  num_samples: 100

# These sections override values from the selected method profile.
objective: {}
alpha_gate: {}
memory_bank: {}

telemetry:
  backend: csv
  metrics_path: null
  wandb_project: justatom
  wandb_run_name: null

artifacts:
  save_dir: null
  collection_name: null
  collection_tag: null
  save_research_checkpoint: true

runtime:
  accelerator: auto
  devices: auto
```

- [ ] **Step 3: Make `api/train.py` thin**

Retain only config loading, argument parsing, and dispatch:

```python
def run(*, config=None, config_path=None, overrides=None) -> TrainingResult:
    raw = load_scenario_config("train", config=config, config_path=config_path, overrides=overrides)
    typed = parse_train_config(raw)
    resolved = resolve_method(typed)
    return TrainingJob(resolved).run()
```

Use an explicit argparse choice:

```python
parser.add_argument("--method", choices=[method.value for method in TrainingMethod])
```

Merge `--method` into the dotted override mapping before parsing. Remove `_normalize_recipe_name`, `_normalize_atom_gate_config`, `_apply_alpha_gate_config`, `_cfg_to_train_kwargs`, job-class exports, and training-data implementations.

- [ ] **Step 4: Simplify the pipeline script**

Replace `--recipe`/`--tune-method` with:

```text
--method vanilla|atom_gate|atomic
```

Delete recipe canonicalization and all method-default blocks. Build the Python command with `--method "$METHOD"`, plus only explicitly provided low-level overrides. Keep dataset, model, baseline/evaluation, Weaviate, cache, and output-root behavior unchanged.

- [ ] **Step 5: Simplify benchmark variants**

Each variant maps one-to-one:

```bash
case "$variant" in
  vanilla|atom_gate|atomic)
    command+=(--method "$variant")
    ;;
  *)
    echo "invalid variant: $variant; expected vanilla,atom_gate,atomic" >&2
    exit 2
    ;;
esac
```

Remove aliases and variant-specific bank defaults; canonical defaults now come from `methods.py`.

- [ ] **Step 6: Run API/script tests**

```bash
conda run -n justatom pytest \
  tests/test_train_cli.py \
  tests/test_scenario_configs.py \
  tests/test_benchmark_variants.py \
  tests/test_training_config.py \
  tests/test_training_methods.py -q
```

Expected: all tests pass. Generated benchmark commands contain `--method vanilla`, `--method atom_gate`, or `--method atomic` and contain no `--recipe`.

- [ ] **Step 7: Commit**

```bash
git add \
  justatom/api/train.py \
  configs/train.yaml \
  justatom/builtins/configs/train.default.yaml \
  scripts/run_pipeline.sh \
  scripts/run_benchmark.sh \
  tests/test_train_cli.py \
  tests/test_scenario_configs.py \
  tests/test_benchmark_variants.py
git commit -m "refactor: expose three training methods"
```

---

### Task 12: Delete Legacy Training And Prove The Final Surface

**Files:**
- Delete: `justatom/running/trainer.py`
- Delete: `justatom/running/trainer_jobs.py`
- Modify: `justatom/running/encoders.py`
- Modify: `justatom/training/loss.py`
- Modify: `justatom/modeling/head.py` only if required by the final loss reference cleanup
- Delete or modify: `justatom/api/tune.py` according to the supported-entry-point reference check
- Modify: `tests/test_soft_contrastive_loss.py`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/research/atomic-experiments/README.md`

**Interfaces:**
- Consumes: the fully switched training path from Tasks 3-11.
- Produces: no active or compatibility implementation of retired modes, plus verified docs and smoke runs.

- [ ] **Step 1: Prove no supported code imports the old trainer modules**

Run:

```bash
rg -n "justatom\.running\.(trainer|trainer_jobs)|from justatom\.running\.trainer" \
  justatom tests scripts README.md docs
```

Expected: no production import remains. Any remaining test import is migrated to `justatom.training.*` before deletion.

- [ ] **Step 2: Delete the old trainer hierarchy**

Delete `justatom/running/trainer.py` and `justatom/running/trainer_jobs.py`. Remove their imports and `__all__` exports.

- [ ] **Step 3: Remove `GammaHybridRunner` and retired heads**

From `justatom/running/encoders.py`, remove:

```text
GammaHybridRunner
gamma1 / gamma2
gamma_joint
alpha(q,d+)
query diagonal head
tau(q) head
margin head
ALPHA_GATE_* reads
TAU_QUERY_* reads
MARGIN_QUERY_* reads
gamma_hybrid serialization
```

Retain `EncoderRunner` and `BiEncoderRunner`. Alpha and margin serialization now belongs to the research checkpoint in `training/module.py`; deployable inference uses the encoder only.

- [ ] **Step 4: Remove the old tune API if it is dead**

Run:

```bash
rg -n "api\.tune|justatom\|tune|python[^\n]*tune" \
  pyproject.toml Makefile README.md LAUNCH.md justatom scripts tests
```

Expected: no supported entry point. Delete `justatom/api/tune.py`. If a supported entry point exists, migrate that entry point to `justatom.api.train` in the same task and then delete the old file.

- [ ] **Step 5: Prune loss classes only when truly unreferenced**

Run:

```bash
rg -n "\b(FocalLoss|SoftContrastiveLoss|MultiMarginLoss|DiceLoss|TverskyLoss|TripletLoss|UMAPLoss)\b" \
  justatom tests
```

Delete classes used only by deleted training/tune paths. Retain classes still used by a supported modeling-head API. Keep the canonical `ContrastiveLoss` kernel used by `ContrastiveObjective`.

- [ ] **Step 6: Prove retired names and mathematical environment flags are gone**

Run:

```bash
rg -n "GammaOnly|EncoderGamma|EncoderOnlyLightning|BiGamma|UniGamma|GammaHybrid|gamma_joint|gamma1|gamma2|query_diagonal|tau_query|alpha\(q,d\+\)|ALPHA_GATE_|TAU_QUERY_|MARGIN_QUERY_|atom_gate_bank|atom_gate_dynamic|e_alpha_gate" \
  justatom configs scripts tests README.md docs/architecture.md
```

Expected: no matches except historical research documents explicitly outside the active architecture. Active code, configs, tests, README, and architecture docs have zero matches.

- [ ] **Step 7: Update architecture and user documentation**

Document:

```text
config -> resolve method -> TrainingJob -> ContrastiveTrainingModule
                                  |-> alpha(q) for atom_gate/atomic
                                  |-> adaptive bank + m(q) for atomic
```

Include one canonical command for each method and one explicitly labeled ablation command. State that alpha and margin heads are train-time controls and are absent from deployable encoder inference.

- [ ] **Step 8: Run the complete non-integration suite**

```bash
conda run -n justatom pytest -m "not integration" -q
```

Expected: all tests pass.

- [ ] **Step 9: Run three CPU/MPS smoke trains**

Use the built-in dataset and an isolated output root:

```bash
conda run -n justatom python -m justatom.api.train \
  --config configs/train.yaml \
  --method vanilla \
  --dataset.name-or-path justatom/builtin \
  --optimization.num-samples 64 \
  --optimization.batch-size 8 \
  --optimization.epochs 1 \
  --artifacts.save-dir .tmp_runs/cleanup-smoke/vanilla
```

```bash
conda run -n justatom python -m justatom.api.train \
  --config configs/train.yaml \
  --method atom_gate \
  --dataset.name-or-path justatom/builtin \
  --optimization.num-samples 64 \
  --optimization.batch-size 8 \
  --optimization.epochs 1 \
  --artifacts.save-dir .tmp_runs/cleanup-smoke/atom_gate
```

```bash
conda run -n justatom python -m justatom.api.train \
  --config configs/train.yaml \
  --method atomic \
  --dataset.name-or-path justatom/builtin \
  --optimization.num-samples 64 \
  --optimization.batch-size 8 \
  --optimization.epochs 1 \
  --memory-bank.size 32 \
  --memory-bank.warmup-steps 1 \
  --memory-bank.hard-warmup-steps 1 \
  --memory-bank.hard-ramp-steps 2 \
  --artifacts.save-dir .tmp_runs/cleanup-smoke/atomic
```

Expected for each run:

```text
.tmp_runs/cleanup-smoke/<method>/run_manifest.yaml
.tmp_runs/cleanup-smoke/<method>/encoder/
.tmp_runs/cleanup-smoke/<method>/research/checkpoint.pt
```

The three losses and all tracked gradients are finite. `vanilla` logs no alpha/bank metrics; `atom_gate` logs alpha but no bank metrics; `atomic` logs alpha, bank, collision, hard-weight, margin, and admission metrics.

- [ ] **Step 10: Verify artifact loading**

Load each deployable encoder through the normal evaluation encoder loader. Load the ATOMIC research checkpoint into `ContrastiveTrainingModule` on CPU and assert the alpha and margin state dictionaries are present.

- [ ] **Step 11: Commit**

```bash
git add -A \
  justatom/running/trainer.py \
  justatom/running/trainer_jobs.py \
  justatom/running/encoders.py \
  justatom/training \
  justatom/api/tune.py \
  justatom/modeling/head.py \
  tests \
  README.md \
  docs/architecture.md \
  docs/research/atomic-experiments/README.md
git commit -m "refactor: remove legacy training stack"
```

Review the staged paths before committing so unrelated dirty files are not included.

---

## Final Review Gate

- [ ] `git diff --check` passes.
- [ ] `pytest -m "not integration" -q` passes in `conda activate justatom`.
- [ ] Golden CPU tests cover vanilla, alpha-gated auxiliary pressure, weighted bank denominator, and live `m(q)` gradient.
- [ ] Only `vanilla`, `atom_gate`, and `atomic` are accepted by Python and shell entry points.
- [ ] Canonical and ablation runs are distinguishable in the manifest.
- [ ] No method mathematics is read from environment variables.
- [ ] `api/train.py` contains no dataset iteration, objective implementation, or recipe compatibility mapping.
- [ ] `running/encoders.py` contains no train-time alpha/margin/gamma logic.
- [ ] `running/trainer.py` and `running/trainer_jobs.py` are deleted.
- [ ] Deployable encoder and research checkpoint load independently.
- [ ] `justatom-rc`, `.env`, `.tmp_runs`, `weights`, `tmp`, and dissertation build products are untouched.
