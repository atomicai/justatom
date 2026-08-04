# Production Training Cleanup Design

**Date:** 2026-08-02  
**Status:** Approved for implementation planning  
**Source of truth:** the root `justatom` package

## 1. Context

The research training path has accumulated several generations of implementation:

- scalar gamma training;
- frozen and trainable encoder combinations;
- query and query-document alpha gates;
- query-diagonal gates;
- query-conditional temperature experiments;
- several memory-bank admission strategies;
- configuration spread across YAML, shell variables, API normalization, job classes, Lightning modules, metadata, and W&B payloads.

The current implementation contains the validated ATOMIC method, but its public contract is obscured by legacy modes and duplicated configuration logic. The cleanup must make the repository suitable for production and dissertation reproducibility without silently changing the validated numerical behavior.

The root repository is the only source of truth. The nested `justatom-rc` repository is outside this cleanup and may be synchronized after the root implementation is complete.

## 2. Goals

1. Expose exactly three public training methods:
   - `vanilla`;
   - `atom_gate`;
   - `atomic`.
2. Replace recipe aliases and boolean combinations with one explicit `method` field.
3. Keep method internals configurable through typed nested sections.
4. Preserve the current validated mathematics of the three methods.
5. Remove obsolete training modes, experiments, aliases, and hidden environment configuration.
6. Make every run reproducible from one resolved configuration manifest.
7. Separate deployable encoder artifacts from research checkpoints.
8. Reduce the API and orchestration layers to narrow, testable responsibilities.

## 3. Non-goals

- Changing the ATOMIC objective or introducing another adaptive head.
- Retuning method defaults during the refactor.
- Changing datasets, evaluation metrics, or Weaviate behavior.
- Migrating the nested `justatom-rc` repository in the same change.
- Preserving backward compatibility for retired training modes or recipe aliases.
- Rewriting unrelated retrieval, storage, service, or evaluation modules.

## 4. Public Training Contract

All public training entry points accept one method:

```yaml
method: vanilla
```

or:

```bash
--method vanilla
--method atom_gate
--method atomic
```

The methods have these fixed semantic boundaries.

### 4.1 `vanilla`

- Train the dense encoder with the canonical contrastive objective.
- Do not instantiate `alpha(q)`.
- Do not instantiate or populate a memory bank.
- Do not instantiate `m(q)`.

### 4.2 `atom_gate`

- Use the same dense encoder and main contrastive objective as `vanilla`.
- Instantiate the query-only, train-time `alpha(q)` head.
- Apply the currently validated alpha-gated auxiliary objective.
- Do not instantiate or populate a memory bank.
- Do not instantiate `m(q)`.

### 4.3 `atomic`

- Include all `atom_gate` behavior.
- Instantiate the dynamic memory bank.
- Compute the collision diagnostic

  \[
  g(q_i) = \max_j s(q_i, b_j) - s(q_i, p_i).
  \]

- Apply adaptive hard-negative pressure using

  \[
  w_h(q_i) = \sigma\left(\frac{t-g(q_i)}{\beta_g}\right).
  \]

- Instantiate the query-conditional too-hard margin `m(q)`.
- Apply differentiable bank admission to the contrastive denominator.

No fourth public method is created for ablations. Ablations are expressed through explicit nested parameters while retaining one of the three method identities.

An ablation is not presented as the canonical method. For example, the fixed-margin control remains `method: atomic`, but uses `experiment.role: ablation` and records `memory_bank.margin.mode: constant`. This keeps the method family stable while making the controlled deviation explicit and machine-readable.

## 5. Typed Configuration

Configuration is parsed once into typed objects and validated once before model or dataset initialization.

```text
TrainConfig
|- method: TrainingMethod
|- experiment: ExperimentConfig
|- model: ModelConfig
|- dataset: DatasetConfig
|- optimization: OptimizerConfig
|- objective: ContrastiveConfig
|- alpha_gate: AlphaGateConfig
|- memory_bank: MemoryBankConfig
|- telemetry: TelemetryConfig
|- artifacts: ArtifactConfig
`- runtime: RuntimeConfig
```

The initial implementation may use frozen Python dataclasses and enums. It must not introduce a new validation dependency solely for this cleanup.

Example:

```yaml
method: atomic

experiment:
  role: canonical

model:
  name_or_path: intfloat/multilingual-e5-small
  query_prefix: "query:"
  content_prefix: "passage:"
  max_query_seq_len: 128
  max_seq_len: 512

optimization:
  optimizer: adamw
  lr_encoder: 2.0e-5
  weight_decay: 0.01
  batch_size: 32
  grad_acc_steps: 1
  epochs: 1

objective:
  temperature: 0.05
  learnable_temperature: true
  decoupled: true

alpha_gate:
  enabled: true
  mix_weight: 0.3
  layers: 1
  hidden_dim: auto
  dropout: 0.0
  activation: gelu

memory_bank:
  enabled: true
  size: 512
  warmup_steps: 50
  mining: mixed
  hard_negatives: 4
  random_negatives: 12
  hard_warmup_steps: 120
  hard_ramp_steps: 200
  adaptive:
    enabled: true
    collision_threshold: 0.0
    collision_beta: 0.05
  margin:
    mode: query
    base: 0.05
    scale: 0.02
    minimum: 0.0
    maximum: 0.15
    admission_beta: 0.05
    regularization_weight: 50.0
```

### 5.1 Validation rules

- Unknown fields are errors.
- Invalid enum values are errors.
- Invalid ranges are errors before model loading.
- `vanilla` rejects enabled alpha-gate or bank sections.
- `atom_gate` requires the query-only alpha gate and rejects an enabled bank.
- `atomic` requires the query-only alpha gate, adaptive bank, and query-conditional margin.
- `alpha_gate.input` is fixed to `query`; query-document gating is not supported.
- Structural controls may relax a method invariant only with `experiment.role: ablation`.
- Every ablation records its deviation from the canonical method in the resolved manifest and run name.
- Public benchmark variants always use `experiment.role: canonical`.
- Environment variables may select operational secrets or hardware behavior, but may not alter method mathematics.
- CLI values override YAML values.
- The final resolved values, including defaults, are serialized before training starts.

## 6. Target Module Boundaries

```text
justatom/training/
|- config.py          # dataclasses, enums, parsing, validation
|- methods.py         # defaults and invariants for the three methods
|- objective.py       # batch and bank contrastive denominator
|- alpha_gate.py      # query-only alpha(q) module
|- memory_bank.py     # FIFO storage, mining, g(q), admission
|- sampling.py        # safe in-batch negatives and lexical overlap
|- module.py          # one compositional LightningModule
|- telemetry.py       # metric collection and logging payloads
|- data.py            # row preparation and loader construction
`- job.py             # data loader, Lightning Trainer, artifacts
```

### 6.1 `config.py`

Owns configuration shape and validation. It has no PyTorch, Lightning, dataset, or storage dependencies.

### 6.2 `methods.py`

Resolves `vanilla`, `atom_gate`, or `atomic` defaults into a fully explicit `ResolvedTrainConfig`. It does not implement training logic.

### 6.3 `objective.py`

Owns the differentiable objective. Given current query/document embeddings and optional bank candidates, it returns loss components and tensor-valued diagnostics. It does not own the encoder, optimizer, logger, or checkpoint paths.

### 6.4 `alpha_gate.py`

Owns only the query-conditioned scalar head and its parameterization. The supported public input is `q`; pair-conditioned and diagonal variants are removed.

### 6.5 `memory_bank.py`

Owns detached FIFO storage, bank selection, collision diagnostics, hard-row weighting, and soft admission. Stored vectors never retain graphs from earlier steps. Current-batch query embeddings remain live so `m(q)` and admission receive gradients.

### 6.6 `module.py`

One `ContrastiveTrainingModule` composes:

```python
ContrastiveTrainingModule(
    encoder=encoder,
    objective=objective,
    alpha_gate=alpha_gate_or_none,
    memory_bank=memory_bank_or_none,
)
```

It owns training-step order and optimizer interaction, but delegates mathematics and telemetry calculation to the focused components.

### 6.7 `job.py`

Owns Lightning startup, artifact directories, and resolved-manifest creation. It receives a validated `ResolvedTrainConfig`; it does not normalize recipe aliases or reinterpret method flags.

### 6.8 `sampling.py`

Owns safe negative-index selection, key-collision avoidance, and lexical overlap helpers. It has no Lightning, optimizer, logging, or artifact responsibilities.

### 6.9 `data.py`

Owns dataset iteration, row normalization, sampling, processor construction, and loader construction. It does not select a training method or implement an objective.

## 7. Training Data Flow

```text
YAML + CLI overrides
        |
        v
TrainConfig parsing
        |
        v
method defaults + validation
        |
        v
ResolvedTrainConfig + run_manifest.yaml
        |
        v
TrainingJob builds processor, loader, encoder, heads, objective, bank
        |
        v
ContrastiveTrainingModule.training_step
        |
        +--> encode Q and P
        +--> compute current-batch similarities
        +--> compute alpha auxiliary path when configured
        +--> select bank candidates when configured
        +--> compute g(q), w_h(q), m(q), and admission
        +--> compute total loss and gradients
        +--> enqueue detached current positive documents
        `--> emit telemetry
```

Bank insertion occurs after the current loss is assembled. This prevents current positives from appearing as stale bank negatives in the same step.

## 8. Numerical Preservation Contract

Before replacing the active implementation, deterministic golden tests capture the current behavior of all three public methods on a synthetic batch with fixed weights and random seeds.

### 8.1 `vanilla` golden values

- embeddings;
- `N x N` similarities;
- temperature;
- total loss;
- encoder gradient tensors or stable gradient summaries.

### 8.2 `atom_gate` golden values

- all `vanilla` values;
- `alpha(q)` values;
- auxiliary loss;
- total loss;
- alpha-head and encoder gradients.

### 8.3 `atomic` golden values

- all `atom_gate` values;
- selected bank indices;
- `N x M` bank similarities;
- positive similarities;
- `g(q)`;
- `w_h(q)`;
- `m(q)`;
- admission weights;
- final `N x (N + M)` denominator contributions;
- total loss;
- encoder, alpha-head, and margin-head gradients.

Old and new implementations are compared with `torch.testing.assert_close` using documented tolerances appropriate for deterministic CPU float32 execution. MPS runs are smoke tests, not the golden numerical oracle.

Intentional public API deletion is not treated as numerical incompatibility.

## 9. Artifacts And Reproducibility

Every run writes a versioned manifest before training begins:

```yaml
schema_version: 1
method: atomic
seed: 42
model: intfloat/multilingual-e5-small
dataset: justatom/mmarco-ru-selected
git_commit: <commit>
git_dirty: true
resolved_config: {}
runtime: {}
```

The artifact directory contains two distinct outputs.

### 9.1 Deployable encoder

Contains the tuned encoder, tokenizer/processor metadata, prefixes, sequence lengths, and model metadata required by retrieval inference. Train-time alpha and margin heads are not required to serve embeddings.

### 9.2 Research checkpoint

Contains the encoder plus alpha head, margin head, objective state, optimizer state when resumable checkpoints are enabled, resolved manifest, and telemetry schema. This artifact supports audit, continuation, and dissertation analysis.

## 10. Telemetry Contract

Telemetry remains research-grade but moves out of the Lightning trainer implementation. Metric names become stable schema fields rather than ad hoc dictionaries duplicated across components.

Required groups include:

- optimization and loss components;
- batch retrieval metrics;
- temperature;
- alpha distribution and alpha gradient norm;
- bank fill, selected counts, hard/random counts, and candidate similarities;
- positive similarity and `g(q)` distribution;
- hard-row weights;
- `m(q)` distribution and margin-head gradient norm;
- admission-weight distribution;
- embedding geometry diagnostics.

Tensor metrics are detached only at the telemetry boundary. Values participating in loss remain in the live graph until backward completes.

## 11. Deliberate Deletions

The implementation removes, rather than deprecates:

- `GammaOnlyTrainingJob`;
- `EncoderGammaTrainingJob`;
- `EncoderOnlyTrainingJob`;
- `UniGammaLightningTrainer`;
- `BiGammaLightningTrainer`;
- scalar `gamma1` and `gamma2` modes;
- query-document `alpha(q,d+)`;
- query-diagonal gating;
- query-conditional `tau(q)`;
- recipe aliases and retired bank variant names;
- hidden `ALPHA_GATE_*`, `TAU_QUERY_*`, and `MARGIN_QUERY_*` mathematical overrides;
- tests and scripts whose only purpose is a retired mode;
- loss implementations that have no remaining production or public API references.

Removal of a loss class requires a repository-wide reference check. Shared losses still used by unrelated public APIs are not deleted as part of this training cleanup.

## 12. CLI And Script Cleanup

`justatom.api.train` becomes a thin entry point:

1. parse CLI arguments;
2. load YAML;
3. apply explicit overrides;
4. build and validate `TrainConfig`;
5. invoke `TrainingJob`.

`scripts/run_pipeline.sh` remains the end-to-end tune/evaluate orchestrator but stops reimplementing method defaults. It forwards `--method` and explicit overrides to the Python configuration layer.

`scripts/run_benchmark.sh` accepts only:

```text
vanilla,atom_gate,atomic
```

Retired names fail as invalid choices; they are not silently mapped.

## 13. Error Handling

- Configuration errors include the full dotted field path and invalid value.
- Method invariant failures occur before dataset or model loading.
- Non-finite loss, temperature, margin, admission, or gradients fail the run with the offending metric name and step.
- Empty or insufficient banks return a valid no-bank objective and explicit telemetry, not shape-dependent special cases.
- Artifact writes use temporary paths followed by atomic replacement where practical.
- A failed run retains its resolved manifest and available telemetry.

## 14. Migration Sequence

### Phase 0: protect current evidence

- Inventory the current public commands and defaults.
- Add deterministic golden tests around the existing implementation.
- Record the current resolved configurations for `vanilla`, `atom_gate`, and `atomic`.

### Phase 1: typed configuration

- Add config dataclasses and method enum.
- Add strict parsing and validation tests.
- Make current jobs consumable from the resolved config without changing their internals.

### Phase 2: compositional training core

- Extract query-only alpha gate.
- Extract the contrastive objective.
- retain and narrow the memory-bank component.
- Introduce the single compositional Lightning module.
- Compare old and new paths in golden tests.

### Phase 3: entry-point switch

- Switch `api/train.py` to the new config and job.
- Switch pipeline and benchmark scripts to `--method`.
- Generate the resolved manifest and split artifacts.
- Run CPU tests and MPS smoke runs for all three methods.

### Phase 4: legacy deletion

- Delete old job and trainer hierarchies.
- Delete scalar gamma, pair alpha, diagonal gate, and query-temperature code.
- Delete retired arguments, aliases, scripts, tests, and dead losses.
- Remove compatibility mapping code.

### Phase 5: final verification

- Run the complete non-integration test suite.
- Run configuration and CLI contract tests.
- Run one short training smoke for each method.
- Verify deployable encoder loading.
- Verify research checkpoint loading and resume metadata.
- Confirm no retired symbol or environment flag remains with `rg`.

## 15. Success Criteria

The cleanup is complete when:

1. Only `vanilla`, `atom_gate`, and `atomic` are accepted publicly.
2. Every training run is represented by a strict resolved config and versioned manifest.
3. The new implementation matches the old golden tensors and gradients within declared tolerances.
4. The active training path contains no scalar gamma, pair-alpha, diagonal-gate, or query-temperature implementation.
5. Mathematical settings are not hidden in environment variables.
6. `api/train.py` contains entry-point orchestration rather than dataset and objective implementation.
7. The Lightning module composes focused alpha, objective, bank, and telemetry components.
8. Deployable encoders and research checkpoints are distinct and loadable.
9. The benchmark exposes the same three method names used in configs and documentation.
10. Unit tests and three method smoke runs pass in the `justatom` conda environment.
