# Difficulty-Supervised Alpha and Count-Normalized Memory Bank

**Date:** 2026-08-14

**Branch:** `feature/difficulty-alpha-normalized-bank`

**Base:** `fix/alpha-gate-objective` at `6038294`

**Status:** approved design

## 1. Objective

Replace two empirically confounded training signals with mechanisms whose
strength has an explicit interpretation:

1. supervise `alpha(q)` from detached retrieval confidence instead of mixing
   uncalibrated semantic cosine and lexical recall scores;
2. make memory-bank pressure invariant to the number of selected candidates,
   then test gradient projection against the matched normalized-bank control.

The primary success criterion is improved retrieval quality against a matched
coupled InfoNCE baseline. Internal controller or gradient diagnostics explain
the result but do not constitute success by themselves.

## 2. Empirical Motivation

The matched Qwen3 0.6B experiment at commit `6038294` used 3,000 training
pairs, seed 42, microbatch 8, gradient accumulation 4, and one epoch.

| Variant | HitRate@1 | MRR@10 |
| --- | ---: | ---: |
| DCL | 0.3513 | 0.4551 |
| coupled InfoNCE | 0.5896 | 0.6929 |
| coupled plus unnormalized bank | 0.5403 | 0.6512 |
| coupled plus projected unnormalized bank | 0.5410 | 0.6523 |
| detached alpha stack | 0.5660 | 0.6733 |

Three diagnostics identify the relevant confounds:

- DCL continues applying a unit positive-logit gradient to already solved
  rows, whereas coupled InfoNCE saturates.
- The old alpha pair objective observed mean semantic-positive similarity
  `0.655` and lexical-positive recall `0.113`. Its alpha derivative was
  negative for every one of 128 inspected rows; the negative hinge was never
  active. Alpha therefore approached one for structural rather than
  query-specific reasons.
- The bank added 12 candidates to only 7 in-batch negatives because gradient
  accumulation does not enlarge the `8 x 8` contrastive matrix. Projection
  activated for 14 of 175 bank-active microbatches, so the projected and naive
  bank updates were identical most of the time.

## 3. Scope and Causal Order

The implementation produces two independently testable changes:

1. `vanilla -> atom_gate`: confidence-supervised query conditioning without a
   memory bank;
2. `vanilla -> normalized bank -> atomic`: count normalization first, gradient
   projection second.

The mechanisms are not combined in the first screening matrix. A combined
method is considered only if both independent contrasts do not degrade the
coupled baseline.

Adaptive mining, learned `m(q)`, query-conditional temperature, lexical score
calibration, and a full hyperparameter sweep are outside this change.

## 4. Shared Coupled Objective

For normalized query and positive embeddings, define

```text
z_ij = cosine(q_i, p_j) / tau.
```

All canonical methods retain coupled InfoNCE:

```text
                            exp(z_ii)
p_i+ = ---------------------------------------------------,
       exp(z_ii) + sum_(j != i) exp(z_ij)

L_main,i = -log(p_i+).
```

DCL remains an explicitly labeled ablation and is not modified by this work.

## 5. Difficulty-Supervised Alpha

### 5.1 Semantics

`alpha_i` predicts how confidently the current retrieval model selects the
positive document for query `i`. High confidence suppresses auxiliary SimCSE
pressure; low confidence retains it.

The online soft target is

```text
t_i = stop_gradient(p_i+).
```

The target is stochastic because the negative documents depend on batch
composition. The head sees only `q_i`, so it learns the expected retrieval
confidence for that query representation rather than memorizing the current
candidate row. Batch size and sampling policy are fixed within every matched
comparison and recorded in the run manifest.

The head consumes a detached query representation:

```text
alpha_i = sigmoid(MLP(stop_gradient(q_i))).
```

Detaching the input allows gradients to update the MLP parameters while
preventing gate supervision from changing the encoder.

### 5.2 Loss

The alpha-head supervision is binary cross entropy with a soft target:

```text
L_gate,i = BCE(alpha_i, t_i)
         = -t_i log(alpha_i) - (1 - t_i) log(1 - alpha_i).
```

For the head logit `u_i`, where `alpha_i = sigmoid(u_i)`, the derivative is

```text
dL_gate,i / du_i = alpha_i - t_i.
```

The complete atom-gate objective is

```text
L_i = L_main,i
    + (1 - stop_gradient(alpha_i)) * lambda_sc * L_SimCSE,i
    + lambda_alpha * L_gate,i.
```

The autograd contract is:

| Term | Encoder gradient | Alpha-head gradient |
| --- | --- | --- |
| coupled InfoNCE | yes | no |
| weighted SimCSE | yes | no |
| gate BCE | no | yes |

The confidence target is computed from the same coupled logits as `L_main`.
It is detached before BCE. Alpha is detached only when used as the SimCSE
coefficient.

### 5.3 Canonical Cleanup

The canonical atom-gate path no longer computes semantic/lexical pair scores
or their pairwise margin objective. The production configuration replaces the
ambiguous pair-mixing weight with an explicit alpha supervision weight.

The canonical configuration is:

```yaml
objective:
  decoupled: false
  simcse_dropout_weight: 0.1

alpha_gate:
  enabled: true
  supervision_weight: 0.3
```

The old semantic/lexical mixing objective and its production plumbing are
removed. Historical checkpoints and metrics remain documented artifacts; a
future lexical ablation must be introduced as a separate objective rather
than restoring two meanings to the same alpha coefficient.

The schema cleanup is explicit:

| Removed field or input | Replacement |
| --- | --- |
| `alpha_gate.mix_weight` | `alpha_gate.supervision_weight` |
| `alpha_gate.mix_weight_warmup_steps` | none; BCE updates only the head |
| `alpha_gate.entropy_weight` | none; the confidence target defines the desired distribution |
| `objective.pairwise_margin` | none in the canonical objective |
| semantic and lexical pair score inputs | detached positive-confidence target |
| training-module lexical lookup | removed from the canonical module data flow |

### 5.4 Alpha Telemetry

Each microbatch records distributions for:

- `alpha/*`;
- `alpha_target/*`;
- `alpha_aux_weight/*`, equal to `1 - alpha`;
- `alpha/absolute_error_mean`;
- `loss/alpha_supervision`;
- retrieval metrics bucketed by target confidence for offline aggregation.

An alpha distribution near one is not automatically a failure when target
confidence is also near one. Failure means poor calibration or no retrieval
improvement, not an arbitrary target mean.

## 6. Count-Normalized Memory Mass

### 6.1 Formula

For row `i`, define

```text
A_batch,i = sum_(j != i) exp(z_ij),
A_bank,i  = sum_(b = 1..K_i) w_ib exp(z_ib),
```

where `K_i` is the number of active, identity-safe bank candidates and `w_ib`
contains any candidate-specific admission weight. The first screening uses
random candidates with `w_ib = 1`; adaptive weights remain out of scope.

The augmented loss is

```text
L_aug,i = -z_ii
        + log(
              exp(z_ii)
              + A_batch,i
              + lambda(t) * (N - 1) / K_i * A_bank,i
          ).
```

The factor `(N - 1) / K_i` removes the raw candidate-count confound. If bank
and in-batch candidates follow the same score distribution, the expected bank
mass is `lambda(t)` times the in-batch negative mass regardless of `K_i`.

The first canonical value is

```text
lambda_max = 0.5.
```

For the screening run with `N=8` and `K_i=12`, each bank candidate receives a
full-ramp mass weight of `0.5 * 7 / 12 = 0.2917` before any candidate-specific
weight.

### 6.2 Ramp

Bank mass grows linearly after the existing warmup:

```text
r(t) = clamp((t - warmup_steps + 1) / mass_ramp_steps, 0, 1),
lambda(t) = mass_ratio * r(t).
```

Before warmup, selection is a no-op. At the first active optimizer step the
mass is one ramp increment rather than the full value.

Canonical memory settings become:

```yaml
memory_bank:
  enabled: true
  size: 512
  warmup_steps: 50
  mining: random
  random_negatives: 12
  mass_ratio: 0.5
  mass_ramp_steps: 20
```

`mass_ratio` must be finite and non-negative. `mass_ramp_steps` must be a
positive integer. A zero mass ratio is equivalent to a disabled bank
contribution.

### 6.3 Row-Level Edge Cases

- `K_i == 0`: the row receives no bank contribution and no logarithm of zero
  is constructed.
- `N < 2`: in-batch contrastive training fails with a clear validation error.
- Identity masks are applied before `K_i` is counted.
- Candidate-specific log weights are added to the count-normalization log
  weight.
- Masked logits remain finite for MPS compatibility.

### 6.4 Memory Telemetry

The bank records:

- configured `memory/mass_ratio`;
- current `memory/mass_ramp` and `memory/effective_mass_ratio`;
- active `K_i` distribution;
- per-candidate normalization weight distribution;
- positive and selected-negative similarity distributions;
- `loss/memory` and all existing projection metrics.

## 7. Gradient Projection

The exact objective decomposition remains

```text
L_primary = L_main,
L_memory  = L_aug - L_main.
```

Therefore, without projection,

```text
grad(L_primary) + grad(L_memory) = grad(L_aug).
```

For trainable-parameter gradients `g_p` and `g_m`, ATOMIC retains the current
one-sided projection:

```text
if dot(g_p, g_m) < 0:
    g_m <- g_m - dot(g_p, g_m) / ||g_p||^2 * g_p

g_update = g_p + memory_weight * g_m.
```

Projection protects only the current microbatch primary direction to first
order. It does not claim that an aligned memory gradient improves validation
quality. The normalized-bank control is therefore mandatory.

## 8. Components and Data Flow

### 8.1 Alpha path

1. Encoder produces normalized query and positive embeddings.
2. The shared coupled similarity matrix produces `L_main` and detached
   positive confidence.
3. Alpha-head receives detached query embeddings.
4. BCE updates only the alpha-head.
5. Detached alpha weights per-row SimCSE; SimCSE updates only the encoder.
6. Telemetry records prediction, target, calibration, and effective weight.

### 8.2 Bank path

1. Bank selection applies identity masks and chooses random candidates.
2. The actual active count `K_i` is measured per row.
3. The optimizer step determines the current mass ramp.
4. Count-normalization is combined with optional candidate log weights.
5. The objective computes `L_aug`, `L_primary`, and `L_memory`.
6. The control run applies the ordinary augmented gradient.
7. ATOMIC projects only conflicting memory gradients before accumulation.

## 9. Testing

### 9.1 Alpha unit tests

- BCE is minimized when `alpha == detached confidence`.
- Gate BCE produces finite, non-zero head gradients.
- Gate BCE produces no encoder gradient through either target or head input.
- Weighted SimCSE produces encoder gradients but no alpha-head gradient.
- Easy and hard synthetic rows produce the expected auxiliary weights.
- Canonical atom-gate configuration contains no lexical pair-mixing path.

### 9.2 Bank unit tests

- Equal score distributions produce the same expected bank mass for different
  `K_i` values.
- For `N=8`, `K=12`, and full-ramp ratio `0.5`, candidate weight is `7/24`.
- Ramp is zero before warmup, increases linearly, and reaches exactly `0.5`.
- Rows with no valid bank candidate are unchanged from `L_main`.
- Candidate-specific and count-normalization log weights compose additively.
- MPS-safe finite masks and all gradients remain finite.
- The decomposition `L_aug == L_primary + L_memory` remains exact.

### 9.3 Integration and regression tests

- Canonical method resolution emits the new explicit defaults.
- Run manifests serialize alpha supervision and memory mass contracts.
- CSV telemetry has stable columns before and after bank warmup.
- Automatic and manual optimization retain matched gradient accumulation.
- The complete offline test suite passes on Python 3.10 through 3.13 CI
  targets; focused MPS execution is checked locally.

## 10. Screening Experiment

Fixed conditions reuse the completed causal Qwen matrix:

```text
model: Qwen/Qwen3-Embedding-0.6B
adaptation: LoRA r=16, alpha=32, dropout=0.05, RS-LoRA, all-linear
dataset: justatom
training pairs: 3000
microbatch: 8
gradient accumulation: 4
optimizer effective batch: 32
contrastive matrix: 8 x 8
epochs: 1
temperature: 0.05, learnable
seed: 42
```

The first matrix is:

| Run | Alpha | Bank | Projection | Contrast |
| --- | --- | --- | --- | --- |
| `coupled` | no | no | no | existing matched baseline |
| `difficulty_alpha` | confidence target | no | no | alpha effect |
| `normalized_bank` | no | normalized random | no | normalized bank effect |
| `normalized_bank_projected` | no | normalized random | yes | projection effect |

All four variants are trained from the same source checkpoint on the
implementation commit. The historical coupled checkpoint is used only as a
sanity reference, not as a row in the new matched matrix.

Primary metrics are HitRate@1 and MRR@10. Secondary metrics are HitRate@5,
HitRate@10, and NDCG@10. A one-seed improvement is exploratory. Any result
used as dissertation evidence is repeated on at least one additional seed and
on multiple datasets.

A mechanism passes screening when neither primary metric is below the matched
baseline and at least one primary metric is higher. Promotion is conditional;
diagnostic improvements without this retrieval result do not qualify.

The combined alpha plus projected-bank method is run only if both independent
mechanisms do not degrade the coupled baseline in screening.

## 11. Documentation and Provenance

Run manifests record:

- the coupled contrastive contract;
- alpha target definition and detach boundaries;
- alpha supervision weight;
- bank mass ratio and ramp;
- contrastive microbatch separately from optimizer effective batch;
- implementation commit, seed, model, dataset, and sample count.

Historical metrics from the uncalibrated lexical-mix objective or
unnormalized bank are retained as ablations and are not pooled with the new
canonical results.
