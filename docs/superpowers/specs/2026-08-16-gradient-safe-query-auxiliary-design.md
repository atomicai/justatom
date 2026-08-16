# Gradient-Safe Query Auxiliary Design

## Status

Approved on 2026-08-16 for implementation as an `atom_gate` ablation. The
mechanism is not promoted to a canonical method until matched retrieval runs
confirm it.

## Research Status

The current evidence distinguishes three different mechanisms:

| Mechanism | Current evidence |
| --- | --- |
| coupled InfoNCE (`vanilla`) | matched primary baseline |
| confidence-controlled SimCSE (`atom_gate`) | small positive two-seed mean on `justatom`; deeper-rank gain with an HR@1 cost on mMARCO |
| detached online bank with projection (`atomic`) | negative on the modern Qwen/justatom screen |

For Qwen3-Embedding-0.6B LoRA on `justatom`, the two-seed paired mean effect of
`atom_gate` over coupled InfoNCE was `+0.112 pp HR@1`, `+0.232 pp MRR@10`, and
`+0.237 pp nDCG@10`. The signal is positive but too small to establish robust
superiority.

On the clean mMARCO screen, the alpha-controlled auxiliary repeatedly improved
HR@10 by about `0.5 pp` while losing `0.3-0.4 pp HR@1`. Separating the target
and SimCSE temperatures repaired mechanism saturation but did not remove this
head-versus-tail trade-off.

The normalized bank reduced HR@1 by about `1.9 pp` on the modern
Qwen/justatom screen. One-sided projection did not rescue it, and only `2.93%`
of training steps had a negative primary-memory gradient dot product. The bank
therefore remains a negative ablation and is out of scope for the first
gradient-safe auxiliary experiment.

Historical `gamma-train-only` and earlier bank results used different objective
contracts, temperatures, batch shapes, or implementations. They motivate the
research question but must not be pooled with the modern matched runs.

## Problem

For normalized query and positive-document embeddings, define

```text
S_ij = q_i^T p_j
pi_ij = softmax(S_i / tau_retrieval)_j
R_i = -log pi_ii
```

The current alpha target is detached positive retrieval confidence:

```text
t_i = stop_gradient(softmax(S_i / tau_target)_i)
alpha_i = sigmoid(head(stop_gradient(q_i)))
```

The atom-gate loss is

```text
J_i = R_i
    + lambda_sc (1 - stop_gradient(alpha_i)) A_i
    + lambda_alpha BCEWithLogits(alpha_logit_i, t_i),
```

where `A_i` is query-query SimCSE between two dropout views. Consequently,

```text
grad_encoder J_i
  = grad_encoder R_i
  + lambda_sc (1 - alpha_i) grad_encoder A_i.
```

The BCE derivative is

```text
d BCEWithLogits(alpha_logit_i, t_i) / d alpha_logit_i = alpha_i - t_i.
```

The head therefore estimates conditional retrieval confidence. It does not
estimate whether the SimCSE gradient helps the retrieval gradient. A difficult
query receives more auxiliary pressure even when the two objectives disagree.

## First-Order Safety Condition

Let the batch-level retrieval and difficulty-weighted auxiliary gradients over
the shared trainable parameters be

```text
g = grad R
h = grad A_beta
A_beta = mean_i[(1 - stop_gradient(alpha_i)) A_i].
```

For an update direction `d = g + w h` and optimizer step
`theta_next = theta - eta d`, first-order Taylor expansion gives

```text
R(theta_next)
  = R(theta) - eta g^T d + O(eta^2)
  = R(theta) - eta ||g||^2 - eta w g^T h + O(eta^2).
```

Relative to the primary-only direction, a non-negative auxiliary weight is
first-order safe exactly when

```text
w g^T h >= 0.
```

The current confidence weight `1 - alpha` is non-negative but places no
constraint on `g^T h`.

## Decision

Use confidence for query-level difficulty and parameter-gradient cosine for
batch-level compatibility:

```text
beta_i = 1 - stop_gradient(alpha_i)
A_beta = mean_i[beta_i lambda_sc A_i]

g = grad R
h = grad A_beta
c = (g^T h) / (||g|| ||h|| + eps)
a = max(0, c)
```

The raw compatibility-controlled auxiliary is `a h`. Bound its shared-gradient
norm relative to the retrieval gradient:

```text
cap = rho ||g||
k = min(1, cap / (a ||h|| + eps))       when a > 0
k = 0                                   when a = 0

h_safe = a k h
d = g + h_safe.
```

Then

```text
g^T d = ||g||^2 + a k g^T h >= ||g||^2.
```

The controlled auxiliary also satisfies

```text
||h_safe|| <= rho ||g||.
```

This is an exact first-order statement over the trainable parameter vector used
to calculate `g` and `h`. It is not a guarantee that a finite optimizer step,
the validation loss, or a discrete retrieval metric cannot regress. Curvature,
AdamW state, stochastic batches, and generalization remain relevant.

## Why Parameter-Space, Batch-Level Compatibility

Per-query representation gradients would be cheaper, but their cosine does not
imply the same cosine after multiplication by the encoder Jacobian. The claimed
first-order guarantee therefore uses gradients over the actual trainable
parameters.

Computing full per-query parameter gradients would require one backward pass per
query or a vectorized per-sample-gradient mechanism with substantially greater
memory. The first implementation instead aggregates the query-conditioned
auxiliary into `A_beta` and computes one batch-level compatibility value. The
controller remains query-aware through `beta_i`, while the safety decision is
made over the true batch update.

## Loss Decomposition

`ContrastiveObjective` must expose an exact decomposition:

```text
retrieval_loss = mean_i R_i
auxiliary_loss = mean_i[beta_i lambda_sc A_i + other encoder auxiliaries]
head_loss = mean_i[lambda_alpha BCEWithLogits(alpha_logit_i, t_i)]
memory_loss = existing detached-bank increment or None

total_loss = retrieval_loss + auxiliary_loss + head_loss + memory_loss
```

For the approved experiment, memory is disabled and the only encoder auxiliary
is SimCSE. The alpha head remains detached from both encoder losses. Its
`head_loss` gradient is added unchanged after controlling the shared encoder
auxiliary.

The learnable retrieval temperature is part of the shared parameter vector when
it receives both retrieval and auxiliary gradients. A fixed SimCSE temperature
has no optimizer state.

## Configuration Contract

Add a focused configuration section:

```yaml
auxiliary_gradient:
  mode: safe             # off | observe | safe
  max_norm_ratio: 0.25   # rho
  eps: 1.0e-12
```

Modes:

| Mode | Update | Purpose |
| --- | --- | --- |
| `off` | existing automatic optimization | canonical behavior and matched baseline |
| `observe` | `g + h` unchanged | measure compatibility without intervention |
| `safe` | `g + h_safe` | gradient-compatible experiment |

`observe` and `safe` require `method: atom_gate` and
`experiment.role: ablation`. They are rejected for `vanilla` and `atomic` in
this phase. Canonical presets remain unchanged.

`max_norm_ratio` must be finite and non-negative. `eps` must be finite and
strictly positive. A ratio of zero is a valid primary-only control inside the
same manual-optimization path.

## Optimization Semantics

Enabling `observe` or `safe` selects Lightning manual optimization. For every
contrastive microbatch:

1. Capture any already accumulated gradients.
2. Backpropagate `retrieval_loss` and capture `g`.
3. Backpropagate `auxiliary_loss` and capture `h`.
4. Backpropagate `head_loss` and capture the detached alpha-head gradients.
5. In `observe`, combine `g + h`; in `safe`, combine `g + h_safe`.
6. Add head gradients and previously accumulated microbatch gradients.
7. Step only at the configured gradient-accumulation boundary.

Compatibility and norm control are calculated per microbatch before the
controlled gradient is accumulated. This matches the existing ATOMIC
microbatch projection semantics and avoids allowing conflicting microbatches to
cancel before the decision.

## Telemetry

Every `observe` or `safe` step records:

```text
gradient/retrieval_norm
gradient/auxiliary_norm
gradient/auxiliary_controlled_norm
gradient/auxiliary_dot
gradient/auxiliary_cosine
gradient/auxiliary_compatible
gradient/auxiliary_cosine_scale
gradient/auxiliary_norm_scale
gradient/auxiliary_total_scale
```

The existing alpha distributions, confidence buckets, main loss, raw SimCSE,
weighted SimCSE, and auxiliary-to-main ratio remain unchanged.

The run manifest records:

```text
objective_contract.auxiliary_gradient = off | observe | cosine_safe
objective_contract.auxiliary_norm = unbounded | retrieval_relative
```

## Numerical Cases

- If either shared gradient norm is at most `eps`, report cosine and dot as
  zero. A zero retrieval norm suppresses the shared auxiliary in `safe` mode.
- A zero auxiliary norm produces scale zero and leaves the primary update
  unchanged.
- A negative or zero dot product produces compatibility scale zero.
- Gradient statistics are accumulated in float32 even when model gradients use
  a lower precision.
- Non-finite gradient statistics fail the step with a descriptive error instead
  of silently applying an uncontrolled update.
- Parameters touched only by `head_loss` retain their head gradients unchanged.

## Rejected Alternatives

### Replace alpha by gradient cosine

Rejected because alpha and cosine answer different questions. Alpha estimates
which queries need regularization; cosine determines whether the aggregate
regularizer currently helps retrieval.

### Representation-space cosine

Rejected for the first claimed method because it does not establish the stated
parameter-update guarantee. It remains a possible low-cost ablation.

### PCGrad as the primary controller

One-sided projection is first-order safe, but it retains the entire orthogonal
auxiliary component. The cosine gate is more conservative and the norm cap
directly limits curvature risk. PCGrad remains an ablation after the safe gate
is understood.

### Bilevel validation weighting

Meta-learning auxiliary weights through a virtual validation step is more
directly aligned with held-out retrieval but requires higher-order optimization
and a clean validation stream. It is too expensive and confounded for the next
M5-scale experiment.

### Reintroduce the memory bank immediately

Rejected because the modern bank is already a failed mechanism. Projection
cannot repair stale or false-negative semantics when gradients are not in
direct conflict. Bank denoising is a separate research phase.

## Preregistered Experiment

### Diagnostic

Run `observe` with the current best non-saturated alpha temperatures and
`lambda_sc=0.03`. Measure compatibility rate, cosine distribution, norm ratio,
and their relationship to alpha-confidence buckets.

The mechanism hypothesis is supported if compatibility varies materially over
training and incompatible steps are not negligible. If conflicts are nearly
absent, a safe controller cannot explain or repair the observed HR@1 trade-off.

### Matched Comparison

Compare:

```text
V0: vanilla, auxiliary_gradient.mode=off
A0: atom_gate, auxiliary_gradient.mode=off
A1: atom_gate ablation, auxiliary_gradient.mode=safe
```

Hold model, LoRA config, split, sampled rows, seed, optimizer, microbatch,
accumulation, epoch count, retrieval temperature, target temperature, SimCSE
temperature, and evaluation corpus fixed. Memory is disabled.

Use Qwen3-Embedding-0.6B LoRA on `justatom` and mMARCO first. Use at least two
training seeds for any positive claim. Habr IR is an external-domain validation
dataset only after its release split and relevance audit are frozen.

Primary metrics are HR@1 and MRR@10. HR@5, HR@10, and nDCG@10 are secondary.
Report paired per-query rank differences and bootstrap confidence intervals, not
only aggregate point estimates.

The method advances only if:

1. the objective decomposition and first-order inequalities pass unit tests;
2. telemetry proves the controller is active and finite;
3. A1 does not regress the mean primary metrics against A0 across matched seeds;
4. A1 is compared against V0 without changing the primary contrastive kernel.

Failure to improve evaluation metrics does not falsify the first-order
optimization result. It falsifies the stronger empirical hypothesis that this
particular auxiliary and controller improve retrieval generalization.

## Literature Basis

- Du et al., *Adapting Auxiliary Losses Using Gradient Similarity*, 2018/2020.
- Yu et al., *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020.
- Gao et al., *SimCSE*, EMNLP 2021.
- Wang and Isola, *Alignment and Uniformity on the Hypersphere*, ICML 2020.
- Ren et al., *Learning to Reweight Examples for Robust Deep Learning*, ICML
  2018.
- Qu et al., *RocketQA*, NAACL 2021.

## Out of Scope

- changing the canonical meanings of `atom_gate` or `atomic`;
- combining alpha, safe SimCSE, and the memory bank in one first experiment;
- per-query full-parameter gradients;
- Hessian-vector products or a learned optimizer;
- claiming a guarantee for validation metrics or finite AdamW steps;
- publishing a new model before multi-seed and multi-dataset confirmation.
