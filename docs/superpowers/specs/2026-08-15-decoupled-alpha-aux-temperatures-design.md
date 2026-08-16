# Decoupled Alpha-Target and Auxiliary Temperatures

## Status

Approved for implementation on 2026-08-15 after the Qwen3-Embedding-0.6B
screening exposed saturation in both the alpha target and the SimCSE auxiliary.

## Problem

The current atom-gate objective reuses the retrieval temperature for three
different jobs:

```text
tau_retrieval = tau_alpha_target = tau_simcse = 0.05.
```

This is mathematically valid but empirically degenerate for an already strong
Qwen embedding model. On the clean mMARCO-ru screening run:

```text
mean(alpha_target) = 0.981255
mean(alpha) = 0.973551
mean(1 - alpha) = 0.026449
mean(L_SimCSE) = 4.47e-6
```

Nearly every query is labeled high-confidence, the gate learns alpha close to
one, and the auxiliary is already saturated before the gate suppresses it.
Additional epochs would therefore train almost the same InfoNCE objective while
changing the dropout random-number trajectory.

## Decision

Keep retrieval temperature unchanged and introduce two optional fixed
temperatures:

```yaml
objective:
  temperature: 0.05
  simcse_temperature: 0.2

alpha_gate:
  enabled: true
  target_temperature: 0.2
```

Both new fields default to `null`. A null value preserves the existing behavior
exactly by reusing the live retrieval temperature. This keeps historical
configs and checkpoints valid and makes the new behavior an explicit,
ablation-friendly choice.

Changing the global retrieval temperature was rejected because it changes the
baseline objective and requires a new matched baseline. Increasing only the
SimCSE weight was rejected because it compensates for a saturated loss with an
arbitrarily large multiplier while leaving the confidence target saturated.

## Objective

The main retrieval loss remains:

```text
S_ij = cosine(q_i, p_j)
L_main,i = -log softmax(S_i / tau_retrieval)_i.
```

The detached alpha target becomes:

```text
t_i = stop_gradient(softmax(S_i / tau_target)_i),
tau_target = alpha_gate.target_temperature ?? tau_retrieval.
```

The SimCSE term becomes:

```text
L_sc,i = -log softmax(cosine(q_i, q'_j) / tau_sc)_i,
tau_sc = objective.simcse_temperature ?? tau_retrieval.
```

The complete loss is unchanged apart from those temperatures:

```text
alpha_i = sigmoid(gate(stop_gradient(q_i)))

L_i = L_main,i
    + lambda_sc * (1 - stop_gradient(alpha_i)) * L_sc,i
    + lambda_alpha * BCEWithLogits(alpha_logit_i, t_i).
```

The autograd contract remains:

| Term | Encoder gradient | Alpha-head gradient |
| --- | --- | --- |
| coupled InfoNCE | yes | no |
| weighted SimCSE | yes | no |
| gate BCE | no | yes |

The fixed target temperature is detached and has no optimizer state. The fixed
SimCSE temperature is represented by a non-learnable contrastive kernel. The
retrieval temperature remains independently learnable when configured so.

## Configuration Contract

Add the fields:

```text
ObjectiveConfig.simcse_temperature: float | None = None
AlphaGateConfig.target_temperature: float | None = None
```

Non-null values must be finite and strictly positive. Dotted CLI overrides are:

```text
--objective.simcse-temperature 0.2
--alpha-gate.target-temperature 0.2
```

No new preset or method name is introduced. The resolved values are already
captured by the existing run manifest and research checkpoint configuration.

## Data Flow

1. The encoder produces `queries`, `positives`, and the second dropout query
   view when SimCSE is enabled.
2. The alpha head consumes `queries.detach()` and emits logits.
3. `ContrastiveObjective` computes main InfoNCE with the retrieval kernel.
4. The objective computes SimCSE with either the retrieval kernel (`null`) or a
   fixed-temperature auxiliary kernel.
5. The objective computes the detached confidence target with either the live
   retrieval temperature (`null`) or the configured fixed target temperature.
6. The existing detached alpha coefficient combines main and auxiliary terms.

No memory-bank path changes in this work.

## Telemetry

Preserve existing metric names and add:

```text
temperature/simcse
temperature/alpha_target
loss/alpha_aux_weighted
loss/alpha_aux_to_main_ratio
```

`loss/alpha_aux` remains the unweighted SimCSE loss for backward compatibility.
The weighted metric is the mean of
`lambda_sc * (1 - alpha.detach()) * L_SimCSE`. The ratio uses a small numerical
epsilon and is telemetry-only.

## Compatibility

- Old YAML without either field resolves to `null` and remains behaviorally
  identical.
- Old checkpoints restore through dataclass defaults without a schema bump.
- Existing metric columns remain unchanged; CSV logging adds columns only when
  alpha/SimCSE is active.
- The retrieval loss and learnable retrieval temperature are untouched.

## Tests

1. Config parsing accepts `null` and positive finite values and rejects zero,
   negative, NaN, and infinity.
2. A fixed target temperature changes only the detached target and BCE term,
   not main InfoNCE.
3. A fixed SimCSE temperature changes the auxiliary value and encoder gradient,
   not main InfoNCE.
4. Null temperatures reproduce legacy objective outputs exactly.
5. Module plumbing forwards the target temperature and emits all new telemetry.
6. Historical checkpoint configuration without the fields still restores.

## Experiment Gate

First run only the new alpha variant on the existing mMARCO-ru seed-44
development screening. Reuse the already saved matched InfoNCE baseline:

```text
train: 6000 pairs from 50000 train rows
eval: 1000 dev queries, 1000 positives, 9000 corpus negatives
model: Qwen/Qwen3-Embedding-0.6B
LoRA: r=16, alpha=32, dropout=0.05, RS-LoRA, all-linear
tau_retrieval=0.05, tau_target=0.2, tau_simcse=0.2
```

Before retrieval metrics are interpreted, the mechanism must satisfy:

```text
mean(alpha_aux_weight) >= 0.10
mean(L_SimCSE) >= 1e-3
mean(weighted_alpha_aux) / mean(L_main) >= 0.01
```

The retrieval screening rule remains: `HitRate@1` and `MRR@10` must both be no
worse than the matched baseline, and at least one must improve. If the mechanism
and retrieval gates pass, repeat the screening at seed 45. Only a replicated
configuration advances to the complete 5000-query / 50000-document evaluation.

Failed configurations remain documented negative ablations. They are not
combined with the failed memory-bank variants.
