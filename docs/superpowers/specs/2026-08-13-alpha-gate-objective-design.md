# Alpha-Gate Objective: Coupled Baseline and Detached Control

**Date:** 2026-08-13

**Branch:** `fix/alpha-gate-objective`

**Status:** approved design

## 1. Problem

The canonical `atom_gate` objective currently uses one query-conditional
coefficient for two different roles:

1. mixing semantic and lexical pair scores;
2. attenuating the SimCSE auxiliary loss.

For query `i`, the relevant part of the current objective is

\[
L_i = L_{\mathrm{main},i}
    + (1-\alpha_i)L_{\mathrm{aux},i}
    + \lambda_{\mathrm{mix}}L_{\mathrm{mix},i}(\alpha_i).
\]

Its direct gradient with respect to the controller is

\[
\frac{\partial L_i}{\partial \alpha_i}
= -L_{\mathrm{aux},i}
+ \lambda_{\mathrm{mix}}
  \frac{\partial L_{\mathrm{mix},i}}{\partial \alpha_i}.
\]

With standard coupled contrastive cross-entropy,
`L_aux >= 0`. The auxiliary path therefore rewards increasing `alpha` even
when that increase has no semantic or lexical justification. At `alpha = 1`,
the head can disable SimCSE. Changing DCL to coupled InfoNCE does not remove
this degeneracy; it only makes the sign of the shortcut unambiguous.

There is a second comparison problem: the canonical methods do not currently
share one contrastive kernel. `vanilla` and `atom_gate` inherit DCL, whereas
`atomic` explicitly uses coupled InfoNCE. This confounds method comparisons.

## 2. Decision

Use coupled InfoNCE as the canonical contrastive kernel for all three public
training methods, and treat `alpha(q)` as a detached control value only on the
SimCSE weighting path.

The new `atom_gate` loss is

\[
\boxed{
L_i = L_{\mathrm{InfoNCE},i}
    + \left(1-\operatorname{sg}[\alpha_i]\right)
      \lambda_{\mathrm{aux}}L_{\mathrm{SimCSE},i}
    + \lambda_{\mathrm{mix}}L_{\mathrm{mix},i}(\alpha_i)
    - \lambda_H H(\alpha_i)
}
\]

where `sg` is the stop-gradient operator. The entropy term remains optional
and keeps its existing configuration and default.

This gives the controller one coherent source of supervision:

\[
\frac{\partial L_i}{\partial \alpha_i}
= \lambda_{\mathrm{mix}}
  \frac{\partial L_{\mathrm{mix},i}}{\partial \alpha_i}
  - \lambda_H\frac{\partial H(\alpha_i)}{\partial \alpha_i}.
\]

The encoder still receives the query-dependent auxiliary gradient

\[
\frac{\partial L_i}{\partial \theta}\bigg|_{\mathrm{aux}}
= \left(1-\operatorname{sg}[\alpha_i]\right)
  \lambda_{\mathrm{aux}}
  \frac{\partial L_{\mathrm{SimCSE},i}}{\partial \theta},
\]

but neither the alpha head nor the encoder can reduce the current auxiliary
loss through the derivative of the gate itself.

## 3. Canonical Method Contract

All canonical methods use `objective.decoupled: false`:

| Method | Primary objective | Additional mechanism |
|---|---|---|
| `vanilla` | coupled InfoNCE | none |
| `atom_gate` | coupled InfoNCE | detached alpha-controlled SimCSE plus semantic/lexical pair mixing |
| `atomic` | coupled InfoNCE | FIFO memory negatives with projected memory gradients |

This is a structural contract, not merely a YAML default. Canonical method
resolution must produce it even when the caller supplies only the method name.
Explicit incompatible overrides in a canonical experiment must be rejected.
Research ablations may enable DCL only under `experiment.role: ablation`.

## 4. Alpha Data Flow

For each training batch:

1. The encoder produces normalized query and positive embeddings.
2. `QueryAlphaGate` produces one `alpha_i` per query.
3. Semantic positive/negative scores come from cosine similarity.
4. Lexical positive/negative scores come from the normalized lexical lookup.
5. The live `alpha` enters `L_mix`, which trains the controller to choose the
   useful semantic/lexical contribution for each query.
6. `alpha.detach()` enters the SimCSE coefficient `1 - alpha.detach()`.
7. Telemetry records the live alpha distribution and the effective auxiliary
   weight distribution without changing either graph.

The detach is local to auxiliary weighting. It must not be applied to the
semantic/lexical mixing objective or to the optional entropy regularizer.

## 5. Configuration

No new public hyperparameters are introduced.

Existing fields retain their meanings:

- `objective.simcse_dropout_weight`: global SimCSE scale
  `lambda_aux`;
- `alpha_gate.mix_weight`: weight of semantic/lexical pair supervision;
- `alpha_gate.mix_weight_warmup_steps`: linear warmup for pair supervision;
- `alpha_gate.entropy_weight`: optional anti-saturation regularizer.

For the canonical `atom_gate` profile:

```yaml
objective:
  decoupled: false
  simcse_dropout_weight: 0.1

alpha_gate:
  enabled: true
  mix_weight: 0.3
```

The implementation must not add a mean-alpha target, budget penalty, second
gate head, or another loss coefficient in this change.

## 6. Telemetry

Keep the existing `alpha/*` distribution and loss metrics. Add an effective
auxiliary-weight distribution under `alpha_aux_weight/*`, computed as
`1 - alpha.detach()`.

The minimum required observables are:

- alpha mean, minimum, maximum, and standard deviation;
- effective auxiliary-weight mean, minimum, maximum, and standard deviation;
- unweighted SimCSE loss;
- semantic/lexical mixing loss;
- main coupled InfoNCE loss.

These metrics allow experiments to distinguish a legitimate semantic-heavy
alpha distribution from optimization collapse.

## 7. Validation

Implementation follows test-first development.

### 7.1 Objective gradient contract

A focused unit test must demonstrate all of the following:

- changing alpha changes the forward value of the weighted auxiliary term;
- the auxiliary-only path produces no alpha gradient;
- the auxiliary-only path still produces encoder gradients;
- adding semantic/lexical pair supervision produces a finite, non-zero alpha
  gradient.

### 7.2 Method resolution

Tests must assert that canonical `vanilla`, `atom_gate`, and `atomic` all
resolve to `decoupled == false`. A canonical DCL override must fail with a
clear message, while the same override remains available for an explicit
ablation role.

### 7.3 Regression coverage

The existing tests for vanilla training, alpha telemetry, lexical score
recovery, memory-bank projection, configuration serialization, and CLI
resolution must remain green. The complete offline test suite is required
before publication.

## 8. Experimental Consequence

Metrics produced before this change used a different optimization contract
and must not be silently pooled with new canonical results. New benchmark
runs must record the resolved `decoupled` value and the alpha detach policy in
their metadata.

The first matched comparison after implementation is:

1. same model checkpoint;
2. same dataset split and seed;
3. same batch size, epoch count, optimizer, temperature, and LoRA settings;
4. `vanilla` versus `atom_gate` versus `atomic`;
5. at least two seeds for any result used as dissertation evidence.

The scientific claim is limited to query-conditioned regularization whose
controller is supervised by retrieval evidence. This design does not claim
that unconstrained loss attenuation is useful.

## 9. Out of Scope

- a separate `w(q)` auxiliary-weight head;
- a batch-level alpha budget or target mean;
- query-conditional temperature;
- changes to ATOMIC gradient projection or memory selection;
- inference-time use of the alpha head;
- reinterpretation of historical benchmark tables.
