# Query-Conditional Contrastive Hyperparameters (QC-CH) — `m(q)` margin head, unified with `tau(q)`

> Design spec. Status as of 2026-06-15.
> Branch: `feature/qc-ch-margin-head` (off `master`).
> Scope: first iteration — minimal unification + factorial ablation (NOT the full
> 3-layer PID/EMA controller from `docs/research/adaptive-too-hard-margin.md`).

---

## 1. Motivation & evidence

The memory-bank false-negative filter `too_hard_margin` is a **global scalar**. Six
fixed-constant runs (see `VERDIKT.md`) show the bank effect is **sign-invariant per
dataset**:

| run | justatom | meme-russian-ir |
|---|---|---|
| 8+8 mixed | −4.18 | +2.58 |
| 4+12 mixed | −2.86 | +1.74 |
| random-only | −0.43 | +0.40 |
| cap=0.45 | −1.11 | +1.40 |
| cap=0.35 | −1.35 | +0.69 |

The optimal constant differs **by dataset** (justatom wants strict, meme wants lax).
A single scalar cannot take two values, so continuing fixed-constant sweeps is
provably exhausted. The next lever must be **dynamic (query-conditional)**, not another
constant.

### The unifying thesis

The project already has **two** query-conditional heads that replace a global scalar
with a learned function `h(q)`, trained jointly and dropped at eval:

- `alpha(q)` — the "atom_gate" auxiliary mixing weight.
- `tau(q)` — the "N3" per-query temperature.

This spec adds the third — `m(q)`, the per-query too-hard margin — and frames all three
under one theory: **Query-Conditional Contrastive Hyperparameters**.

---

## 2. The unifying frame

Every global scalar of InfoNCE becomes a small zero-initialized query head that
modulates the **same** geometric object — the effective negative cap `C(q)` on the unit
hypersphere — through a different mechanism:

| Head | Hyperparameter | Mechanism on the cap | Status |
|---|---|---|---|
| `alpha(q)` | aux mixing weight | weight of the cap's gradient | built (atom_gate) |
| `tau(q)` | temperature | **soft** cap radius (Boltzmann; vol ∝ τ^((D−1)/2)) | built (N3) |
| `m(q)` | too-hard margin | **hard→soft** cutoff of cap membership | **this spec** |

**Geometric spine:** `tau` sets the soft boundary, `m` sets the (softened) hard
boundary, `alpha` weights the cap's gradient. Three control surfaces, one object.

**Regime condition (the testable teeth):** a head `h(q)` helps **iff** the base scalar
sits in the *smooth* (non-saturated) region of its effect function. N3's failure on the
saturated γ-train-only base (τ₀=0.025) is a **prediction** of this, not a counterexample
(see `VERDIKT.md` §3.6). This converts a negative result into evidence for the frame.

---

## 3. Architecture

### 3.1 The `m(q)` head (template = `tau(q)`)

A 2-layer MLP on the runner, mirroring the tau head at
[encoders.py:431-446](../../../justatom/running/encoders.py#L431-L446):

```
margin_head = Sequential(Linear(D, h), GELU, Linear(h, 1))   # final layer zero-init
m(q) = m0 + s_m * tanh(margin_head(q))                        # ADDITIVE (tau is multiplicative)
```

- Zero-init final layer ⇒ `m(q) == m0` at step 0.
- `m0` defaults to the current constant `too_hard_margin` (e.g. 0.05).
- Bounded: `m(q) ∈ [m0 − s_m, m0 + s_m]`, then clamped to `[0, 0.15]`.
- Accessors `margin_weights(q)` (mirror `tau_weights`,
  [encoders.py:629-647](../../../justatom/running/encoders.py#L629-L647)) and
  `margin_parameters()` (mirror `tau_parameters`,
  [encoders.py:489-492](../../../justatom/running/encoders.py#L489-L492)).
- **`margin_parameters()` MUST be added to `mixing_parameters()`**
  ([encoders.py:494-501](../../../justatom/running/encoders.py#L494-L501)) or the head
  silently never trains.

### 3.2 The soft gate lives in the LOSS, not the bank (gradient hazard)

**Critical finding.** `ContrastiveMemoryBank.get` computes `bank_sim`/`pos_sim` from
**detached** vectors ([memory_bank.py:164-174](../../../justatom/training/memory_bank.py#L164-L174)).
Computing the soft gate there gives `m(q)` **zero gradient** — a silent no-op.

Resolution: compute `m(q)` in the runner (live graph) and apply the soft admission
weight inside `info_nce`, where `memory_logits = q @ memory.T`
([loss.py:626](../../../justatom/training/loss.py#L626)) is already live in `q`:

```
w(d|q) = sigmoid( (pos_cos(q) − m(q) − sim(q,d)) / beta )
```

with `sim(q,d) = memory_logits` (pre-tau cosine, live), `pos_cos` the positive cosine
(live, `sim.diagonal()` pre-tau), `m(q)` live from the head. The bank is **not** the
gradient path; it only supplies the candidate set, the collision mask, and `g(q)`
telemetry.

### 3.3 Weighted log-sum-exp in the loss (loss has no weight hook today)

`info_nce` currently supports only a **hard** `masked_fill` to a large *finite* negative
(−1e9 / −1e4, deliberately finite for MPS backward stability,
[loss.py:502-508](../../../justatom/training/loss.py#L502-L508)). Add a weighted path in
the DCL in-batch branch ([loss.py:696-699](../../../justatom/training/loss.py#L696-L699)):

```
memory_logits = memory_logits + log(w.clamp_min(eps))   # bank columns only
```

- Choose `eps` so `log(eps)` ≈ the existing finite floor (avoid reintroducing −inf).
- The hard **key-collision** mask (same doc/content/query id) stays hard.
- Weight applies to **bank negative columns only** — never the positive, never in-batch.
- `has_negatives` bookkeeping ([loss.py:700](../../../justatom/training/loss.py#L700))
  uses a bool view (`w>0`), not the float weight.
- New `info_nce` kwargs: `memory_margin` (the `m(q)` [B] tensor) and `soft_beta`.

### 3.4 `soft_mode ∈ {hard, soft-const, soft}` — confound isolation

Turning `m(q)` on changes **two** things at once: (a) hard→soft cutoff, and
(b) const→learned margin. Zero-init does **not** make step-0 a no-op (hard is recovered
only as `beta→0`). The `soft-const` control isolates (a) from (b):

| mode | gate | margin | role |
|---|---|---|---|
| `hard` | hard cutoff (today) | constant `m0` | baseline / current behavior |
| `soft-const` | sigmoid, `beta` | constant `m0` | isolates "softening" |
| `soft` | sigmoid, `beta` | learned `m(q)` | the contribution |

### 3.5 Eval-drop is structural (no code needed)

`eval.py` loads a plain encoder; `retriever.py`/`indexer.py` never call any head
([map confirms](../../research/atom-gate-memory-bank-validation.md)). So `m(q)`, like
`tau(q)`, is dropped at eval for free. Do **not** persist or use it at inference.

### 3.6 Scope: Gamma/atom_gate trainer path only

`m(q)` (like `tau(q)`) is wired only into `_BaseGammaLightningTrainer` →
Uni/BiGamma. `EncoderOnlyLightningTrainer`
([trainer.py:1175-1489](../../../justatom/running/trainer.py#L1175-L1489)) is a separate
duplicate path with no tau/alpha wiring — **out of scope for v1**.

---

## 4. Critical hazards (guard explicitly)

1. **Detached gradient** (§3.2) — `m(q)` must be live; gate applied in loss.
2. **No weight hook** (§3.3) — must add weighted log-sum-exp; keep finite floor.
3. **Step-0 ≠ no-op** (§3.4) — softening changes admission even at `m0`; needs soft-const control.
4. **Two trainer classes** — only the Gamma path is touched; the second `memory_bank.get`
   call site ([trainer.py:1413](../../../justatom/running/trainer.py#L1413)) gets the new
   kwarg defaulted to `None` so it stays inert.
5. **`pos_sim` is gated** — only computed when `too_hard_margin is not None`
   ([memory_bank.py:167](../../../justatom/training/memory_bank.py#L167)); must compute it
   unconditionally whenever the soft gate or `g(q)` diagnostic is active.
6. **No per-row dataset label** — the batch carries only `*_key_id`/`input_ids`/
   `attention_mask`; the justatom-vs-meme `g(q)` split is **per-run**, not per-batch.

---

## 5. Edit plan by phase (file:line)

### Phase 0 — collision diagnostic `g(q)`, no head

Single file: [memory_bank.py:161-178](../../../justatom/training/memory_bank.py#L161-L178).
After `bank_sim` (166) and `pos_sim` (168-171), using the **pre-margin** `valid` mask:

```
bank_max_sim = bank_sim.masked_fill(~valid, -inf).max(dim=1).values
g_q = bank_max_sim - pos_sim.view(-1)
metrics.update(_scalar_distribution_metrics(g_q, "MemoryBankCollisionG"))
```

Seed `MemoryBankCollisionG*` in the metrics init block
([memory_bank.py:107-135](../../../justatom/training/memory_bank.py#L107-L135)); guard the
empty/warmup early-return. Run **two separate runs** (justatom, meme).

### Phase 1 — `m(q)` head + soft admission

- [encoders.py:431-446](../../../justatom/running/encoders.py#L431-L446) — `margin_head`
  (zero-init), `MARGIN_QUERY_CONDITIONAL`/`MARGIN_QUERY_SCALE` env or ctor kwargs.
- [encoders.py:489-501](../../../justatom/running/encoders.py#L489-L501) —
  `margin_parameters()`, add to `mixing_parameters()`.
- [encoders.py:629-647](../../../justatom/running/encoders.py#L629-L647) — `margin_weights(q)`.
- [trainer.py:784-798](../../../justatom/running/trainer.py#L784-L798) — compute
  `m_per_query` after the tau block; telemetry via `_scalar_distribution_metrics`
  ([trainer.py:36-58](../../../justatom/running/trainer.py#L36-L58)) as
  `ContrastiveLossMQuery*`; pass `m_per_query` + `beta` into `info_nce`.
- [loss.py:588-591](../../../justatom/training/loss.py#L588-L591) — new kwargs;
  [loss.py:696-699](../../../justatom/training/loss.py#L696-L699) — weighted log-sum-exp.
- [memory_bank.py:167-177](../../../justatom/training/memory_bank.py#L167-L177) — compute
  `pos_sim` unconditionally when gate active; disable hard line-172 cutoff when
  `soft_mode ≠ hard`.

### Phase 2 — plumbing + factorial

- Dataclass fields `memory_bank_margin_head: bool`, `memory_bank_soft_beta: float|None`,
  `memory_bank_soft_mode: str='hard'` — [trainer_jobs.py:105-113](../../../justatom/running/trainer_jobs.py#L105-L113);
  validate at 181-195; forward through **all three** `build_lightning_module` ctor sites
  (545-553, 635-643, 722-730) + metadata (271-279, 407-415).
- [train.py:381-397 + 304-320](../../../justatom/api/train.py#L381-L397) — read keys
  (both copies); [train.py:166-203](../../../justatom/api/train.py#L166-L203) — atom_gate map.
- `configs/train.yaml:77` + `builtins/configs/train.default.yaml:84` — defaults.
- `run_pipeline.sh` (39-40, 564-599, 748-773, 978-983) + `run_benchmark.sh`
  (35-43, 232-267, 435-452, 17+125-144, 469-500) — env, parse, validate, forward, variants.
- **`tau` axis**: driven via existing `TAU_QUERY_CONDITIONAL` env per-variant in
  `run_benchmark.sh` (no new tau config). **Verify env propagates** through
  `run_pipeline.sh` → python.

---

## 6. Experimental design

### 6.1 Collision diagnostic (Phase 0, pre-registered)

Run `g(q)` dump on justatom and meme separately. Compare **within-dataset spread** vs
**between-dataset shift**.

**Pre-registered prediction:**
- If within-justatom spread of `g(q)` is large (overlaps the boundary) → query-level
  signal → `m(q)` should have nonzero spread and yield a gain.
- If `g(q)` is tight within dataset but shifted between datasets → dataset-level →
  `m(q)` collapses toward a constant (spread ≈ 0, like `TauQueryStd`); `soft-const`
  ≈ `soft`.

### 6.2 Baselines

- **R0** — plain InfoNCE, **no bank**, fixed τ, no heads. Closes `VERDIKT.md` §6
  (all prior "Base" columns are cached off-the-shelf numbers). The honest yardstick.
- **R1** — `(alpha-off, tau-const, m=hard-const)` — all-global-scalar within the bank setting.

### 6.3 Factorial 2³ (α × τ × m)

Axes (per §3.6 / §5):
- `alpha`: off = `gamma_joint=False`; on = atom_gate recipe. (Note: true off is
  `gamma_joint=False`, not merely the loss-gate flag — the score-space mix also reads α.)
- `tau`: off = global τ₀; on = `TAU_QUERY_CONDITIONAL=1`.
- `m`: off = `hard` (constant); on = `soft` (`m(q)`). Plus a `soft-const` cell in the
  (α-off, τ-off) slice to isolate softening.

Layout:
- **Full 2³** on the dichotomy pair: `justatom`, `meme-russian-ir`.
- **Reduced** (R1 + all-on `(α,τ,m)`) on OOD: `electrical-engineering-ru`.
- **2 seeds** on the key justatom cells (acceptance criterion #4 from
  `atom-gate-memory-bank-validation.md`).
- All runs batch=32, 1 epoch (the regime where the recipe matters; see
  `best-recipe-evolution` memory).

### 6.4 What we measure

- **Marginal effect** of each head (vs R1).
- **τ×m interaction** — the non-identifiability check (§9). Requires the τ-only, m-only,
  AND τ+m cells; if combined gain ≈ max(individual), the two are substitutes.
- **Regime condition** — does each head help only where its base scalar is unsaturated.

Metrics: HR@1/@5/@10, MRR@10, NDCG@10, Δ vs R0 and vs R1. Telemetry:
`ContrastiveLossMQuery*` spread; `MemoryBankCollisionG*`; correlation `corr(m(q), g(q))`;
soft-admitted effective negative mass.

---

## 7. Success / acceptance criteria

1. R0 measured — every Δ now references an honest fine-tuned baseline.
2. `soft` beats `soft-const` on at least the dichotomy pair → learned margin adds value
   beyond softening (else the contribution is "softening", reported honestly).
3. `m(q)` does **not** collapse geometry (effective rank / anisotropy stable).
4. The τ×m interaction is characterized (complementary or substitute — either is a result).
5. The Phase-0 prediction is confirmed or refuted (and the refutation is reported).

---

## 8. Out of scope (YAGNI for v1)

- The 3-layer PID/EMA adaptive controller (`docs/research/adaptive-too-hard-margin.md`) —
  deferred; only the minimal zero-init `m(q)` head is built.
- `EncoderOnlyLightningTrainer` wiring (§3.6).
- `hard_similarity_cap` changes — left as an orthogonal hard cutoff.
- New tau config plumbing — tau stays env-driven.

---

## 9. Open questions / risks

- **Non-identifiability (the crux):** `tau(q)` and `m(q)` both shape the same cap (soft
  radius vs softened cutoff). A combined gain may not be attributable. The factorial is
  designed to detect this; if they are substitutes, that is itself a finding.
- `beta` choice: too small → recovers the hard cutoff (no softening signal); too large →
  gate near-uniform (m has no effect). Needs a small sweep or a principled default.
- `eps`/finite-floor interaction in the weighted log-sum-exp (MPS stability).
- Whether `m(q)` should also see `d+` (like `alpha_include_doc`) — kept query-only for v1.

---

## 10. References

- `VERDIKT.md` — the dichotomy evidence, the geometric (cap-volume) view, §6 baseline gap.
- `docs/research/atom-gate-memory-bank-validation.md` — validation plan, acceptance criteria.
- `docs/research/adaptive-too-hard-margin.md` — the deferred full controller; GloFND (ICML 2025).
- `docs/math/N3_query_conditional_tau.md` — the tau(q) formal definition (template).
</content>
</invoke>
