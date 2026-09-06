# Training

Justatom has one production training path and three public methods.

| Method | Primary objective | Auxiliary objective | Gradient control |
| --- | --- | --- | --- |
| `vanilla` | coupled InfoNCE | none | none |
| `atom_gate` | coupled InfoNCE | confidence-supervised query-controlled SimCSE | detached alpha target and head input |
| `atomic` | coupled InfoNCE | detached online memory | one-sided orthogonal projection |

## Objective

For a batch of query embeddings `Q` and positive document embeddings `P`, the
in-batch similarity matrix is

```text
S = Q P^T,       S_ij = cosine(q_i, p_j).
```

All three canonical methods use the same coupled InfoNCE row loss:

```text
L_i = -S_ii / tau + log sum_j exp(S_ij / tau).
```

`atom_gate` learns a query-only scalar from a detached query representation,
with its target derived from the detached in-batch positive retrieval
confidence:

```text
t_i = stop_gradient(softmax(S / tau_target)_ii)
alpha_i = sigmoid(MLP(stop_gradient(q_i)))
L_i = L_InfoNCE,i
    + (1 - stop_gradient(alpha_i)) lambda_sc L_SimCSE,i
    + lambda_alpha BCEWithLogits(alpha_logit_i, t_i)
```

The gate cannot lower the current SimCSE loss by moving toward one. Its BCE
term trains the head against retrieval confidence while leaving the encoder
path detached. The gate is training-only; the saved encoder has the same
inference interface as the source embedding model.

The retrieval, confidence-target, and SimCSE temperatures can be separated
without changing the primary retrieval objective:

```yaml
objective:
  temperature: 0.05
  simcse_temperature: 0.2

alpha_gate:
  target_temperature: 0.2
```

Both auxiliary fields default to `null`. In that compatibility mode,
`tau_target` and `tau_simcse` reuse the live retrieval temperature exactly.
A non-null value is a fixed, non-learnable auxiliary temperature. The target
temperature must be finite and greater than zero. The SimCSE temperature must
also fit the contrastive kernel range `[1e-3, 1.0]`, so it is never silently
clamped to a value different from the manifest. The dotted CLI overrides are
`--objective.simcse-temperature 0.2` and
`--alpha-gate.target-temperature 0.2`. The value `0.2` is an experimental
setting for reducing target and SimCSE saturation, not a canonical method
default.

The alpha head still consumes `stop_gradient(q_i)`. The target is detached,
and the auxiliary multiplier uses `stop_gradient(alpha_i)`, so gate BCE updates
only the head while SimCSE updates both query views in the encoder. Training
telemetry exposes `temperature/simcse`, `temperature/alpha_target`, the raw
`loss/alpha_aux`, its actual contribution as `loss/alpha_aux_weighted`, and
the signed contribution ratio `loss/alpha_aux_to_main_ratio`. These additional
columns are emitted only when the corresponding alpha or SimCSE path is active.

### Gradient-safe auxiliary ablation

`atom_gate` ablations may control the encoder-side SimCSE gradient against the
retrieval gradient for every contrastive microbatch. Let `g_r` be the retrieval
gradient over shared encoder parameters and `g_a` the detached-alpha-weighted
SimCSE gradient. In `safe` mode, the controller computes

```text
c = max(0, dot(g_r, g_a) / max(||g_r|| ||g_a||, eps))
s = min(1, max_norm_ratio ||g_r|| / (c ||g_a|| + eps))
g_a_safe = c s g_a
```

When `dot(g_r, g_a) <= 0` or either norm is at most `eps`, it sets both scales
to zero. This bounds `||g_a_safe|| <= max_norm_ratio ||g_r||`; more importantly,
the shared encoder update has the first-order boundary
`dot(g_r, g_r + g_a_safe) >= ||g_r||^2`. The controller therefore never adds an
auxiliary component that opposes the retrieval descent direction to first order.
The alpha BCE/head gradient is retained unchanged on its separate parameters.

`observe` performs the same per-microbatch manual capture and emits the same
statistics, but applies the auxiliary gradient unchanged. Controller telemetry
is `gradient/retrieval_norm`, `gradient/auxiliary_norm`,
`gradient/auxiliary_controlled_norm`, `gradient/auxiliary_dot`,
`gradient/auxiliary_cosine`, `gradient/auxiliary_compatible`,
`gradient/auxiliary_cosine_scale`, `gradient/auxiliary_norm_scale`, and
`gradient/auxiliary_total_scale`.

Use `observe` before a safe run to collect the compatibility distribution:

```bash
bash scripts/run_pipeline.sh \
  --train-config configs/experiments/qwen3-06b-lora-alpha-gradient-safe.yaml \
  --method atom_gate \
  --experiment-role ablation \
  --dataset-ids justatom \
  --model Qwen/Qwen3-Embedding-0.6B \
  --batch-size 8 \
  --grad-acc-steps 4 \
  --epochs 1 \
  --nsamples 3000 \
  --temperature 0.05 \
  --aux-gradient-mode observe \
  --aux-gradient-max-norm-ratio 0.25 \
  --aux-gradient-eps 1e-12 \
  --wandb-mode disabled
```

`--train-config` replaces only the pipeline's `--config configs/train.yaml`
argument. Pipeline defaults and explicit shell options still override values in
the selected YAML. Run manifests expose this decision under `objective_contract`:
`auxiliary_gradient` is `off`, `observe`, or `cosine_safe`, while
`auxiliary_norm` is `unbounded` except for `safe`, which records
`retrieval_relative`.

## ATOMIC: protected online memory

`atomic` keeps a FIFO queue `B` of detached document embeddings. The bank does
not retain previous autograd graphs. Unlike an ordinary memory-bank objective,
ATOMIC does not place the extra negatives directly into the protected primary
loss. It decomposes the objective exactly into:

```text
L_primary = InfoNCE(Q, P)
L_memory  = InfoNCE(Q, P, B) - InfoNCE(Q, P)
```

For a query `i` with `K_i` selected bank candidates in a contrastive
microbatch of `N` pairs, the augmented denominator is count-normalized:

```text
L_aug,i = -z_ii + log(
  exp(z_ii) + A_batch,i + lambda(t) (N - 1) / K_i A_bank,i
)
```

The selected-bank term is omitted when `K_i = 0`. `lambda(t)` applies the
configured memory-mass ramp, so `mass_ratio` controls the normalized bank mass
without changing its meaning when the number of selected candidates changes.

Let `g_p` and `g_m` be their gradients over the trainable parameters. The
primary gradient is protected. When the memory gradient conflicts with it,
ATOMIC removes only the opposing component:

```text
if dot(g_p, g_m) < 0:
    g_m <- g_m - dot(g_p, g_m) / ||g_p||^2 * g_p

g_update = g_p + lambda_memory * g_m
```

The projected memory component is orthogonal to `g_p`, so it cannot oppose the
primary descent direction to first order. Aligned memory gradients are retained
unchanged. Parameters owned only by an optional memory-side head also retain
their gradients. Projection is training-only and adds no inference components.

Gradient accumulation is performed by the ATOMIC manual optimization step so
each microbatch is projected before its update is accumulated. The same path is
implemented with ordinary PyTorch operations and works on CUDA, MPS, and CPU.

## Canonical Profiles

Selecting a method applies its registered defaults before YAML and CLI
overrides. Canonical `vanilla`, `atom_gate`, and `atomic` share coupled
InfoNCE so method comparisons do not change the primary contrastive kernel.
Using decoupled InfoNCE requires `experiment.role: ablation` explicitly.

Canonical `atomic` adds a 512-entry FIFO bank, 50 optimizer-step warmup, 12
random candidates per query, and memory weight `1.0`. It does not construct
the alpha gate or a query-margin head. Structural additions must be labeled as
ablations:

Run manifests record the resolved kernel, alpha gradient policy, and memory
mass policy under `objective_contract`; enabled banks declare
`memory_mass: count_normalized`, while disabled banks declare
`memory_mass: not_applicable`. Their `batch_contract` records the contrastive
microbatch, gradient accumulation, and optimizer effective batch as
`contrastive_microbatch`, `gradient_accumulation`, and
`optimizer_effective_batch`, respectively. Results produced before this
contract was introduced used different canonical objectives and must not be
pooled with or directly compared against new runs. Re-run matched methods with
the same model, split, seed, batch size, optimizer, and epoch count before
drawing method-level conclusions.

```bash
python -m justatom.api.train \
  --config configs/train.yaml \
  --method atomic \
  --experiment.role ablation \
  --memory-bank.adaptive.enabled true \
  --memory-bank.margin.mode constant
```

A plain InfoNCE control with detached bank negatives keeps the `vanilla`
method identity but must also be labeled as an ablation. It does not construct
an alpha gate or query-margin head:

```bash
python -m justatom.api.train \
  --config configs/train.yaml \
  --method vanilla \
  --experiment.role ablation \
  --objective.decoupled false \
  --optimization.epochs 1 \
  --optimization.num-samples 3000 \
  --memory-bank.enabled true \
  --memory-bank.size 512 \
  --memory-bank.mining random \
  --memory-bank.random-negatives 12 \
  --memory-bank.adaptive.enabled false \
  --memory-bank.margin.mode off
```

## Commands

Train one method:

```bash
python -m justatom.api.train \
  --config configs/train.yaml \
  --method atomic \
  --dataset.id justatom \
  --optimization.batch-size 32 \
  --optimization.epochs 2
```

Run the full retrieval pipeline:

```bash
bash scripts/run_pipeline.sh \
  --method atomic \
  --dataset-ids justatom \
  --model intfloat/multilingual-e5-small \
  --batch-size 32 \
  --epochs 2 \
  --auto-e5-prefixes
```

Compare all three methods with the same model and datasets:

```bash
bash scripts/run_benchmark.sh \
  --dataset-ids justatom,meme-russian-ir \
  --model intfloat/multilingual-e5-small \
  --variants vanilla,atom_gate,atomic \
  --batch-size 32 \
  --epochs 2 \
  --auto-e5-prefixes
```

## LoRA adapters

LoRA is an encoder configuration, so it composes with every training method.
The objective and memory-gradient projection do not change; the optimizer
simply updates PEFT adapter parameters instead of the frozen backbone
parameters. For ATOMIC, projection is therefore applied in adapter-parameter
space, including the learnable temperature.

The default `all-linear` target is resolved by PEFT from the Hugging Face model
itself. This keeps the same config usable for Qwen3-Embedding, E5, BGE, and
mBERT. An explicit list is still available for controlled experiments.
`justatom/pfbert` is intentionally unsupported because it is not a Hugging Face
encoder.

```yaml
method: atomic

model:
  name_or_path: Qwen/Qwen3-Embedding-0.6B
  query_prefix: |-
    Instruct: Given a web search query, retrieve relevant passages that answer the query
    Query:
  content_prefix: ""
  lora:
    enabled: true
    rank: 16
    alpha: 32
    dropout: 0.05
    target_modules: all-linear
    use_rslora: true
    bias: none

optimization:
  lr_encoder: 0.00002

runtime:
  accelerator: auto
  precision: auto
  gradient_checkpointing: true
```

This is ordinary Hugging Face PEFT; it does not require Unsloth, bitsandbytes,
or a quantized base model. With `precision: auto`, CUDA uses BF16 when the GPU
supports it and otherwise FP16. MPS and CPU default to FP32 for compatibility;
`16-mixed` can be selected explicitly on a supported Mac. Gradient
checkpointing is independent of LoRA and can be enabled when sequence length
or batch size needs more memory.

The reproducible Qwen3 0.6B vanilla-plus-bank control is available at
`configs/experiments/qwen3-06b-lora-vanilla-bank.yaml`. It uses standard
coupled InfoNCE, 3,000 sampled pairs, one epoch, and 12 random detached bank
negatives per query. Override `dataset.id` and `artifacts.save_dir` on the
command line to reuse the recipe.

The matching gradient-safe `atom_gate` ablation is
`configs/experiments/qwen3-06b-lora-alpha-gradient-safe.yaml`. It keeps the
same Qwen3 LoRA, data, optimization, and runtime values, disables the memory
bank, and fixes `tau=0.05`, `tau_simcse=0.2`, `tau_target=0.2`,
`lambda_sc=0.03`, and the `safe` controller ratio at `0.25`.

### Qwen3-VL-Embedding-2B: text retrieval

`Qwen/Qwen3-VL-Embedding-2B` is supported for **text-only** training and local
retrieval through native Transformers (>=4.57) and standard PEFT. No vision
processor, `qwen-vl-utils`, Unsloth or quantization package is required for this
path. Image/video training and multimodal serving are not implemented here.

`ITokenizer` applies the checkpoint's chat template with the default system
instruction `Represent the user's input.` and an assistant generation prefix,
following the [official embedder](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/blob/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda/scripts/qwen3_vl_embedding.py).
This happens for both training and local retrieval, including exported encoders.
Keep `query_prefix` and `content_prefix` empty in this recipe; nonempty prefixes
are prepended to the user text, not substituted into the system instruction.
Sequence limits apply to the complete formatted prompt, with the same right
truncation as the official text path. Last-valid-token pooling yields normalized
2048-dimensional embeddings; optional MRL dimensions range from 64 to 2048.

For this architecture, `all-linear` resolves only to language-tower linear
layers. The visual tower remains frozen, and LoRA requires `bias: none`.
Geometry AnchorBank uses the same adapter-disabled frozen base and the same
one-sided projection as before: no reranker or negative-bank denominator is added.

Two matched starting configs are provided:

```bash
conda activate justatom-env
python -m justatom.api.train --config configs/experiments/qwen3-vl-2b-lora-vanilla.yaml
python -m justatom.api.train --config configs/experiments/qwen3-vl-2b-lora-geometry-anchor-bank.yaml
```

These are starting configs, not evaluated benchmark results: one seed/epoch,
3,000 sampled JustAtom pairs, rank 16 / alpha 32 / rsLoRA, learning rate 2e-5,
query/document lengths 128/512 and a microbatch of 4. Accumulation
of 8 gives 32 pairs per optimizer update, **not** 32 in-batch candidates. Hold the
microbatch fixed in matched comparisons, and establish a disjoint train/dev/test
protocol separately before measuring quality. Use a fresh `artifacts.save_dir`
for each real run. The configs disable the large optional research checkpoint;
the PEFT adapter and merged encoder are still exported.

A four-step CUDA smoke at microbatch 8 / document length 512 passed on a 24 GB
RTX 5090 Laptop, but reserved about 23.2 GiB. The starting configs use microbatch
4 (about 13.7 GiB reserved in the same smoke) to leave headroom; that changes the contrastive negative count, so it must also
be used in the matched control. Short synthetic smokes are not long-run memory or
thermal guarantees.

`model.revision` pins both model and tokenizer. `model.dtype` optionally controls
CUDA backbone storage (`float32`, `float16`, `bfloat16`); null preserves existing
loading behavior. CPU/MPS use float32 when this field is set. This is distinct
from `runtime.precision`, which controls Lightning's numerical execution mode.
The VL configs use BF16 storage on CUDA and non-reentrant gradient checkpointing.
AnchorBank's deterministic eval-mode student view retains its graph, so measure
its memory separately; gradient checkpointing does not eliminate that overhead.

With the pinned model already cached, run a short offline synthetic check:

```bash
python scripts/smoke_qwen3_vl_lora.py --device cuda
```

This checks live adapter updates and the active geometry constraint for both
configs, logs batch metrics and peak CUDA allocated/reserved memory, and saves
manifests under a fresh `.tmp_runs/qwen3-vl-lora-smoke-*` directory. It does not
download models, save large weights, or estimate retrieval quality. `--device cpu`
and `--device mps` use the same path; actual MPS validation needs Apple hardware.

## Artifacts

Every successful training run writes:

- `encoder/`: deployable Hugging Face-compatible encoder artifact
- `adapter/`: PEFT adapter and its config when LoRA is enabled
- `research/checkpoint.pt`: complete training state for analysis or continuation
- `run_manifest.yaml`: resolved method, data, seed, hyperparameters, and Git state
- `batch_metrics.csv`: losses, retrieval ranks, bank geometry, gradient cosine, conflict flag, projection coefficient, and gradient norms

The adapter and research checkpoint are saved before LoRA is merged into the
deployable encoder. Loading `encoder/` therefore does not require PEFT, while
`adapter/` retains the small reusable delta for reproducibility.

The benchmark additionally writes commands, retrieval tables, geometry tables,
and process RSS snapshots. These files are the reproducibility boundary for
reported experiments.
