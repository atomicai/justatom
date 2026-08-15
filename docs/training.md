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
t_i = stop_gradient(softmax(S / tau)_ii)
alpha_i = sigmoid(MLP(stop_gradient(q_i)))
L_i = L_InfoNCE,i
    + (1 - stop_gradient(alpha_i)) lambda_sc L_SimCSE,i
    + lambda_alpha BCE(alpha_i, t_i)
```

The gate cannot lower the current SimCSE loss by moving toward one. Its BCE
term trains the head against retrieval confidence while leaving the encoder
path detached. The gate is training-only; the saved encoder has the same
inference interface as the source embedding model.

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
