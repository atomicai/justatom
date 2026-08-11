# Training

Justatom has one production training path and three public methods.

| Method | Objective | Query gate | Memory bank | Query margin |
| --- | --- | --- | --- | --- |
| `vanilla` | decoupled InfoNCE | no | no | no |
| `atom_gate` | InfoNCE plus query-controlled auxiliary pressure | `alpha(q)` | no | no |
| `atomic` | `atom_gate` plus adaptive bank negatives | `alpha(q)` | yes | `m(q)` |

## Objective

For a batch of query embeddings `Q` and positive document embeddings `P`, the
in-batch similarity matrix is

```text
S = Q P^T,       S_ij = cosine(q_i, p_j).
```

The decoupled InfoNCE row loss is

```text
L_i = -S_ii / tau + log sum_{j != i} exp(S_ij / tau).
```

`atom_gate` learns a query-only scalar `alpha_i = alpha(q_i)` and adds the
SimCSE dropout-view pressure per query:

```text
L_i_gate = L_i + (1 - alpha_i) lambda_sc L_i_sc.
```

When lexical metadata is available, the same gate also controls a pairwise
semantic/lexical auxiliary term. The gate is training-only; the saved encoder
has the same inference interface as the source embedding model.

## Adaptive Bank

`atomic` keeps a FIFO queue `B` of detached document embeddings. The bank does
not retain previous autograd graphs. For each current query:

```text
g(q_i) = max_j cosine(q_i, b_j) - cosine(q_i, p_i)
w_h(q_i) = sigmoid((c - g(q_i)) / beta_c)
```

Large positive `g(q)` indicates that a bank candidate is closer than the known
positive and may be a false negative. Its hard-negative contribution is then
reduced by `w_h(q)`.

The query margin head predicts

```text
m_raw(q) = m_0 + s tanh(h_m(q))
m(q) = clip(m_raw(q), m_min, m_max).
```

Each selected bank logit receives differentiable soft admission:

```text
a_ij = sigmoid((S_i+ - m(q_i) - S_ij_bank) / beta_m)
z_ij_bank = S_ij_bank / tau + log w_h(q_i) + log a_ij.
```

The bank columns are concatenated with in-batch negative columns inside the
same log-sum-exp denominator. Gradients flow through current query embeddings,
temperature, `alpha(q)`, and `m(q)`, but never through stored bank embeddings.

## Canonical Profiles

Selecting a method applies its registered defaults before YAML and CLI
overrides. Canonical `atomic` currently uses a 512-entry mixed bank, 4 hard and
12 random candidates, adaptive collision weighting, and a query margin centered
at `0.05`. Structural changes must be labeled as ablations:

```bash
python -m justatom.api.train \
  --config configs/train.yaml \
  --method atomic \
  --experiment.role ablation \
  --memory-bank.margin.mode constant
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
The objective, query gate, and memory bank do not change; the optimizer simply
updates PEFT adapter parameters instead of the frozen backbone parameters.

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
  lr_encoder: 0.0002

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

## Artifacts

Every successful training run writes:

- `encoder/`: deployable Hugging Face-compatible encoder artifact
- `adapter/`: PEFT adapter and its config when LoRA is enabled
- `research/checkpoint.pt`: complete training state for analysis or continuation
- `run_manifest.yaml`: resolved method, data, seed, hyperparameters, and Git state
- `batch_metrics.csv`: losses, retrieval ranks, `alpha(q)`, bank geometry, collision `g(q)`, hard weights, and margin distributions

The adapter and research checkpoint are saved before LoRA is merged into the
deployable encoder. Loading `encoder/` therefore does not require PEFT, while
`adapter/` retains the small reusable delta for reproducibility.

The benchmark additionally writes commands, retrieval tables, geometry tables,
and process RSS snapshots. These files are the reproducibility boundary for
reported experiments.
