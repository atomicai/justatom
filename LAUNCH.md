# Launch Guide

This document explains how to run `justatom` training and evaluation with scenario configs, dataset preset IDs, and direct dataset paths.

## Core Idea

`justatom` uses scenario configs as the main entrypoint:

- `configs/evaluate.yaml`
- `configs/train.yaml`

Inside those files, the `dataset` section can be configured in two ways:

1. Set `dataset.id` and let `justatom` resolve a preset.
2. Set `dataset.name_or_path` directly.

## How `dataset.id` Works

When you set:

```yaml
dataset:
  id: justatom
```

`justatom` looks for a preset in this order:

1. `configs/dataset/<id>.yaml`
2. `justatom/builtins/configs/dataset/<id>.yaml`

Then it merges that preset into the current scenario config.

This means `dataset.id` is just a short alias for a fuller dataset description.

## Supported Dataset Sources

`dataset.name_or_path` can point to several source types.

### 1. Repo-local named dataset

```yaml
name_or_path: justatom
```

This resolves to:

- `.data/polaroids.ai.data.json`

Use it when you want the repository's own built-in working dataset.

### 2. Packaged built-in dataset

```yaml
name_or_path: builtin://datasets/demo_retrieval.jsonl
```

Use it for tiny smoke tests and examples.

### 3. Hugging Face dataset

```yaml
name_or_path: hf://MLNavigator/russian-retrieval
split: train
```

Use it for datasets loaded via the `datasets` package.

If split names differ between datasets, you can provide fallback candidates:

```yaml
split: dev|test
```

For HF datasets this means: try `dev`, and if that split does not exist, try `test`.

### 4. Regular file path

Examples:

```yaml
name_or_path: data/eval.jsonl
name_or_path: /absolute/path/to/train.parquet
```

Supported file formats currently include:

- `.json`
- `.jsonl`
- `.parquet`
- `.csv`
- `.xlsx`

## Current Preset IDs

### `justatom`

Source:

- `configs/dataset/justatom.yaml`

Meaning:

- uses `.data/polaroids.ai.data.json`
- `content_field: content`
- `labels_field: queries`
- `chunk_id_col: chunk_id`
- keyword metadata comes from `keywords_or_phrases`

Best for:

- repo-local experiments
- quick training runs
- evaluation on the built-in dataset

### `demo-eval`

Source:

- `justatom/builtins/configs/dataset/demo-eval.yaml`

Meaning:

- uses packaged demo JSONL
- `labels_field: labels`

Best for:

- evaluation smoke tests
- examples in docs/tests

### `demo-train`

Source:

- `justatom/builtins/configs/dataset/demo-train.yaml`

Meaning:

- uses packaged demo JSONL
- `labels_field: queries`

Best for:

- training smoke tests
- gamma/training pipeline checks

### `mlnavigator-russian-retrieval`

Source:

- `justatom/builtins/configs/dataset/mlnavigator-russian-retrieval.yaml`

Meaning:

- uses `hf://MLNavigator/russian-retrieval`
- `split: train`
- `content_field: text`
- `labels_field: q`

Dataset schema observed from HF:

- `text`
- `q`
- `a`
- `context`

Best for:

- Russian retrieval experiments
- realistic HF-backed evaluation/training inputs

## Quick Start: Evaluation

### Option A. Evaluate with the repo-local `justatom` dataset

```bash
python -m justatom.api.eval --config configs/evaluate.yaml --dataset.id justatom
```

### Option B. Evaluate with the tiny packaged demo dataset

```bash
python -m justatom.api.eval --config configs/evaluate.yaml --dataset.id demo-eval
```

### Option C. Evaluate with MLNavigator on Hugging Face

```bash
python -m justatom.api.eval --config configs/evaluate.yaml --dataset.id mlnavigator-russian-retrieval
```

### Option D. Evaluate with direct overrides instead of preset IDs

```bash
python -m justatom.api.eval \
  --config configs/evaluate.yaml \
  --dataset.name_or_path hf://MLNavigator/russian-retrieval \
  --dataset.split train \
  --dataset.content_field text \
  --dataset.labels_field q
```

### Useful evaluation overrides

```bash
python -m justatom.api.eval \
  --config configs/evaluate.yaml \
  --dataset.id justatom \
  --search.pipeline keywords \
  --search.top_k 10 \
  --index.flush_collection true
```

Notes:

- Evaluation usually needs your retrieval backend available.
- For local smoke runs, the `keywords` pipeline is often the easiest starting point.
- `configs/evaluate.yaml` is the base config; CLI dotted flags override it.

## Evaluation Metrics

`justatom.api.eval` computes retrieval metrics over the labels extracted from `dataset.labels_field`.

Important behavior:

- If `dataset.labels_field` is not set, `eval` only indexes data and skips metric calculation.
- If `dataset.labels_field` is set but no labels are found in the dataset, metric calculation is skipped.
- Results are written as a CSV into `output.save_results_to_dir`.
- By default, results go to `evals/`.

### Supported metrics

The evaluator currently supports these metric names:

- `HitRate`
- `mrr`
- `map`
- `ndcg`

The metric name normalization is forgiving, so these variants are treated the same:

- `HitRate`, `hitrate`, `hr`, `HitRate@`
- `mrr`, `MRR`, `mrr@`
- `map`, `MAP`, `map@`
- `ndcg`, `NDCG`, `ndcg@`

### How metric config works

In the scenario config, the relevant block is:

```yaml
metrics:
  top_k:
    - HitRate
    - mrr
    - map
    - ndcg
  eval_top_k:
    - 1
    - 5
    - 10
```

Meaning:

- `metrics.top_k`: which metric families to compute
- `metrics.eval_top_k`: at which cutoffs to compute them, for example `@1`, `@5`, `@10`

So the example above produces metrics like:

- `HitRate@1`
- `HitRate@5`
- `mrr@10`
- `map@5`
- `ndcg@10`

Defaults:

- `metrics.top_k` defaults to `['HitRate']`
- `metrics.eval_top_k` defaults internally to `[1, 2, 5, 10, 12, 15, 20]` when not set

`eval` automatically retrieves enough documents to cover the largest requested cutoff.

### Metric examples

Run evaluation with only HitRate at default cutoffs:

```bash
python -m justatom.api.eval \
  --config configs/evaluate.yaml \
  --dataset.id justatom
```

Run evaluation with several metrics and custom cutoffs:

```bash
python -m justatom.api.eval \
  --config configs/evaluate.yaml \
  --dataset.id justatom \
  --metrics-top-k HitRate mrr map ndcg \
  --eval-top-k 1 5 10
```

The same configuration through scenario overrides:

```bash
python -m justatom.api.eval \
  --config configs/evaluate.yaml \
  --dataset.id mlnavigator-russian-retrieval \
  --metrics.top_k HitRate mrr ndcg \
  --metrics.eval_top_k 1 5 20
```

### Output format

The produced CSV contains one row per metric, with columns:

- `name`
- `mean`
- `std`
- `dataset`

Typical output rows look like:

```text
HitRate@1,0.33,0.57,hf://MLNavigator/russian-retrieval?split=train
mrr@5,0.61,0.34,hf://MLNavigator/russian-retrieval?split=train
ndcg@10,0.71,0.25,hf://MLNavigator/russian-retrieval?split=train
```

The output filename is assembled from the search pipeline, metric snapshot, model name, and optional runtime properties.

## Quick Start: Training

### Option A. Train with the repo-local `justatom` dataset

```bash
python -m justatom.api.train --config configs/train.yaml --dataset.id justatom
```

### Option B. Train with the packaged demo dataset

```bash
python -m justatom.api.train --config configs/train.yaml --dataset.id demo-train
```

### Option C. Train with MLNavigator on Hugging Face

```bash
python -m justatom.api.train --config configs/train.yaml --dataset.id mlnavigator-russian-retrieval
```

### Option D. Train with direct dataset path

```bash
python -m justatom.api.train \
  --config configs/train.yaml \
  --dataset.name_or_path justatom \
  --dataset.content_field content \
  --dataset.labels_field queries
```

### Useful training overrides

```bash
python -m justatom.api.train \
  --config configs/train.yaml \
  --dataset.id justatom \
  --training.batch_size 16 \
  --training.n_epochs 3 \
  --training.freeze_encoder false
```

Notes:

- Training does not require Weaviate.
- `configs/train.yaml` is the base config; CLI dotted flags override it.

## Direct Python Usage

### Repo-local named dataset

```python
from justatom.tooling.dataset import DatasetRecordAdapter

adapter = DatasetRecordAdapter.from_source(
    "justatom",
    content_col="content",
    queries_col="queries",
    chunk_id_col="chunk_id",
    keywords_col="keywords_or_phrases",
    keywords_nested_col="keyword_or_phrase",
    explanation_nested_col="explanation",
    lazy=True,
)

first = next(adapter.iterator())
print(first["id"])
print(first["content"])
print(first["meta"]["labels"])
```

### Hugging Face dataset

```python
from justatom.tooling.dataset import DatasetRecordAdapter

adapter = DatasetRecordAdapter.from_source(
  "hf://MLNavigator/russian-retrieval",
    content_col="text",
    queries_col="q",
  split="train",
    lazy=True,
)

first = next(adapter.iterator())
print(first["content"])
print(first["meta"]["labels"])
```

## What the Fields Mean

Common dataset fields in `justatom` configs:

- `name_or_path`: where to load the dataset from
- `labels_field`: the field used as the retrieval query or supervision label list
- `content_field`: the field mapped to document content
- `split`: dataset split to load; for HF datasets can be `train`, `test`, `dev`, or fallback chains like `dev|test`
- `limit`: optional cap on the number of rows to read from the dataset
- `chunk_id_col`: unique document/chunk identifier
- `keywords_col`: optional keyword metadata field
- `keywords_nested_col`: nested key inside keyword objects
- `explanation_nested_col`: nested explanation key inside keyword objects

## Recommended Starting Points

If you want the simplest path:

1. For training: start with `dataset.id=justatom` or `dataset.id=demo-train`.
2. For evaluation: start with `dataset.id=demo-eval` or `dataset.id=justatom`.
3. For a real Russian HF retrieval dataset: use `dataset.id=mlnavigator-russian-retrieval`.

## Troubleshooting

### `dataset.id` does not resolve

Check that one of these files exists:

- `configs/dataset/<id>.yaml`
- `justatom/builtins/configs/dataset/<id>.yaml`

### `justatom` source cannot be found

Make sure you run from the repository root so this file exists:

- `.data/polaroids.ai.data.json`

### HF dataset fails to load

Check:

- internet access
- `datasets` package installed
- correct URI format, for example:
  - `hf://MLNavigator/russian-retrieval?split=train`

### Need to override only one field from a preset

Use a preset plus dotted CLI override:

```bash
python -m justatom.api.eval \
  --config configs/evaluate.yaml \
  --dataset.id justatom \
  --dataset.labels_field queries
```
