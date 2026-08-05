# Launch Guide

This page is the documentation version of the repository launch notes. It explains how `justatom` resolves datasets and how to run evaluation or training from scenario configs.

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

`justatom` resolves a preset in this order:

1. `configs/dataset/<id>.yaml`
2. `justatom/builtins/configs/dataset/<id>.yaml`

This keeps scenario files short while still allowing richer dataset definitions.

## Supported Dataset Sources

### Repo-local dataset preset

```yaml
id: justatom
```

Resolves to `.data/polaroids.ai.data.json` with `lazy: false`.

### Packaged built-in dataset

```yaml
name_or_path: demo
lazy: true
```

Useful for smoke tests and tiny examples.

### Hugging Face dataset

```yaml
name_or_path: MLNavigator/russian-retrieval
config: null
lazy: true
split: train
```

For HF datasets you can also provide fallback split candidates:

```yaml
split: dev|test
```

### File path

```yaml
name_or_path: data/eval.jsonl
name_or_path: /absolute/path/to/train.parquet
```

Supported formats include `.json`, `.jsonl`, `.parquet`, `.csv`, and `.xlsx`.
Lazy mode is a real row iterator for `.jsonl`, `.parquet`, and `.csv`. Use
`lazy: false` for `.json` and `.xlsx`; these formats raise a clear error when
lazy mode is requested.

## Current Preset IDs

### `justatom`

- Source: `configs/dataset/justatom.yaml`
- Best for: repo-local experiments and quick evaluation runs

### `demo-eval`

- Source: `justatom/builtins/configs/dataset/demo-eval.yaml`
- Best for: evaluation smoke tests

### `demo-train`

- Source: `justatom/builtins/configs/dataset/demo-train.yaml`
- Best for: training smoke tests

### `boolq-ru`

- Source: `configs/dataset/boolq-ru.yaml`
- Best for: compact Russian QA retrieval experiments backed by Hugging Face

## Quick Start: Evaluation

### Evaluate with the repo-local preset

```bash
python -m justatom.api.eval --config configs/evaluate.yaml --dataset.id justatom
```

### Evaluate with the packaged demo preset

```bash
python -m justatom.api.eval --config configs/evaluate.yaml --dataset.id demo-eval
```

### Evaluate with MLNavigator on Hugging Face

```bash
python -m justatom.api.eval --config configs/evaluate.yaml --dataset.id boolq-ru
```

### Evaluate with direct overrides

```bash
python -m justatom.api.eval \
  --config configs/evaluate.yaml \
  --dataset.name_or_path MLNavigator/russian-retrieval \
  --dataset.split train \
  --dataset.content_field text \
  --dataset.labels_field q
```

## Useful Overrides

```bash
python -m justatom.api.eval \
  --config configs/evaluate.yaml \
  --dataset.id justatom \
  --search.pipeline keywords \
  --search.top_k 10 \
  --index.flush_collection true
```

## Notes

- Evaluation usually requires a retrieval backend to be available.
- For local smoke runs, the `keywords` pipeline is often the fastest starting point.
- CLI dotted flags override the base scenario config.
