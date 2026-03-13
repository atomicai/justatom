# Getting Started

## Installation

Install the library in editable mode for local development:

```bash
pip install -e .
```

Install the documentation toolchain when you want the local docs site:

```bash
pip install -r requirements-docs.txt
```

## Common Local Commands

```bash
make format-check
make fix-format
make docs-serve
make docs-build
pytest tests -m "not integration"
```

## Project Highlights

### Dataset adapters

`DatasetRecordAdapter` converts external datasets into a canonical document shape while preserving non-standard fields inside `meta`.

```python
from pathlib import Path
from justatom.tooling.dataset import DatasetRecordAdapter

adapter = DatasetRecordAdapter.from_source(
    Path("data/eval.parquet"),
    lazy=True,
    content_col="output",
    queries_col="input",
)

first_doc = next(adapter.iterator())
print(first_doc["content"])
print(first_doc["meta"]["labels"])
```

Supported source formats:

- `json`
- `jsonl`
- `parquet`
- `csv`
- `xlsx`

### Scenario-driven workflows

The repository treats scenario configs as the main interface for evaluation and training.

Main files:

- `configs/evaluate.yaml`
- `configs/train.yaml`
- `configs/dataset/<id>.yaml`

A scenario can reference a dataset preset by `dataset.id`, or point directly to a file or remote source via `dataset.name_or_path`.

### Test layout

- standard tests: `pytest tests -m "not integration"`
- integration tests: `pytest tests -m integration`

Integration tests expect external services such as Weaviate to be available.

## Documentation Workflow

Serve the documentation site locally:

```bash
make docs-serve
```

Build the static site exactly like CI:

```bash
make docs-build
```

The generated site is written to `site/`.
