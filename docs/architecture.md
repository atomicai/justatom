# Architecture

## Repository Shape

`justatom` is organized around a few clear layers that map well to real retrieval systems.

![Architecture overview](atom-arch.png)

## Main Packages

### `justatom.api`

Command-line entrypoints for:

- evaluation
- training
- runtime and service helpers

### `justatom.running`

Execution layer for:

- retrievers
- evaluators
- services
- embeddings

### `justatom.training`

The production training path is compositional and has one owner per concern:

- `config.py` and `methods.py`: strict configuration and canonical profiles
- `data.py` and `sampling.py`: bounded loading and safe negative selection
- `alpha_gate.py`: query-only `alpha(q)`
- `memory_bank.py`: detached FIFO candidates and adaptive mining
- `objective.py`: InfoNCE, auxiliary terms, soft admission, and `m(q)` regularization
- `module.py`: the only Lightning training module
- `job.py`: orchestration, manifests, and model artifacts

### `justatom.processing`

Data preparation layer for:

- loading
- tokenization
- sampling
- data `silo` and `batch`es

### `justatom.storing`

Persistence and external data interfaces:

- dataset storage
- vector-related integrations
- Weaviate helpers

### `justatom.modeling`

Model implementations and shared numeric or metric helpers.

### `configs`

Scenario configuration files for training, evaluation, and dataset presets.

## Why This Layout Works

- API entrypoints stay thin.
- Training composition lives separately from retrieval runtime code.
- Data loading and persistence are isolated from experimentation logic.
- Config-driven runs are easy to reproduce inside CI and notebooks.

## Assets Already In The Repo

The `docs/` directory already stores project visuals such as:

- `Logo.png`
- `atom-arch.png`
- charts comparing retrieval behavior
- math-related diagrams in `docs/math/`

MkDocs now turns those assets into a browsable documentation site instead of leaving them as disconnected files.
