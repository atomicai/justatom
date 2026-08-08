# Python 3.10-3.13 Compatibility And CI Design

## Goal

Make the retrieval runtime PR installable and testable on Python 3.10 through
3.13 inclusive while keeping Python 3.12 as the production container runtime.

## Support Contract

- Package metadata declares `requires-python = ">=3.10,<3.14"`.
- Ubuntu CI tests Python 3.10, 3.11, 3.12, and 3.13.
- Windows and macOS CI test Python 3.12 as representative platform jobs.
- API, CPU embedder, and CUDA embedder images remain based on Python 3.12.
- Black and isort continue targeting Python 3.10 syntax because it is the
  oldest supported interpreter.

## Dependency Boundaries

The regular test matrix installs the `torch`, `serve`, and `test` extras. API
tests may therefore import Quart and Hypercorn without making those packages
base dependencies.

Python 3.10 test tooling uses `tomli` as a conditional backport and Python
3.11 or newer uses the standard-library `tomllib` module.

BERTopic and UMAP remain optional clustering dependencies. Importing the
dependency-light `EmbeddingBackendAdapter` must not require either package.
Code that constructs `IBTRunner`, `IHFWrapperBackend`, or `IUMAPDimReducer`
must fail at construction time with an actionable dependency error when the
required clustering package is absent.

## Portable Docker Contracts

Compose contract tests assert semantic mount fields and allow
Compose-version-generated metadata such as `bind.create_host_path`. They must
continue rejecting changes to source, target, read-only status, volume type,
or retrieval topology.

## Verification

- Focused tests prove Python metadata, conditional TOML loading, lazy optional
  clustering imports, and Compose mount normalization.
- The complete local test suite and formatting/documentation checks pass.
- Clean dependency resolution succeeds for Python 3.10, 3.11, 3.12, and 3.13.
- GitHub Actions is green for all required jobs before the PR is merged.

## Out Of Scope

- Running the full operating-system/Python Cartesian product.
- Changing the pinned PyTorch or model versions.
- Installing BERTopic in the default API or embedding containers.
- Claiming support for Python 3.14.
