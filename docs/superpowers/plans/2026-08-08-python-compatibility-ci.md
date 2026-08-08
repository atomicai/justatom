# Python 3.10-3.13 Compatibility And CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PR #40 pass required CI while honestly supporting Python 3.10 through 3.13 inclusive.

**Architecture:** Keep runtime capabilities in optional extras, make clustering-only imports lazy, and test the declared compatibility contract directly. Treat Docker Compose output as a semantic contract instead of an exact serialization contract.

**Tech Stack:** Python 3.10-3.13, pytest, setuptools extras, Quart, Hypercorn, Docker Compose, GitHub Actions.

## Global Constraints

- Package support is exactly `>=3.10,<3.14`.
- Production Docker images remain on Python 3.12.
- Ubuntu covers all supported Python versions; Windows and macOS use Python 3.12.
- BERTopic and UMAP remain optional and absent from API/embedder images.
- Every observed CI failure receives a focused regression test before its implementation changes.

---

### Task 1: Lock The Python And Extra Contracts

**Files:**
- Modify: `tests/test_runtime_extras.py`
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yaml`

**Interfaces:**
- Consumes: PEP 621 `project.requires-python` and `project.optional-dependencies`.
- Produces: testable support metadata and a CI matrix that installs `torch`, `serve`, and `test`.

- [ ] Add tests asserting `>=3.10,<3.14`, conditional `tomli` coverage, and the presence of Quart/Hypercorn only in runtime extras.
- [ ] Run `python -m pytest tests/test_runtime_extras.py -q` and observe the metadata/fallback failures.
- [ ] Add `tomli>=2; python_version < '3.11'`, update package metadata, and load TOML with a 3.10-compatible import fallback.
- [ ] Change Ubuntu CI to 3.10-3.13, Windows/macOS to 3.12, and install `.[torch,serve,test]` in test jobs.
- [ ] Re-run the focused tests and commit the completed compatibility contract.

### Task 2: Make Clustering Dependencies Truly Optional

**Files:**
- Modify: `tests/test_clustering_embedding_adapter.py`
- Modify: `justatom/running/clusters.py`

**Interfaces:**
- Consumes: `EmbeddingBackendAdapter(Embedder)` and optional BERTopic/UMAP packages.
- Produces: module import and adapter operation without clustering packages; actionable construction errors for clustering features.

- [ ] Add subprocess tests that block BERTopic/UMAP imports, import `justatom.running.clusters`, exercise `EmbeddingBackendAdapter`, and assert constructor errors for unavailable clustering features.
- [ ] Run the focused subprocess tests and observe import-time failure.
- [ ] Replace eager BERTopic/UMAP imports with small lazy loader functions and a dependency-free adapter base when BERTopic is unavailable.
- [ ] Ensure constructors load their dependency before use and preserve real BERTopic `BaseEmbedder` inheritance when BERTopic is installed.
- [ ] Run clustering and retrieval adapter tests and commit the optional dependency boundary.

### Task 3: Normalize Docker Compose Mount Assertions

**Files:**
- Modify: `tests/test_docker_assets.py`

**Interfaces:**
- Consumes: rendered Compose JSON from different Docker Compose releases.
- Produces: `_assert_bind_mount(mount, source, target, read_only)` semantic validation.

- [ ] Add a focused test showing that `bind.create_host_path: true` is accepted while changed mount source/target/type/read-only fields are rejected.
- [ ] Run the focused test and observe failure under the current exact-dictionary comparison.
- [ ] Introduce the semantic mount assertion and use it in the rendered topology contract.
- [ ] Run all Docker asset tests available locally and commit the portability fix.

### Task 4: Verify Clean Environments And Publish

**Files:**
- Modify only files required by failures discovered during verification.

**Interfaces:**
- Consumes: final package metadata, tests, Compose launcher, and CI workflow.
- Produces: a pushed PR revision with green required checks.

- [ ] Resolve `.[torch,serve,test]` for Python 3.10, 3.11, 3.12, and 3.13 with `uv pip compile`.
- [ ] Run the complete conda test suite, format check, strict MkDocs build, Compose renders, Bash syntax checks, and `git diff --check`.
- [ ] Commit any verification-only correction with its regression test.
- [ ] Push `feature/retrieval-runtime` and monitor PR #40.
- [ ] For any new GitHub Actions failure, inspect the exact job log, add a regression test, fix it, and repeat until every required check is green.
