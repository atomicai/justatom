# Habr IR Pilot Release and Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize the completed 100-request Habr v4 pilot into checksummed Parquet pair/corpus/qrel artifacts and a human-review CSV without starting another API batch.

**Architecture:** Add a focused `release.py` module that owns immutable generation bindings, pure release-frame construction, invariant validation, atomic release-directory writing, and pilot audit export. The existing CLI receives a local-only `finalize` stage that validates all source/generation checksums before calling the module. JSONL remains the audit source; Parquet becomes the canonical training and Hugging Face representation.

**Tech Stack:** Python 3.12, Polars 1.38.1, PyYAML, Hugging Face Datasets 2.18.0, pytest.

## Global Constraints

- `evidence` is an exact continuous quote and never replaces `positive_passage`.
- `positive_passage` exactly equals the corpus row's `serialized_passage`.
- Release IDs use full domain-separated SHA-256 digests and never depend on row order or generated wording.
- Accepted generations produce exactly one pair and one qrel; rejected generations produce neither.
- No article may appear in more than one pair split.
- Existing source, target, context, collected-output, and request-shard checksums are mandatory trust boundaries.
- JSONL audit artifacts are not uploaded by default.
- `generation.scale_authorized` remains `false`; this plan performs no OpenAI submission and no Hugging Face upload.

---

### Task 1: Pure Release Materialization

**Files:**
- Create: `justatom/tooling/ir_dataset/release.py`
- Create: `tests/test_ir_release.py`
- Modify: `justatom/tooling/ir_dataset/__init__.py`

**Interfaces:**
- Consumes: collected terminal records, target rows, corpus rows, and exact request/shard bindings.
- Produces: `GenerationBinding`, `ReleaseFrames`, `stable_query_id`, `stable_pair_id`, and `materialize_release_frames`.

- [ ] **Step 1: Write failing tests for stable IDs and accepted-only materialization**

Create `tests/test_ir_release.py` with fixtures that contain two targets from separate articles, one accepted generation, and one rejected generation:

```python
from __future__ import annotations

import polars as pl

from justatom.tooling.ir_dataset.release import (
    GenerationBinding,
    materialize_release_frames,
    stable_pair_id,
    stable_query_id,
)


def corpus() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "corpus_rank": 0,
                "passage_id": "p1",
                "article_id": "a1",
                "title": "Redis",
                "section": "Retry",
                "content": "Redis повторяет запрос после сетевой ошибки.",
                "serialized_passage": "passage: Redis\nRetry\n\nRedis повторяет запрос после сетевой ошибки.",
                "url": "https://example.test/redis",
                "flows": ["develop"],
                "hubs": ["redis"],
                "tags": ["redis", "retry"],
                "char_count": 43,
                "token_count": 8,
                "source_hash": "source-1",
            },
            {
                "corpus_rank": 1,
                "passage_id": "p2",
                "article_id": "a2",
                "title": "PostgreSQL",
                "section": "WAL",
                "content": "PostgreSQL хранит журнал WAL.",
                "serialized_passage": "passage: PostgreSQL\nWAL\n\nPostgreSQL хранит журнал WAL.",
                "url": "https://example.test/postgresql",
                "flows": ["develop"],
                "hubs": ["postgresql"],
                "tags": ["wal"],
                "char_count": 31,
                "token_count": 6,
                "source_hash": "source-2",
            },
        ]
    )


def targets() -> pl.DataFrame:
    return corpus().with_columns(
        pl.Series("split", ["train", "test"]),
        pl.Series("requested_intent", ["how_to", "concept"]),
    )


def records() -> list[dict[str, object]]:
    return [
        {
            "custom_id": "gen-accepted",
            "status": "accepted",
            "reason_codes": [],
            "output": {
                "query": "Как Redis обрабатывает запрос после сетевой ошибки?",
                "answer": "Redis повторяет запрос.",
                "evidence": "Redis повторяет запрос после сетевой ошибки.",
                "requested_intent": "how_to",
                "actual_intent": "how_to",
            },
        },
        {
            "custom_id": "gen-rejected",
            "status": "rejected",
            "reason_codes": ["usable_false"],
            "output": {
                "query": "",
                "answer": "",
                "evidence": "",
                "requested_intent": "concept",
                "actual_intent": "concept",
            },
        },
    ]


def bindings() -> dict[str, GenerationBinding]:
    return {
        "gen-accepted": GenerationBinding(
            custom_id="gen-accepted",
            passage_id="p1",
            article_id="a1",
            source_hash="source-1",
            prompt_hash="prompt-1",
            generation_attempt=1,
            batch_id="batch-1",
        ),
        "gen-rejected": GenerationBinding(
            custom_id="gen-rejected",
            passage_id="p2",
            article_id="a2",
            source_hash="source-2",
            prompt_hash="prompt-2",
            generation_attempt=1,
            batch_id="batch-1",
        ),
    }


def test_release_ids_are_domain_separated_and_stable():
    query_id = stable_query_id("gen-accepted")
    pair_id = stable_pair_id(query_id, "p1")

    assert query_id.startswith("q-") and len(query_id) == 66
    assert pair_id.startswith("pair-") and len(pair_id) == 69
    assert query_id == stable_query_id("gen-accepted")
    assert pair_id == stable_pair_id(query_id, "p1")


def test_materialization_keeps_only_accepted_rows_and_full_positive():
    result = materialize_release_frames(
        records=records(),
        targets=targets(),
        corpus=corpus(),
        bindings=bindings(),
        generator_model="test-model",
    )

    assert result.pairs.height == 1
    assert result.qrels.height == 1
    assert result.corpus.height == 2
    pair = result.pairs.row(0, named=True)
    assert pair["positive_passage_id"] == "p1"
    assert pair["positive_passage"] == corpus().row(0, named=True)["serialized_passage"]
    assert pair["evidence"] in pair["positive_passage"]
    assert result.qrels.row(0, named=True) == {
        "query_id": pair["query_id"],
        "passage_id": "p1",
        "relevance": 1,
    }
    assert result.corpus.filter(pl.col("passage_id") == "p1")["is_positive"].item() is True
    assert result.corpus.filter(pl.col("passage_id") == "p2")["is_positive"].item() is False
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
conda run -n justatom python -m pytest tests/test_ir_release.py -q
```

Expected: collection fails with `ModuleNotFoundError: justatom.tooling.ir_dataset.release`.

- [ ] **Step 3: Implement stable IDs, bindings, release frames, and hard invariants**

Create `justatom/tooling/ir_dataset/release.py` with these public interfaces:

```python
@dataclass(frozen=True, slots=True)
class GenerationBinding:
    custom_id: str
    passage_id: str
    article_id: str
    source_hash: str
    prompt_hash: str
    generation_attempt: int
    batch_id: str


@dataclass(frozen=True, slots=True)
class ReleaseFrames:
    pairs: pl.DataFrame
    corpus: pl.DataFrame
    qrels: pl.DataFrame


def stable_query_id(custom_id: str) -> str:
    digest = hashlib.sha256(f"habr-ir-query-v1\0{custom_id}".encode("utf-8")).hexdigest()
    return f"q-{digest}"


def stable_pair_id(query_id: str, positive_passage_id: str) -> str:
    payload = f"habr-ir-pair-v1\0{query_id}\0{positive_passage_id}"
    return f"pair-{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def materialize_release_frames(
    *,
    records: Sequence[Mapping[str, Any]],
    targets: pl.DataFrame,
    corpus: pl.DataFrame,
    bindings: Mapping[str, GenerationBinding],
    generator_model: str,
) -> ReleaseFrames:
    """Validate immutable bindings and build deterministic release frames."""
```

The implementation must validate all records before building frames: unique
`custom_id`; exact record/binding key equality; unique target and corpus
`passage_id`; binding article/source identity matching both target and corpus;
accepted outputs containing all required strings; exact raw evidence substring;
accepted normalized-query uniqueness; target split in `train/dev/test`; and
article-disjoint accepted splits. It maps target split `dev` to release split
`validation`, preserves `train` and `test`, sorts pairs and qrels by `(split,
query_id)`, and sorts corpus by `corpus_rank`.

Export the five public symbols from `justatom/tooling/ir_dataset/__init__.py`.

- [ ] **Step 4: Run focused and existing generation tests**

```bash
conda run -n justatom python -m pytest tests/test_ir_release.py tests/test_ir_batch.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add justatom/tooling/ir_dataset/release.py justatom/tooling/ir_dataset/__init__.py tests/test_ir_release.py
git commit -m "feat: materialize Habr IR release frames"
```

---

### Task 2: Checksummed Atomic Release and Audit Sheet

**Files:**
- Modify: `justatom/tooling/ir_dataset/release.py`
- Modify: `tests/test_ir_release.py`

**Interfaces:**
- Consumes: `passages.parquet`, source manifest, generation state, targets, context, request shards, and collected JSONL.
- Produces: `ReleaseSummary`, `finalize_release`, `pilot-review.csv`, release Parquet files, Dataset Card, and `release-manifest.json`.

- [ ] **Step 1: Add failing tests for trust-bound input loading**

Add fixture helper `write_release_workspace(tmp_path)` that writes a two-row
corpus, source manifest, bound targets/context artifacts, request JSONL,
generation state, and collected JSONL with matching SHA-256 values. Then add:

```python
def test_finalize_rejects_tampered_collected_output(tmp_path):
    source_root, generation_root, release_root = write_release_workspace(tmp_path)
    with (generation_root / "generation_collected.jsonl").open("ab") as stream:
        stream.write(b"{}\n")

    with pytest.raises(ValueError, match="collected artifact checksum mismatch"):
        finalize_release(source_root, generation_root, release_root, git_sha="test-sha", git_dirty=False)


def test_finalize_rejects_missing_or_ambiguous_request_binding(tmp_path):
    source_root, generation_root, release_root = write_release_workspace(tmp_path)
    request_path = next((generation_root / "generation_requests").glob("*.jsonl"))
    request_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="request shard checksum mismatch"):
        finalize_release(source_root, generation_root, release_root, git_sha="test-sha", git_dirty=False)
```

- [ ] **Step 2: Verify the trust-boundary tests fail**

```bash
conda run -n justatom python -m pytest tests/test_ir_release.py -k "tampered or ambiguous" -q
```

Expected: FAIL because `finalize_release` does not exist.

- [ ] **Step 3: Add failing tests for file layout, manifest, idempotence, and audit columns**

```python
def test_finalize_writes_hf_layout_manifest_and_review_sheet(tmp_path):
    source_root, generation_root, release_root = write_release_workspace(tmp_path)

    result = finalize_release(source_root, generation_root, release_root, git_sha="test-sha", git_dirty=False)

    assert result.pair_count == 1
    assert result.review_count == 2
    assert (release_root / "data/pairs/train.parquet").exists()
    assert (release_root / "data/pairs/validation.parquet").exists()
    assert (release_root / "data/pairs/test.parquet").exists()
    assert (release_root / "data/corpus-100k/corpus.parquet").exists()
    assert (release_root / "data/qrels/train.parquet").exists()
    assert (release_root / "audit/pilot-review.csv").exists()
    assert (release_root / "README.md").exists()
    manifest = json.loads((release_root / "data/manifests/release-manifest.json").read_text())
    assert manifest["git"] == {"sha": "test-sha", "dirty": False}
    assert manifest["counts"]["pairs"] == 1
    assert manifest["counts"]["audit_rows"] == 2
    assert all(item["sha256"] for item in manifest["artifacts"])
    review = pl.read_csv(release_root / "audit/pilot-review.csv")
    assert review.columns[-10:] == [
        "human_target_answers_query",
        "human_evidence_supports_answer",
        "human_self_contained",
        "human_single_interpretation",
        "human_no_competitor_answers",
        "human_natural_not_copied",
        "human_correct_intent",
        "human_accept",
        "reviewer",
        "notes",
    ]


def test_finalize_exactly_reuses_matching_release(tmp_path):
    source_root, generation_root, release_root = write_release_workspace(tmp_path)
    first = finalize_release(source_root, generation_root, release_root, git_sha="test-sha", git_dirty=False)
    second = finalize_release(source_root, generation_root, release_root, git_sha="test-sha", git_dirty=False)

    assert first.reused is False
    assert second.reused is True
    assert first.root == second.root
    assert first.manifest_path == second.manifest_path
    assert first.pair_count == second.pair_count
    assert first.corpus_count == second.corpus_count
    assert first.qrel_count == second.qrel_count
    assert first.review_count == second.review_count
    assert first.fingerprint == second.fingerprint
```

- [ ] **Step 4: Verify the layout tests fail**

```bash
conda run -n justatom python -m pytest tests/test_ir_release.py -k "writes_hf or exactly_reuses" -q
```

Expected: FAIL because no release writer exists.

- [ ] **Step 5: Implement validated input loading and atomic release writing**

Add:

```python
@dataclass(frozen=True, slots=True)
class ReleaseSummary:
    root: Path
    manifest_path: Path
    pair_count: int
    corpus_count: int
    qrel_count: int
    review_count: int
    fingerprint: str
    reused: bool


def finalize_release(
    source_root: str | Path,
    generation_root: str | Path,
    release_root: str | Path,
    *,
    git_sha: str,
    git_dirty: bool,
) -> ReleaseSummary:
    """Validate collected generation state and atomically write a local release."""
```

`finalize_release` must:

1. validate source manifest fingerprint, chunker version, and actual passages SHA;
2. validate `targets_state.json` and `generation_context_state.json` contracts;
3. validate generation-state source, target, context, collected, diagnostics,
   pilot-metrics, and request-shard checksums;
4. parse each request's `body.metadata` and bind it to exactly one state shard
   containing the same `custom_id` and a non-empty `batch_id`;
5. call `materialize_release_frames`;
6. build an all-request audit frame with automatic output, exact target,
   three competitor IDs/passages, and the ten blank human columns from the test;
7. write a complete temporary release directory using Zstandard Parquet,
   fsync files, compute checksums, write the manifest last, fsync, and atomically
   rename the directory;
8. exactly reuse an existing release only when its manifest fingerprint and
   every artifact checksum match; otherwise fail without overwriting.

The generated `README.md` contains explicit `pairs`, `corpus-100k`, and `qrels`
configurations matching the approved design. Empty pair/qrel splits are written
with the same schema as non-empty splits.

- [ ] **Step 6: Run release tests**

```bash
conda run -n justatom python -m pytest tests/test_ir_release.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add justatom/tooling/ir_dataset/release.py tests/test_ir_release.py
git commit -m "feat: write checksummed Habr IR release artifacts"
```

---

### Task 3: Local Finalize CLI

**Files:**
- Modify: `justatom/api/ir_dataset.py`
- Modify: `configs/datasets/habr-ir.yaml`
- Modify: `tests/test_ir_dataset_cli.py`

**Interfaces:**
- Consumes: `finalize_release` from Task 2 and `output.release_root`.
- Produces: `finalize_stage(config: IRDatasetConfig) -> ReleaseSummary` and the `finalize` CLI stage.

- [ ] **Step 1: Add failing config and CLI tests**

```python
def test_checked_in_config_has_separate_local_release_root():
    config = load_ir_dataset_config(CONFIG_PATH)
    assert config.output.release_root == Path(".tmp_runs/datasets/habr-ir/pilot-release-v1")


def test_cli_accepts_local_finalize_stage():
    parsed = parse_cli(["--config", str(CONFIG_PATH), "finalize"])
    assert parsed.stage == "finalize"


def test_finalize_stage_passes_bound_roots_and_git_identity(monkeypatch, tmp_path):
    config = load_ir_dataset_config(
        CONFIG_PATH,
        overrides={
            "output": {
                "root": str(tmp_path / "source"),
                "generation_root": str(tmp_path / "generation"),
                "release_root": str(tmp_path / "release"),
            }
        },
    )
    captured = {}
    monkeypatch.setattr(ir_dataset_module, "_git_identity", lambda: ("abc123", False))
    monkeypatch.setattr(
        ir_dataset_module,
        "finalize_release",
        lambda source_root, generation_root, release_root, **kwargs: captured.update(
            source_root=source_root,
            generation_root=generation_root,
            release_root=release_root,
            **kwargs,
        ),
    )

    finalize_stage(config)

    assert captured == {
        "source_root": tmp_path / "source",
        "generation_root": tmp_path / "generation",
        "release_root": tmp_path / "release",
        "git_sha": "abc123",
        "git_dirty": False,
    }
```

- [ ] **Step 2: Verify CLI tests fail**

```bash
conda run -n justatom python -m pytest tests/test_ir_dataset_cli.py -k "release_root or local_finalize or passes_bound_roots" -q
```

Expected: FAIL because `release_root`, `finalize`, and `finalize_stage` are absent.

- [ ] **Step 3: Implement config parsing and finalize dispatch**

Add `release_root: Path` to `OutputConfig`, convert it to `Path` in
`load_ir_dataset_config`, add `finalize` to the exact-one-stage tuple, and add:

```python
def _git_identity() -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return sha, dirty


def finalize_stage(config: IRDatasetConfig) -> ReleaseSummary:
    git_sha, git_dirty = _git_identity()
    return finalize_release(
        config.output.root,
        config.output.generation_root,
        config.output.release_root,
        git_sha=git_sha,
        git_dirty=git_dirty,
    )
```

Make `collect-generation` an explicit dispatch branch and reserve the final
`else` for `finalize`, preventing unknown future stages from silently collecting.

Set in `configs/datasets/habr-ir.yaml`:

```yaml
output:
  release_root: .tmp_runs/datasets/habr-ir/pilot-release-v1
```

- [ ] **Step 4: Run CLI, release, and generation tests**

```bash
conda run -n justatom python -m pytest tests/test_ir_dataset_cli.py tests/test_ir_release.py tests/test_ir_batch.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add justatom/api/ir_dataset.py configs/datasets/habr-ir.yaml tests/test_ir_dataset_cli.py
git commit -m "feat: add local Habr IR finalize stage"
```

---

### Task 4: Materialize and Inspect the Completed Pilot

**Files:**
- Modify: `docs/research/habr-ir-v4-generation-pilot.md`
- Generated, ignored: `.tmp_runs/datasets/habr-ir/pilot-release-v1/`

**Interfaces:**
- Consumes: completed v3 pilot and v4 corpus already present locally.
- Produces: 96 release pairs, 100,000 corpus rows, 96 qrels, and a 100-row human-audit CSV.

- [ ] **Step 1: Run the local-only finalizer**

```bash
conda run -n justatom python -m justatom.api.ir_dataset \
  --config configs/datasets/habr-ir.yaml \
  finalize
```

Expected summary: `pair_count=96`, `corpus_count=100000`,
`qrel_count=96`, and `review_count=100`.

- [ ] **Step 2: Verify split and evidence invariants against real artifacts**

```bash
conda run -n justatom python - <<'PY'
from pathlib import Path
import polars as pl

root = Path('.tmp_runs/datasets/habr-ir/pilot-release-v1')
pairs = pl.concat([
    pl.read_parquet(root / 'data/pairs/train.parquet'),
    pl.read_parquet(root / 'data/pairs/validation.parquet'),
    pl.read_parquet(root / 'data/pairs/test.parquet'),
])
qrels = pl.concat([
    pl.read_parquet(root / 'data/qrels/train.parquet'),
    pl.read_parquet(root / 'data/qrels/validation.parquet'),
    pl.read_parquet(root / 'data/qrels/test.parquet'),
])
corpus = pl.read_parquet(root / 'data/corpus-100k/corpus.parquet')
review = pl.read_csv(root / 'audit/pilot-review.csv')

assert pairs.group_by('split').len().sort('split').to_dicts() == [
    {'split': 'test', 'len': 10},
    {'split': 'train', 'len': 76},
    {'split': 'validation', 'len': 10},
]
assert pairs.height == qrels.height == 96
assert corpus.height == 100_000
assert review.height == 100
assert all(evidence in positive for evidence, positive in pairs.select('evidence', 'positive_passage').iter_rows())
assert qrels['query_id'].n_unique() == 96
assert set(qrels['passage_id']) == set(pairs['positive_passage_id'])
print({'pairs': pairs.height, 'corpus': corpus.height, 'qrels': qrels.height, 'review': review.height})
PY
```

Expected: assertions pass and the four counts are printed.

- [ ] **Step 3: Verify Hugging Face-native local loading**

```bash
conda run -n justatom python - <<'PY'
from datasets import load_dataset

root = '.tmp_runs/datasets/habr-ir/pilot-release-v1'
pairs = load_dataset(root, 'pairs')
corpus = load_dataset(root, 'corpus-100k', split='train')
qrels = load_dataset(root, 'qrels')
assert pairs.num_rows == {'train': 76, 'validation': 10, 'test': 10}
assert len(corpus) == 100_000
assert qrels.num_rows == {'train': 76, 'validation': 10, 'test': 10}
print(pairs)
PY
```

Expected: all configurations load without a custom dataset script.

- [ ] **Step 4: Record artifact checksums and human-audit instructions**

Update `docs/research/habr-ir-v4-generation-pilot.md` with the release manifest
fingerprint, Parquet counts/checksums, audit CSV path, and these allowed human
labels for each boolean audit column: empty means unreviewed, `true` means pass,
and `false` means fail. State explicitly that scale remains disabled until the
100-row audit has been reviewed and summarized.

- [ ] **Step 5: Run repository verification**

```bash
conda run -n justatom python -m pytest -m 'not integration' -q
conda run -n justatom python -m black --check \
  justatom/tooling/ir_dataset/release.py \
  justatom/tooling/ir_dataset/__init__.py \
  justatom/api/ir_dataset.py \
  tests/test_ir_release.py \
  tests/test_ir_dataset_cli.py
conda run -n justatom python -m compileall -q justatom
git diff --check
```

Expected: all unit tests and focused formatting checks pass; compile and diff
checks produce no errors.

- [ ] **Step 6: Commit Task 4 documentation**

```bash
git add docs/research/habr-ir-v4-generation-pilot.md
git commit -m "docs: record Habr IR pilot release artifacts"
```

The generated release directory remains ignored and is not committed.
