# Dataset Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace mixed eager/lazy dataset loading with one strict, Polars-first API that resolves local files, the packaged demo, and ordinary Hugging Face IDs without custom URIs.

**Architecture:** A typed resolver converts a user source into a local, packaged, or Hugging Face source. Focused readers expose lazy row iteration and eager Polars collection, while `DatasetLoader` enforces the public return contract. Train, eval, dataset generation, and the HTTP indexing API consume that facade exactly once.

**Tech Stack:** Python 3.12, Polars 1.38.1, Hugging Face `datasets` 2.18.0, `huggingface_hub`, pytest.

## Global Constraints

- `lazy=True` always returns `Iterator[dict[str, Any]]`.
- `lazy=False` always returns `polars.DataFrame`.
- JSON and XLSX raise `DatasetStreamingUnsupportedError` for lazy reads.
- No lazy reader may fall back to eager materialization.
- Remove `builtin://`, `hf://`, the magic source `justatom`, and the dead source `url` without compatibility aliases.
- Direct HTTP and HTTPS dataset loading is out of scope.
- Keep pandas only in visualization; core loading and filters use Polars.
- Preserve training limit semantics after query expansion.
- Do not modify `phd.paper` or `justatom-rc`.
- Run commands through `conda run -n justatom`.

---

## File Map

**Create**

- `justatom/storing/datasets/__init__.py`: public exports.
- `justatom/storing/datasets/api.py`: strict `DatasetLoader.read()` facade.
- `justatom/storing/datasets/errors.py`: dataset-specific exception hierarchy.
- `justatom/storing/datasets/readers.py`: local, packaged, and Hugging Face readers.
- `justatom/storing/datasets/source.py`: source dataclasses, options, and resolver.
- `tests/test_dataset_loader.py`: resolver and local contract tests.
- `tests/test_hf_dataset_loader.py`: Hugging Face streaming, eager, auth, and fallback tests.
- `tests/test_filters.py`: Polars filter-value coverage.

**Modify**

- `justatom/tooling/dataset.py`: consume strict loader results without type guessing.
- `justatom/training/data.py`: open each source once and preserve query-expanded limit behavior.
- `justatom/training/config.py`: add typed `lazy`, `config`, and `drop_columns` dataset fields.
- `justatom/api/eval.py`: pass loader options and capture labels during the indexing stream.
- `justatom/api/datasets.py`: use the loader and internal packaged prompt defaults.
- `justatom/api/run.py`: use the loader in `/indexing`.
- `justatom/configuring/builtins.py`: remove public built-in URI helpers.
- `justatom/etc/filters.py`: replace pandas frame checks with Polars checks.
- dataset YAML presets, README, `LAUNCH.md`, and `docs/launch-guide.md`: new source syntax.
- `pyproject.toml` and `requirements.txt`: remove `jsonlines` when unused.
- existing dataset, train, eval, and scenario tests: assert the new contract.

**Delete**

- `justatom/storing/dataset.py`: mixed legacy loader.
- obsolete HF loader tests after equivalent coverage moves to the new test module.
- `IDataset` from `justatom/storing/mask.py` when no references remain.

---

### Task 1: Typed Sources and Actionable Errors

**Files:**
- Create: `justatom/storing/datasets/errors.py`
- Create: `justatom/storing/datasets/source.py`
- Create: `tests/test_dataset_loader.py`

**Interfaces:**
- Produces: `DatasetReadOptions`, `LocalDatasetSource`, `PackagedDatasetSource`, `HuggingFaceDatasetSource`, `resolve_dataset_source()`.
- Produces: `DatasetError`, `DatasetNotFoundError`, `UnsupportedDatasetSourceError`, `UnsupportedDatasetFormatError`, `DatasetStreamingUnsupportedError`, `DatasetReadError`.

- [ ] **Step 1: Write resolver tests**

Add tests proving local-path priority, `~` expansion, `demo`, ordinary Hub IDs, legacy URI rejection, HTTP rejection, and no magic `justatom` source:

```python
def test_existing_owner_dataset_path_wins_over_hf(tmp_path, monkeypatch):
    local = tmp_path / "owner" / "dataset"
    local.parent.mkdir()
    local.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    source = resolve_dataset_source("owner/dataset")

    assert source == LocalDatasetSource(local.resolve())


@pytest.mark.parametrize("value", ["builtin://datasets/demo.jsonl", "hf://owner/data"])
def test_legacy_uri_is_rejected(value):
    with pytest.raises(UnsupportedDatasetSourceError, match="custom URI syntax"):
        resolve_dataset_source(value)


def test_bare_justatom_is_not_magic():
    with pytest.raises(DatasetNotFoundError):
        resolve_dataset_source("justatom")
```

- [ ] **Step 2: Verify resolver tests fail**

Run:

```bash
conda run -n justatom pytest tests/test_dataset_loader.py -q
```

Expected: import failure because `justatom.storing.datasets` does not exist.

- [ ] **Step 3: Implement errors, read options, and resolver**

Implement immutable dataclasses and strict validation:

```python
@dataclass(frozen=True)
class DatasetReadOptions:
    split: str | None = None
    config: str | None = None
    limit: int | None = None
    drop_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.limit is not None and self.limit < 0:
            raise ValueError("dataset limit must be >= 0")


def resolve_dataset_source(value: str | Path) -> DatasetSource:
    raw = str(value).strip()
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return LocalDatasetSource(candidate.resolve())
    if raw == "demo":
        return PackagedDatasetSource(name="demo", resource=_demo_resource())
    if raw.startswith(("http://", "https://")):
        raise UnsupportedDatasetSourceError(
            "HTTP dataset sources are not supported yet; download the file and pass its local path."
        )
    if "://" in raw:
        raise UnsupportedDatasetSourceError(
            "Dataset custom URI syntax is not supported; use owner/dataset for Hugging Face or a local path."
        )
    if _HF_REPO_ID.fullmatch(raw):
        return HuggingFaceDatasetSource(repo_id=raw)
    raise DatasetNotFoundError(
        f"Dataset source {raw!r} is neither an existing file, 'demo', nor a Hugging Face owner/dataset ID."
    )
```

- [ ] **Step 4: Run resolver tests**

Run `conda run -n justatom pytest tests/test_dataset_loader.py -q`.

Expected: resolver tests pass.

- [ ] **Step 5: Commit**

```bash
git add justatom/storing/datasets/errors.py justatom/storing/datasets/source.py tests/test_dataset_loader.py
git commit -m "refactor: add typed dataset source resolver"
```

---

### Task 2: Strict Local and Packaged Readers

**Files:**
- Create: `justatom/storing/datasets/readers.py`
- Modify: `tests/test_dataset_loader.py`

**Interfaces:**
- Consumes: source dataclasses and `DatasetReadOptions` from Task 1.
- Produces: `iter_source_rows(source, options)` and `source_to_frame(source, options)`.

- [ ] **Step 1: Add parameterized local format tests**

Create CSV, JSONL, NDJSON, Parquet, JSON, and XLSX fixtures. Assert exact return behavior:

```python
@pytest.mark.parametrize("suffix", [".csv", ".jsonl", ".ndjson", ".parquet"])
def test_streaming_local_formats_return_iterator(dataset_file):
    source = LocalDatasetSource(dataset_file.resolve())

    rows = iter_source_rows(source, DatasetReadOptions(limit=1))

    assert isinstance(rows, Iterator)
    assert list(rows) == [{"id": 1, "content": "one"}]


@pytest.mark.parametrize("suffix", [".json", ".xlsx"])
def test_non_streaming_formats_fail_before_read(dataset_file, monkeypatch):
    source = LocalDatasetSource(dataset_file.resolve())

    with pytest.raises(DatasetStreamingUnsupportedError, match="lazy=False"):
        iter_source_rows(source, DatasetReadOptions())
```

Add assertions that eager reads return `pl.DataFrame`, `drop_columns` is applied, and wrapped JSON `{\"data\": [...]}` remains supported.

- [ ] **Step 2: Verify reader tests fail**

Run `conda run -n justatom pytest tests/test_dataset_loader.py -q`.

Expected: failures because reader functions do not exist.

- [ ] **Step 3: Implement local scanning and eager collection**

Use one Polars lazy pipeline for streaming formats:

```python
def _iter_lazy_frame(frame: pl.LazyFrame) -> Iterator[dict[str, Any]]:
    for batch in frame.collect_batches(maintain_order=True):
        yield from batch.iter_rows(named=True)


def _apply_lazy_options(frame: pl.LazyFrame, options: DatasetReadOptions) -> pl.LazyFrame:
    names = set(frame.collect_schema().names())
    removable = [name for name in options.drop_columns if name in names]
    if removable:
        frame = frame.drop(removable)
    if options.limit is not None:
        frame = frame.limit(options.limit)
    return frame
```

Dispatch `.csv`, `.jsonl`, `.ndjson`, and `.parquet` to scans. Raise before opening `.json` and `.xlsx` in lazy mode. In eager mode return a Polars frame for all supported extensions and preserve existing JSON shapes.

For `PackagedDatasetSource`, enter `as_file(source.resource)` inside the generator body and yield all rows before exiting the context.

- [ ] **Step 4: Run focused tests**

Run `conda run -n justatom pytest tests/test_dataset_loader.py -q`.

Expected: all resolver and local reader tests pass.

- [ ] **Step 5: Commit**

```bash
git add justatom/storing/datasets/readers.py tests/test_dataset_loader.py
git commit -m "refactor: add strict Polars dataset readers"
```

---

### Task 3: Hugging Face Reader and Public Facade

**Files:**
- Create: `justatom/storing/datasets/api.py`
- Create: `justatom/storing/datasets/__init__.py`
- Create: `tests/test_hf_dataset_loader.py`
- Modify: `justatom/storing/datasets/readers.py`
- Modify: `tests/test_dataset_loader.py`

**Interfaces:**
- Produces: overloaded `DatasetLoader.read(source, *, lazy, split, config, limit, drop_columns)`.
- Lazy result: `Iterator[dict[str, Any]]`.
- Eager result: `pl.DataFrame`.

- [ ] **Step 1: Write HF and facade tests**

Mock `load_dataset`, `list_repo_files`, and `hf_hub_download`. Verify:

```python
def test_hf_lazy_uses_streaming_and_stops_at_limit(monkeypatch):
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return iter([{"id": 1}, {"id": 2}, {"id": 3}])

    monkeypatch.setattr(readers, "load_dataset", fake_load_dataset)
    rows = DatasetLoader.read("owner/data", lazy=True, split="train", limit=2)

    assert list(rows) == [{"id": 1}, {"id": 2}]
    assert calls[0][1]["streaming"] is True


def test_hf_eager_converts_arrow_to_polars(monkeypatch):
    fake = FakeHFDataset(pa.table({"id": [1, 2], "blob": ["a", "b"]}))
    monkeypatch.setattr(readers, "load_dataset", lambda *args, **kwargs: fake)

    frame = DatasetLoader.read("owner/data", lazy=False, drop_columns=["blob"])

    assert isinstance(frame, pl.DataFrame)
    assert frame.to_dicts() == [{"id": 1}, {"id": 2}]
```

Also test token priority, config forwarding, split fallback, lazy/eager parquet fallback, local option rejection, query-string rejection, and facade overload behavior at runtime.

- [ ] **Step 2: Verify HF tests fail**

Run:

```bash
conda run -n justatom pytest tests/test_hf_dataset_loader.py tests/test_dataset_loader.py -q
```

Expected: failure because `DatasetLoader` and HF readers are absent.

- [ ] **Step 3: Implement HF lazy/eager readers**

Lazy behavior:

```python
def _iter_hf_rows(source: HuggingFaceDatasetSource, options: DatasetReadOptions):
    dataset = _load_hf_split(source, options, streaming=True)
    rows = (dict(row) for row in dataset)
    if options.drop_columns:
        rows = ({key: value for key, value in row.items() if key not in options.drop_columns} for row in rows)
    return rows if options.limit is None else islice(rows, options.limit)
```

Eager behavior converts `dataset.data.table` with `pl.from_arrow`, applies projection and `head()`, and returns only `pl.DataFrame`. Preserve environment token priority and split chains. Preserve parquet fallback while converting its lazy scan to row batches at the reader boundary.

- [ ] **Step 4: Implement strict facade**

Use overloads and no backend kwargs:

```python
class DatasetLoader:
    @staticmethod
    @overload
    def read(source: str | Path, *, lazy: Literal[True], split: str | None = None,
             config: str | None = None, limit: int | None = None,
             drop_columns: Sequence[str] | None = None) -> Iterator[dict[str, Any]]:
        pass

    @staticmethod
    @overload
    def read(source: str | Path, *, lazy: Literal[False], split: str | None = None,
             config: str | None = None, limit: int | None = None,
             drop_columns: Sequence[str] | None = None) -> pl.DataFrame:
        pass
```

The concrete method resolves once, creates `DatasetReadOptions`, validates local/HF option compatibility, and dispatches to one reader operation.

- [ ] **Step 5: Run focused tests**

Run `conda run -n justatom pytest tests/test_dataset_loader.py tests/test_hf_dataset_loader.py -q`.

Expected: all new loader tests pass.

- [ ] **Step 6: Commit**

```bash
git add justatom/storing/datasets tests/test_dataset_loader.py tests/test_hf_dataset_loader.py
git commit -m "refactor: expose strict dataset loader facade"
```

---

### Task 4: Typed Config, Record Adapter, and Training

**Files:**
- Modify: `justatom/training/config.py`
- Modify: `justatom/tooling/dataset.py`
- Modify: `justatom/training/data.py`
- Modify: `tests/test_training_config.py`
- Modify: `tests/test_eval_data_normalization.py`
- Modify: `tests/test_train_data_preparation.py`

**Interfaces:**
- Consumes: `DatasetLoader.read()` from Task 3.
- Produces: `DatasetConfig.lazy`, `DatasetConfig.config`, and `DatasetConfig.drop_columns`.

- [ ] **Step 1: Write typed config and single-open tests**

Assert YAML-shaped lists normalize to tuples and round-trip:

```python
def test_dataset_loader_options_are_typed_and_round_trip():
    config = parse_train_config({
        "method": "vanilla",
        "dataset": {
            "name_or_path": "owner/data",
            "lazy": True,
            "config": "russian",
            "drop_columns": ["photos"],
        },
    })

    assert config.dataset.lazy is True
    assert config.dataset.config == "russian"
    assert config.dataset.drop_columns == ("photos",)
    assert parse_train_config(train_config_to_dict(config)) == config
```

Replace old multi-type adapter tests with exact lazy iterator/eager Polars tests. Add a loader spy proving `iterate_training_rows()` calls `DatasetLoader.read()` once and applies `limit` after expanding multiple queries from one source row.

- [ ] **Step 2: Verify focused tests fail**

Run:

```bash
conda run -n justatom pytest tests/test_training_config.py tests/test_eval_data_normalization.py tests/test_train_data_preparation.py -q
```

Expected: failures for missing config fields and old loader calls.

- [ ] **Step 3: Add config fields and validation**

Add:

```python
@dataclass(frozen=True)
class DatasetConfig:
    id: str | None = None
    name_or_path: str | None = None
    lazy: bool = True
    config: str | None = None
    labels_field: str = "queries"
    content_field: str = "content"
    split: str | None = None
    limit: int | None = None
    drop_columns: tuple[str, ...] = ()
```

Normalize list-like `drop_columns` to a tuple before overlaying the dataclass. Remove `drop_columns` from `_DATASET_METADATA_FIELDS`. Validate `lazy` as bool and non-negative `limit`.

- [ ] **Step 4: Simplify `DatasetRecordAdapter.from_source()`**

Replace signature introspection with explicit loader parameters. Call `DatasetLoader.read()` once and convert only an eager Polars frame with `iter_rows(named=True)`. Keep normalization methods and duplicate ID behavior unchanged.

- [ ] **Step 5: Simplify training iteration**

Delete `_frame_batches_from_source`, `_iterate_from_frame_batches`, and the adapter reload branch. Add loader options to `iterate_training_rows()` and call:

```python
source = DatasetLoader.read(
    dataset_name_or_path,
    lazy=lazy,
    split=split,
    config=config,
    drop_columns=drop_columns,
)
samples = source if lazy else source.iter_rows(named=True)
rows = _iterate_from_raw_samples(samples, normalized_arguments)
return rows if limit is None else islice(rows, limit)
```

Pass config fields from `prepare_training_data_from_config()`. Do not pass the training limit into `DatasetLoader`.

- [ ] **Step 6: Run focused tests**

Run the three focused test modules from Step 2.

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add justatom/training/config.py justatom/tooling/dataset.py justatom/training/data.py tests/test_training_config.py tests/test_eval_data_normalization.py tests/test_train_data_preparation.py
git commit -m "refactor: unify training dataset consumption"
```

---

### Task 5: Evaluation and Remaining API Consumers

**Files:**
- Modify: `justatom/api/eval.py`
- Modify: `justatom/api/datasets.py`
- Modify: `justatom/api/run.py`
- Modify: `justatom/builtins/configs/datasets.default.yaml`
- Modify: `tests/test_eval_streaming_integration.py`
- Modify: `tests/test_scenario_configs.py`
- Create or modify: focused tests for `justatom.api.datasets`

**Interfaces:**
- Consumes: strict loader and explicit adapter options.
- Produces: single-open eval flow and internal packaged prompt defaults.

- [ ] **Step 1: Write eval single-open and prompt tests**

Mock `DatasetRecordAdapter.from_source()` and assert one call while queries remain available after indexing. Assert null prompt paths load packaged prompt text, while a non-null path reads only the filesystem.

```python
def test_eval_captures_labels_while_indexing(monkeypatch):
    calls = []
    documents = [
        {"content": "one", "meta": {"labels": ["q1"]}},
        {"content": "two", "meta": {"labels": ["q2"]}},
    ]
    monkeypatch.setattr(DatasetRecordAdapter, "from_source", fake_adapter(calls, documents))

    asyncio.run(run_eval_with_fake_service())

    assert len(calls) == 1
    assert captured_queries == ["q1", "q2"]
```

- [ ] **Step 2: Verify API tests fail**

Run:

```bash
conda run -n justatom pytest tests/test_scenario_configs.py tests/test_eval_streaming_integration.py -q
```

Expected: old URI and two-open assumptions fail after new assertions.

- [ ] **Step 3: Refactor eval**

Add resolved dataset fields `dataset_lazy` and `dataset_config`. Remove URI/path pre-resolution. Wrap the document iterator:

```python
queries: list[str] = []

def capture_labels(documents: Iterable[dict[str, Any]]):
    for document in documents:
        labels = document.get("meta", {}).get("labels", [])
        if isinstance(labels, str):
            labels = [labels]
        queries.extend(label for label in labels if isinstance(label, str) and label.strip())
        yield document
```

Index `capture_labels(docs_adapter.iterator())`, remove the second adapter, and evaluate with `queries`.

- [ ] **Step 4: Refactor dataset generation prompts and source reading**

When prompt paths are null, load `prompts/datasets_system_prompt.txt` and `prompts/datasets_user_template_prompt.txt` through `load_builtin_prompt()`. When provided, treat paths as ordinary files. Forward lazy, config, split, limit, and drop columns to `DatasetLoader.read()`.

- [ ] **Step 5: Refactor `/indexing`**

Replace the singleton call with:

```python
docs = (
    list(DatasetLoader.read(dataset_name_or_docs, lazy=True))
    if isinstance(dataset_name_or_docs, str)
    else normalize_inline_documents(dataset_name_or_docs)
)
```

- [ ] **Step 6: Run focused API tests**

Run scenario, eval normalization, eval streaming integration, and dataset generation tests.

Expected: all pass; integration may skip only when Weaviate cannot be started.

- [ ] **Step 7: Commit**

```bash
git add justatom/api/eval.py justatom/api/datasets.py justatom/api/run.py justatom/builtins/configs/datasets.default.yaml tests
git commit -m "refactor: share dataset loader across APIs"
```

---

### Task 6: Preset Migration, Core Polars, and Documentation

**Files:**
- Modify: `configs/dataset/*.yaml`
- Modify: `justatom/builtins/configs/dataset/*.yaml`
- Modify: `justatom/configuring/builtins.py`
- Modify: `justatom/etc/filters.py`
- Create: `tests/test_filters.py`
- Modify: `tests/test_scenario_configs.py`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `README.md`
- Modify: `LAUNCH.md`
- Modify: `docs/launch-guide.md`

**Interfaces:**
- Preserves: `dataset.id` preset resolution.
- Removes: all active custom dataset/resource URI syntax.

- [ ] **Step 1: Write migration and Polars filter tests**

Assert preset values:

```python
def test_justatom_preset_uses_explicit_eager_json():
    config = resolve_train_config(config={"method": "vanilla", "dataset": {"id": "justatom"}})

    assert config.dataset.name_or_path == ".data/polaroids.ai.data.json"
    assert config.dataset.lazy is False


def test_demo_preset_uses_short_source_name():
    kwargs = resolve_eval_kwargs(config={"dataset": {"id": "demo-eval"}})

    assert kwargs["dataset_name_or_path"] == "demo"
```

Add filter tests using `pl.DataFrame` and assert `justatom.etc.filters` has no pandas import.

- [ ] **Step 2: Verify migration tests fail**

Run `conda run -n justatom pytest tests/test_scenario_configs.py tests/test_filters.py -q`.

Expected: old preset values and pandas behavior fail.

- [ ] **Step 3: Migrate active presets**

Apply these exact forms:

```yaml
# configs/dataset/justatom.yaml
name_or_path: .data/polaroids.ai.data.json
lazy: false
```

```yaml
# built-in demo presets
name_or_path: demo
lazy: true
```

Remove `hf://` prefixes. Move mMARCO `?config=russian` to `config: russian`. Replace the mMARCO manifest custom URI with a standard HTTPS resolve URL.

- [ ] **Step 4: Remove public built-in path resolution**

Delete `is_builtin_uri()` and `resolve_builtin_path()`. Retain private `importlib.resources` helpers used by scenario YAML and prompt defaults.

- [ ] **Step 5: Replace pandas in filters**

Use `import polars as pl`, reject `list` and `pl.DataFrame` for ordered comparisons, and serialize a Polars frame with `write_json()` only where equality handling previously serialized pandas.

- [ ] **Step 6: Remove unused JSONL dependency and update docs**

Remove `jsonlines` from both dependency files. Rewrite current source examples to local paths, `demo`, or `owner/dataset`; document `lazy: false` for JSON/XLSX and separate HF `config`/`split` fields.

- [ ] **Step 7: Run focused tests and URI scans**

Run:

```bash
conda run -n justatom pytest tests/test_scenario_configs.py tests/test_filters.py -q
rg -n "builtin://|hf://" README.md LAUNCH.md docs/launch-guide.md configs justatom/builtins
```

Expected: tests pass; `rg` returns no matches.

- [ ] **Step 8: Commit**

```bash
git add configs justatom/builtins justatom/configuring/builtins.py justatom/etc/filters.py tests/test_scenario_configs.py tests/test_filters.py pyproject.toml requirements.txt README.md LAUNCH.md docs/launch-guide.md
git commit -m "refactor: migrate dataset configuration syntax"
```

---

### Task 7: Remove Legacy Loader and Verify the Repository

**Files:**
- Delete: `justatom/storing/dataset.py`
- Modify: `justatom/storing/mask.py`
- Delete or replace: `tests/test_hf_dataset_auth.py`
- Delete or replace: `tests/test_hf_dataset_fallback.py`
- Modify: `justatom/api/eval.py`, `justatom/api/datasets.py`, `justatom/api/run.py`, `justatom/tooling/dataset.py`, and `justatom/training/data.py` if the final scan still finds a legacy import.
- Modify: `docs/superpowers/specs/2026-08-05-dataset-loading-design.md`

**Interfaces:**
- Leaves `justatom.storing.datasets.DatasetLoader` as the only production dataset-loading entry point.

- [ ] **Step 1: Scan for legacy references**

Run:

```bash
rg -n "storing\.dataset|DatasetApi|ByName|JUSTATOMDataset|URLInJSONDataset|IDataset|resolve_builtin_path|jsonlines" justatom tests pyproject.toml requirements.txt
```

Expected: matches identify only files scheduled for deletion or migration.

- [ ] **Step 2: Delete legacy implementation and redundant tests**

Delete the old module and old tests only after all equivalent assertions exist in `test_dataset_loader.py` and `test_hf_dataset_loader.py`. Remove `IDataset` from `storing/mask.py` when the reference scan is empty.

- [ ] **Step 3: Run focused dataset and consumer suites**

Run:

```bash
conda run -n justatom pytest tests/test_dataset_loader.py tests/test_hf_dataset_loader.py tests/test_eval_data_normalization.py tests/test_train_data_preparation.py tests/test_training_config.py tests/test_scenario_configs.py tests/test_filters.py -q
```

Expected: all pass.

- [ ] **Step 4: Run the full suite**

Run:

```bash
conda run -n justatom pytest -q
```

Expected: all tests pass with no new warnings from dataset loading.

- [ ] **Step 5: Run final static scans**

```bash
rg -n "storing\.dataset|DatasetApi|ByName|JUSTATOMDataset|URLInJSONDataset|resolve_builtin_path|jsonlines" justatom tests pyproject.toml requirements.txt
rg -n "builtin://|hf://" README.md LAUNCH.md docs/launch-guide.md configs justatom/builtins
rg -n "from pandas|import pandas" justatom/storing justatom/training justatom/tooling/dataset.py justatom/etc/filters.py
git diff --check
```

Expected: all three `rg` commands return no matches; `git diff --check` returns success.

- [ ] **Step 6: Mark the design implemented and commit**

Change the design status to `Implemented`, then commit:

```bash
git add -A
git commit -m "refactor: remove legacy dataset loading API"
```

- [ ] **Step 7: Review final branch history and diff**

Run:

```bash
git status --short --branch
git log --oneline origin/master..HEAD
git diff --stat origin/master...HEAD
```

Expected: clean branch, focused commits in task order, and no changes below `phd.paper` or `justatom-rc`.
