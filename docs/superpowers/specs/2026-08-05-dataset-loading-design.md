# Dataset Loading Design

**Date:** 2026-08-05
**Status:** Implemented
**Scope:** root `justatom` package only

## 1. Context

Dataset loading currently exposes several incompatible meanings for the same
`lazy` flag:

- local CSV, JSONL, and Parquet sources may return a Polars `LazyFrame`;
- local JSON and XLSX sources silently materialize data even when `lazy=True`;
- Hugging Face sources return a `datasets.Dataset` and force
  `streaming=False`;
- a Hugging Face parquet fallback returns either a Polars `DataFrame` or
  `LazyFrame`;
- `DatasetRecordAdapter` then inspects the result and guesses how to turn it
  into rows;
- the training path first probes for a Polars frame and opens non-Polars
  sources a second time through the adapter.

Source resolution is similarly overloaded. Local paths, `builtin://` URIs,
`hf://` URIs, bare Hugging Face IDs, the magic name `justatom`, and the
unimplemented name `url` are handled by one `ByName.named()` method. This
makes the public contract difficult to explain and creates hidden behavior,
including `justatom` resolving to `.data/polaroids.ai.data.json`.

This repository has not yet shipped its stable PyPI data-loading API. The
cleanup therefore intentionally replaces the old API instead of maintaining a
deprecation layer.

## 2. Goals

1. Give `lazy` one exact meaning across all supported sources.
2. Return a true row iterator in lazy mode and a Polars frame in eager mode.
3. Never hide eager materialization behind `lazy=True`.
4. Resolve each source once and read it once.
5. Make local paths, packaged demo data, and Hugging Face datasets simple to
   identify without custom URI syntax.
6. Use Polars throughout the core data path and core filter validation.
7. Keep format, transport, and train/eval normalization concerns separate.
8. Preserve bounded-memory sampling and evaluation behavior.

## 3. Non-goals

- Supporting direct HTTP or HTTPS dataset URLs in this change.
- Adding a download cache or wrapping `wget`.
- Implementing streaming parsers for JSON arrays or XLSX files.
- Rewriting plotting and visualization code that currently interoperates with
  pandas.
- Changing `content`, `queries`, keyword, or document normalization semantics.
- Changing the training objective, evaluation metrics, or Weaviate behavior.
- Updating the nested `justatom-rc` repository.
- Accepting `builtin://` or `hf://` as deprecated aliases.

## 4. Public Contract

The public facade is `DatasetLoader` in `justatom.storing.datasets`:

```python
from justatom.storing.datasets import DatasetLoader

rows = DatasetLoader.read(
    "justatom/meme-russian-ir",
    lazy=True,
    split="train",
)

frame = DatasetLoader.read(
    "data/train.parquet",
    lazy=False,
)
```

The return type is strict:

```text
lazy=True  -> Iterator[dict[str, Any]]
lazy=False -> polars.DataFrame
```

The implementation should provide overloads using `Literal[True]` and
`Literal[False]` so static type checkers see the same contract.

`DatasetLoader.read()` accepts source-independent options:

```text
source: str | Path
lazy: bool
split: str | None
config: str | None
limit: int | None
drop_columns: Sequence[str] | None
```

Reader-specific Polars or Hugging Face kwargs are not passed through the
public API. New options are added deliberately to the typed contract instead
of leaking backend APIs through `**kwargs`.

The loader validates option/source combinations. `split` and `config` are
valid for Hugging Face sources and rejected for local or packaged files when
non-null. For Hugging Face sources, query strings embedded in `source` are
rejected with guidance to use the separate fields.

## 5. Components

Dataset loading becomes a small package:

```text
justatom/storing/datasets/
|- __init__.py
|- api.py
|- errors.py
|- readers.py
`- source.py
```

### 5.1 Source resolver

`source.py` contains immutable source descriptions:

```text
LocalDatasetSource(path: Path)
PackagedDatasetSource(name: str, resource: Traversable)
HuggingFaceDatasetSource(repo_id: str)
```

Resolution uses this fixed priority:

1. An existing local file is a `LocalDatasetSource`.
2. The exact name `demo` is the packaged retrieval demo.
3. A string matching `owner/dataset` is a Hugging Face source.
4. `http://` and `https://` raise an unsupported-source error.
5. Any URI-like value, including `builtin://` and `hf://`, raises an
   unsupported-source error with an example of the new syntax.
6. Any remaining value raises a dataset-not-found error.

Checking an existing path before the Hugging Face pattern lets a local path
such as `owner/dataset` win when it exists. Directories are not dataset
sources in this version.

The exact name `justatom` has no resolver-level meaning. A dataset preset may
still be named `justatom`, but its `name_or_path` must be explicit.

### 5.2 Readers

`readers.py` dispatches a resolved source to a local, packaged, or Hugging
Face reader. Readers expose two internal operations:

```python
iter_rows(options) -> Iterator[dict[str, Any]]
to_frame(options) -> pl.DataFrame
```

`DatasetLoader.read()` chooses one operation from `lazy`; consumers never
inspect backend-specific result types.

A small immutable `DatasetReadOptions` value carries normalized `split`,
`config`, `limit`, and `drop_columns` values. Normalization and validation
happen once at the facade boundary instead of independently in each reader.

Packaged demo data uses the same local JSONL reader as a user-provided JSONL
file. Packaging only changes how the path is obtained. When an installed
package requires `importlib.resources.as_file()`, the lazy generator owns that
context for its complete lifetime so the temporary path cannot disappear
between `read()` and row consumption.

## 6. Local Format Behavior

| Format | `lazy=True` | `lazy=False` |
| --- | --- | --- |
| CSV | `scan_csv` plus ordered batches | `read_csv` to `pl.DataFrame` |
| JSONL/NDJSON | `scan_ndjson` plus ordered batches | `read_ndjson` to `pl.DataFrame` |
| Parquet | `scan_parquet` plus ordered batches | `read_parquet` to `pl.DataFrame` |
| JSON | explicit streaming error | eager parse to `pl.DataFrame` |
| XLSX | explicit streaming error | `read_excel` to `pl.DataFrame` |

Lazy Polars scans are converted to row dictionaries by iterating
`collect_batches(maintain_order=True)`. They are never collected into one
frame first.

`drop_columns` is projected before batch collection. `limit` is applied to
the lazy query before batches are consumed and before eager results are
returned. A negative limit is rejected.

The supported newline-delimited extensions are `.jsonl` and `.ndjson`.
Source paths expand `~` before existence checks and otherwise remain relative
to the current working directory.

Eager JSON keeps the supported input shapes from the existing reader:

- a top-level list of records;
- a mapping with a top-level `data` list;
- one top-level record.

JSON parsing may use the standard or current JSON parser before constructing
a Polars frame. The requirement is a Polars result, not a forced Polars JSON
parser for shapes it does not represent correctly.

There is no `lazy -> eager` fallback. A reader failure is reported as a read
failure in the requested mode.

## 7. Hugging Face Behavior

Hugging Face IDs use the ordinary Hub form:

```yaml
dataset:
  name_or_path: unicamp-dl/mmarco
  config: russian
  split: train
  limit: 50000
  drop_columns:
    - unused_blob
```

Query parameters are not encoded in `name_or_path`.

### 7.1 Lazy mode

Lazy mode calls:

```python
datasets.load_dataset(
    repo_id,
    name=config,
    split=split,
    streaming=True,
    token=resolved_token,
)
```

The returned `IterableDataset` is mapped to dictionaries. Columns are removed
before yielding where the backend supports it, and `limit` is enforced with
bounded iteration. Only consumed rows are requested from the stream.

### 7.2 Eager mode

Eager mode calls `load_dataset(..., streaming=False)`. Its Arrow table is
converted directly to Polars without a pandas intermediate. Column removal
and limits are applied before returning the final `pl.DataFrame` whenever the
backend permits pushdown.

### 7.3 Authentication and split fallback

Token lookup retains the existing environment priority:

```text
HF_TOKEN
HUGGINGFACE_HUB_TOKEN
HF_HUB_TOKEN
HF_API_KEY
```

Split fallback chains such as `dev|test` remain supported. Candidates are
tried in order in both modes. The final error includes the repository ID,
requested candidates, and the last backend error.

### 7.4 Parquet fallback

The current Hub parquet-shard fallback remains available for builder or
metadata incompatibilities. It may change the transport used to obtain the
data, but it must preserve the requested memory contract:

- lazy mode scans cached shards and yields ordered row batches;
- eager mode reads cached shards into a Polars frame.

A lazy parquet fallback must never return a `LazyFrame` to the consumer and
must never collect all shards before yielding the first row.

## 8. Consumer Flow

### 8.1 Record adapter

`DatasetRecordAdapter.from_source()` calls `DatasetLoader.read()` exactly
once. Its source records are either the returned iterator or
`frame.iter_rows(named=True)` for an explicitly eager call.

Its signature names loader options (`lazy`, `split`, `config`, `limit`, and
`drop_columns`) separately from document-normalization options. The current
`inspect.signature()` routing of arbitrary `**kwargs` is removed.

The `_to_records()` type-dispatch over Polars frames, lazy frames, Hugging
Face datasets, bytes, and generic iterables is removed. Row coercion may remain
as a narrow validation helper, but readers own backend conversion.

Document normalization remains in `DatasetRecordAdapter`.

### 8.2 Training

Training consumes `DatasetLoader.read(..., lazy=config.dataset.lazy)`
directly. The current probe for Polars frame batches and the subsequent
adapter reload are removed. Rows are normalized once and then query-expanded.

Training does not pass `dataset.limit` into the source reader because the
existing training contract applies that limit after query expansion. Eval and
dataset-generation limits remain source-row limits and may be pushed down by
the loader. Tests preserve this distinction explicitly.

Reservoir sampling remains bounded and deterministic for a finite requested
sample. Full training data is materialized only where the existing training
pipeline explicitly requires the sampled rows or final training frame.

### 8.3 Evaluation

Evaluation passes the same lazy row stream through `DatasetRecordAdapter` to
produce documents and labels. It does not special-case URI syntax or convert
potential Hub identifiers to `Path` objects before resolution.

Evaluation forwards the dataset's explicit `lazy` setting. Lazy remains the
default for streaming-capable sources; an eager source is converted to rows
with `frame.iter_rows(named=True)` inside the adapter without changing the
loader's return contract.

The indexing iterator records only each document's normalized labels before
yielding that document to the indexer. Once indexing consumes the stream, the
collected label strings are used for evaluation. The dataset is therefore not
opened a second time, passages are not retained in memory, and label memory is
no larger than the query list already required by `evaluate_topk()`.

### 8.4 Dataset generation

`justatom.api.datasets` uses the same loader and receives `config`, `split`,
`lazy`, `limit`, and `drop_columns` through the same public options. Its
generic Python-version advice is replaced by the specific dataset error and
chained backend cause.

### 8.5 Other API consumers

`justatom.api.run` and every remaining import of the old singleton API move to
`DatasetLoader`. No production caller may retain a private source-resolution
path.

## 9. Configuration and Built-in Resources

`DatasetConfig` gains first-class source options:

```python
lazy: bool = True
config: str | None = None
drop_columns: tuple[str, ...] = ()
```

They are no longer hidden in `dataset.metadata`. Scenario resolution and CLI
overrides preserve these typed fields.

Repository and packaged presets are migrated as follows:

```text
builtin://datasets/demo_retrieval.jsonl -> demo
hf://d0rj/boolq-ru                     -> d0rj/boolq-ru
hf://unicamp-dl/mmarco?config=russian  -> unicamp-dl/mmarco + config: russian
justatom                               -> explicit path in the justatom preset
```

The exact explicit source for the `justatom` preset is the repository dataset
path currently hidden by `JUSTATOMDataset`:

```yaml
name_or_path: .data/polaroids.ai.data.json
lazy: false
```

The explicit eager setting is required because JSON arrays do not support the
new streaming contract. At roughly 25 MB the repository dataset is suitable
for the explicit eager path. There is no silent mode switch.

The packaged smoke presets use `name_or_path: demo`.

The mMARCO manifest metadata also drops the custom `hf://` value. If the
manifest remains in the preset, it uses a normal Hugging Face HTTPS resolve
URL or separate repository and filename metadata; it is not parsed as a
dataset source.

`builtin://` is also removed from prompt configuration. Packaged dataset
generation defaults load their packaged system and user prompt resources when
the corresponding custom path is null. A user-provided `*_path` always means
a normal filesystem path; there is no public packaged-resource URI.

The `is_builtin_uri()` and `resolve_builtin_path()` helpers are removed.
Internal config and prompt loading may continue using `importlib.resources`
through private helpers.

The old `jsonlines` dependency is removed if no use remains after JSONL
loading moves fully to Polars.

## 10. Errors

The dataset package defines:

```text
DatasetError
|- DatasetNotFoundError
|- UnsupportedDatasetSourceError
|- UnsupportedDatasetFormatError
|- DatasetStreamingUnsupportedError
`- DatasetReadError
```

Expected messages are actionable:

- legacy `hf://owner/data`: use `owner/data`;
- legacy `builtin://datasets/...`: use `demo` or a local path;
- HTTP URL: download it locally before loading;
- JSON/XLSX lazy request: use `lazy=False` for a small file or convert it to
  JSONL/Parquet;
- unsupported extension: list the supported local extensions;
- missing source: explain both valid local-path and Hub-ID forms.

Backend exceptions are chained with `raise ... from exc` so diagnostics are
not discarded.

## 11. Polars Boundary

The core loading path and core filter validation do not import pandas.
`justatom.etc.filters` replaces pandas `DataFrame` checks with backend-neutral
collection checks and the relevant Polars types.

Visualization is out of scope. Existing Plotly/Altair code may continue to
import or convert through pandas, so pandas is not removed from all dependency
files in this change. The boundary is documented instead of claiming a
repository-wide pandas removal.

## 12. Tests

### 12.1 Resolver tests

- existing local paths win over the Hub-ID pattern;
- `demo` resolves to the packaged JSONL resource;
- `owner/dataset` resolves to Hugging Face;
- bare missing paths produce `DatasetNotFoundError`;
- HTTP, `builtin://`, and `hf://` produce actionable unsupported-source
  errors;
- the bare value `justatom` has no magic resolver behavior.

### 12.2 Contract tests

Parameterized tests cover every supported format and source:

| Source | lazy iterator | eager Polars |
| --- | --- | --- |
| CSV | yes | yes |
| JSONL | yes | yes |
| Parquet | yes | yes |
| JSON | explicit error | yes |
| XLSX | explicit error | yes |
| Hugging Face | yes | yes |
| packaged demo | yes | yes |

Tests assert exact return categories rather than accepting multiple backend
types.

### 12.3 Streaming tests

- reading one row does not iterate the remainder of a mocked source;
- Polars lazy readers use ordered batches without full collection;
- HF receives `streaming=True` in lazy mode and `False` in eager mode;
- `limit` stops source consumption;
- `drop_columns` is applied before rows leave the reader;
- JSON and XLSX never perform hidden eager reads for lazy requests;
- the `justatom` preset explicitly selects eager JSON and remains loadable;
- parquet fallback preserves lazy/eager contracts.

### 12.4 Consumer tests

- training opens a source once;
- adapter and evaluation paths consume the strict row contract;
- evaluation captures labels during indexing and opens the source once;
- reservoir sampling remains deterministic and bounded;
- custom content/query fields, duplicate chunk IDs, keyword normalization,
  and query expansion retain existing behavior;
- dataset generation uses the shared loader options.

### 12.5 Migration checks

- active configs, package data, README, and launch guide contain no
  `builtin://` or `hf://` values;
- built-in prompt defaults work without a public resource URI;
- `justatom` preset resolves to its explicit source;
- core data modules and filters contain no pandas import.

The full local suite and the existing CI commands must pass after focused
tests.

## 13. Removal List

The implementation removes:

- `justatom.storing.dataset.ByName` and its singleton `API`;
- `JUSTATOMDataset`;
- the unimplemented `URLInJSONDataset`;
- reader return unions containing `pl.LazyFrame` or Hugging Face dataset
  objects;
- `DatasetRecordAdapter` backend result guessing;
- the training frame-probe and second source open;
- the unused `IDataset` interface when no implementation references it;
- public `builtin://` and `hf://` parsing;
- hidden eager fallback for lazy reads;
- pandas imports from core filtering/data-loading code.

## 14. Acceptance Criteria

The change is complete when:

1. every lazy dataset read returns a true iterator and never a materialized
   frame disguised as lazy;
2. every eager dataset read returns `pl.DataFrame`;
3. unsupported lazy formats fail before reading the file body;
4. train, eval, and dataset generation share one source resolver and loader;
5. all API consumers use the shared loader and each opens its source once;
6. local, demo, and Hugging Face syntax is documented without custom URIs;
7. the old magic source names and dead source classes are gone;
8. focused tests and the full test suite pass;
9. no changes are made under `phd.paper` or `justatom-rc`.
