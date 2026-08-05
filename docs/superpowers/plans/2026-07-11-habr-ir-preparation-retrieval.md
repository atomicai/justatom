# Habr IR Preparation and Retrieval Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the local first milestone of Habr IR: Markdown-aware passage preparation, persistent BM25, dense E5 embeddings, and inspectable lexical/dense top-k neighbors.

**Architecture:** Preserve `justatom.api.datasets` and add a focused `justatom.api.ir_dataset` CLI. Pure modules under `justatom.tooling.ir_dataset` own chunking, source access, artifacts, sparse retrieval, and dense retrieval. This milestone stops before OpenAI generation; a later plan consumes its stable artifacts.

**Tech Stack:** Python 3.12, Polars/Parquet, Hugging Face Hub, `markdown-it-py`, `bm25s`, Transformers/PyTorch MPS, pytest.

## Global Constraints

- Source: private `justatom/habr-ds`, config `default`, split `train`.
- Read `HF_API_KEY` through existing environment conventions and never persist it.
- Parse Markdown through `markdown-it-py` tokens, not regular expressions.
- Preserve code content while dropping Markdown fences and language markers.
- Serialize as `passage: {title}\n{section}\n\n{content}`.
- Accept at most 504 E5 tokens; assert that no row exceeds 512.
- Stable SHA-256 IDs, seed `42`, no Weaviate dependency.
- Do not edit training recipes or unrelated dirty files.
- Use red-green-refactor for every production behavior.

## File Map

- `justatom/tooling/ir_dataset/chunking.py`: Markdown normalization and packing.
- `justatom/tooling/ir_dataset/source.py`: projected private HF parquet access.
- `justatom/tooling/ir_dataset/artifacts.py`: atomic Parquet/manifest preparation.
- `justatom/tooling/ir_dataset/sparse.py`: BM25S persistence and retrieval.
- `justatom/tooling/ir_dataset/dense.py`: E5 memmap and exact top-k.
- `justatom/tooling/ir_dataset/neighbors.py`: RRF union and diagnostics.
- `justatom/api/ir_dataset.py`: CLI orchestration.
- `configs/datasets/habr-ir.yaml`: reproducible local config.
- `tests/test_ir_*.py`: isolated unit/CLI coverage.

---

### Task 1: Markdown Normalization and Token-aware Packing

**Files:**
- Create: `tests/fixtures/habr_article.md`
- Create: `tests/test_ir_chunking.py`
- Create: `justatom/tooling/ir_dataset/__init__.py`
- Create: `justatom/tooling/ir_dataset/chunking.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces `ChunkingConfig`, `Passage`, `MarkdownPassageChunker`, `serialize_passage`.
- `chunk_article(row: Mapping[str, Any]) -> list[Passage]` feeds Task 2.

- [ ] **Step 1: Declare dependencies**

Add:

```text
markdown-it-py>=4.0,<5
bm25s>=0.2.14,<1
```

- [ ] **Step 2: Add a synthetic Markdown fixture**

Include headings, emphasis, a link, alt-text image, list, quote, table, inline
code, fenced Python, a long paragraph, and two sections. Use invented text.

- [ ] **Step 3: Write failing tests**

```python
def test_markdown_becomes_plain_structural_units():
    units = MarkdownPassageChunker.for_tests().parse_units(FIXTURE.read_text())
    text = "\n".join(unit.text for unit in units)
    assert "[документация](" not in text
    assert "```" not in text
    assert "def configure_client" in text
    assert "схема компонентов" in text

def test_passages_fit_complete_serialized_token_budget():
    chunker = MarkdownPassageChunker.for_tests(max_tokens=40, reserve_tokens=4)
    passages = chunker.chunk_article(sample_article())
    assert passages
    assert all(row.token_count <= 36 for row in passages)

def test_passage_ids_are_stable_and_source_sensitive():
    chunker = MarkdownPassageChunker.for_tests()
    assert ids(chunker, sample_article()) == ids(chunker, sample_article())
    assert ids(chunker, sample_article()) != ids(chunker, changed_article())
```

- [ ] **Step 4: Verify RED**

Run `conda run -n justatom pytest tests/test_ir_chunking.py -q`.

Expected: import failure because the chunking module does not exist.

- [ ] **Step 5: Implement the minimal parser-backed chunker**

```python
@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    tokenizer_name: str = "intfloat/multilingual-e5-small"
    min_chars: int = 600
    target_chars: int = 1200
    max_chars: int = 1800
    overlap_max_chars: int = 250
    model_max_tokens: int = 512
    safety_reserve_tokens: int = 8

    @property
    def accepted_max_tokens(self) -> int:
        return self.model_max_tokens - self.safety_reserve_tokens

@dataclass(frozen=True, slots=True)
class Passage:
    passage_id: str
    article_id: str
    title: str
    section: str
    content: str
    serialized_passage: str
    char_count: int
    token_count: int
    overlap_prefix_chars: int
    source_hash: str
```

Use `MarkdownIt("commonmark", {"html": False}).enable("table")` plus
`SyntaxTreeNode`. Recursively retain text, code, soft breaks, and image alt text.
Pack units within sections; split oversized prose by sentence and code by line.
Count complete serialized text with `add_special_tokens=True`, no truncation.

- [ ] **Step 6: Verify GREEN and commit**

Run `conda run -n justatom pytest tests/test_ir_chunking.py -q`, then:

```bash
git add requirements.txt tests/fixtures/habr_article.md tests/test_ir_chunking.py justatom/tooling/ir_dataset
git commit -m "feat: add markdown-aware IR passage chunking"
```

---

### Task 2: Authenticated Source and Reproducible Passage Artifacts

**Files:**
- Create: `tests/test_ir_source_and_artifacts.py`
- Create: `justatom/tooling/ir_dataset/source.py`
- Create: `justatom/tooling/ir_dataset/artifacts.py`

**Interfaces:**
- Consumes `MarkdownPassageChunker`.
- Produces `HabrSource.iter_rows`, `PrepareConfig`, `PrepareSummary`, `prepare_passages`.
- Produces ordered `passages.parquet` for sparse/dense indexing.

- [ ] **Step 1: Write failing tests**

```python
def test_habr_source_projects_required_columns_without_comments(tmp_path, monkeypatch):
    source = HabrSource(repo_id="justatom/habr-ds", cache_dir=tmp_path)
    monkeypatch.setattr(source, "_parquet_paths", lambda: [fixture_parquet(tmp_path)])
    row = next(source.iter_rows(limit=1))
    assert set(row) == set(HABR_SOURCE_COLUMNS)
    assert "comments" not in row

def test_prepare_filters_and_writes_ranked_passages(tmp_path):
    summary = prepare_passages(synthetic_rows(), tmp_path, test_chunker(), PrepareConfig(max_passages=20))
    frame = pl.read_parquet(summary.passages_path)
    assert frame["corpus_rank"].to_list() == list(range(frame.height))
    assert frame["passage_id"].n_unique() == frame.height

def test_prepare_reuses_matching_fingerprint(tmp_path):
    first = prepare_synthetic(tmp_path)
    second = prepare_synthetic(tmp_path)
    assert second.reused is True
    assert second.fingerprint == first.fingerprint
```

- [ ] **Step 2: Verify RED**

Run `conda run -n justatom pytest tests/test_ir_source_and_artifacts.py -q`.

- [ ] **Step 3: Implement projected HF parquet access**

```python
HABR_SOURCE_COLUMNS = (
    "id", "language", "url", "title", "text_markdown", "type",
    "time_published", "statistics", "labels", "hubs", "flows", "tags",
    "reading_time", "format", "complexity",
)
```

Use `list_repo_files` and `hf_hub_download` with the existing HF token priority.
Read cached shards using `pl.scan_parquet(paths).select(HABR_SOURCE_COLUMNS)` so
comments and HTML never materialize. Iterate Polars batches.

- [ ] **Step 4: Implement deterministic atomic preparation**

```python
@dataclass(frozen=True, slots=True)
class PrepareConfig:
    seed: int = 42
    max_articles: int | None = None
    max_passages: int | None = None
    max_passages_per_article: int = 8
```

Filter Russian articles with title/body. Normalize topic arrays. Rank by a stable
seeded hash of article and passage IDs, cap passages per article, and assign
contiguous `corpus_rank`. Write temporary Parquet/JSON, validate, then `os.replace`.
Fingerprint source revision/shards plus chunking/preparation config.

- [ ] **Step 5: Verify GREEN and commit**

Run `conda run -n justatom pytest tests/test_ir_source_and_artifacts.py -q`, then:

```bash
git add tests/test_ir_source_and_artifacts.py justatom/tooling/ir_dataset/source.py justatom/tooling/ir_dataset/artifacts.py
git commit -m "feat: prepare reproducible Habr passage artifacts"
```

---

### Task 3: Persistent Russian BM25 Top-k

**Files:**
- Create: `tests/test_ir_sparse_retrieval.py`
- Create: `justatom/tooling/ir_dataset/sparse.py`

**Interfaces:**
- Consumes ordered `(passage_id, serialized_passage)` rows.
- Produces `BM25Index.build`, `BM25Index.load`, `BM25Index.search`.

- [ ] **Step 1: Write failing rank and persistence tests**

```python
def test_bm25_ranks_rare_technical_terms_first(tmp_path):
    index = BM25Index.build(sample_rows(), tmp_path / "bm25")
    hits = index.search(["почему rootless docker не открывает порт"], k=2)[0]
    assert hits[0].passage_id == "p1"
    assert hits[0].score >= hits[1].score

def test_bm25_mmap_reload_preserves_results(tmp_path):
    built = BM25Index.build(sample_rows(), tmp_path / "bm25")
    expected = built.search(["postgresql wal"], k=3)
    loaded = BM25Index.load(tmp_path / "bm25", mmap=True)
    assert loaded.search(["postgresql wal"], k=3) == expected
```

- [ ] **Step 2: Verify RED**

Run `conda run -n justatom pytest tests/test_ir_sparse_retrieval.py -q`.

- [ ] **Step 3: Implement BM25S-backed retrieval**

Implement immutable `SearchHit(passage_id: str, score: float, rank: int)` and
`BM25Index` with these exact public methods: classmethod
`build(rows: Iterable[tuple[str, str]], output_dir: Path) -> BM25Index`,
classmethod `load(output_dir: Path, mmap: bool = True) -> BM25Index`, and
`search(queries: Sequence[str], k: int = 20) -> list[list[SearchHit]]`.

Use a Unicode splitter that preserves technical identifiers. Lowercase without
stemming; preserve product/API names. Use Lucene BM25 with `k1=1.2`, `b=0.75`.
Persist index, tokenizer vocabulary/stopwords, ordered passage IDs, config, and
checksums. Default to `mmap=True` on load.

- [ ] **Step 4: Verify GREEN and commit**

Run `conda run -n justatom pytest tests/test_ir_sparse_retrieval.py -q`, then:

```bash
git add tests/test_ir_sparse_retrieval.py justatom/tooling/ir_dataset/sparse.py
git commit -m "feat: add persistent BM25 passage retrieval"
```

---

### Task 4: Dense E5 Memmap and Exact Blockwise Top-k

**Files:**
- Create: `tests/test_ir_dense_retrieval.py`
- Create: `justatom/tooling/ir_dataset/dense.py`

**Interfaces:**
- Produces `TextEncoder`, `E5TextEncoder`, `DenseIndex.build/load/search_embeddings/search_texts`.

- [ ] **Step 1: Write failing tests using a fake encoder**

```python
def test_dense_build_normalizes_and_persists_embeddings(tmp_path):
    index = DenseIndex.build(sample_rows(), tmp_path / "dense", FakeEncoder(), batch_size=2)
    matrix = index.embedding_rows([0, 1, 2])
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)

def test_blockwise_topk_matches_exact_result(tmp_path):
    index = dense_fixture(tmp_path)
    hits = index.search_embeddings(np.array([[1, 0, 0]], dtype=np.float32), k=2, block_size=2)[0]
    assert [hit.passage_id for hit in hits] == ["p1", "p3"]

def test_excluding_self_still_returns_k_neighbors(tmp_path):
    index = dense_fixture(tmp_path)
    hits = index.search_embeddings(index.embedding_rows([0]), k=2, exclude_ids=["p1"])[0]
    assert len(hits) == 2
    assert all(hit.passage_id != "p1" for hit in hits)
```

- [ ] **Step 2: Verify RED**

Run `conda run -n justatom pytest tests/test_ir_dense_retrieval.py -q`.

- [ ] **Step 3: Implement encoder and index**

```python
class TextEncoder(Protocol):
    dimension: int
    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        raise NotImplementedError

class E5TextEncoder:
    def __init__(self, model_name: str, device: str = "mps", max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
```

Use `AutoTokenizer`/`AutoModel`, attention-mask mean pooling, L2 normalization,
`torch.inference_mode`, and CPU fallback. Persist float32 normalized embeddings,
ordered IDs, model/revision, source fingerprint, dimension, and checksums.

Passage rows already include the `passage:` prefix. `search_texts` must add
`query:` to free-text queries exactly once. Passage-to-passage neighbor creation
uses stored passage embeddings directly and does not re-prefix them.

Search corpus blocks exactly on MPS with a CPU/NumPy fallback; merge block-local candidates with `argpartition`,
then stable-sort by descending score and passage ID. Excluding self must not reduce
the requested result count.

- [ ] **Step 4: Verify GREEN and commit**

Run `conda run -n justatom pytest tests/test_ir_dense_retrieval.py -q`, then:

```bash
git add tests/test_ir_dense_retrieval.py justatom/tooling/ir_dataset/dense.py
git commit -m "feat: add exact dense passage retrieval"
```

---

### Task 5: Resumable CLI and Hybrid Neighbor Diagnostics

**Files:**
- Create: `tests/test_ir_dataset_cli.py`
- Create: `justatom/tooling/ir_dataset/neighbors.py`
- Create: `justatom/api/ir_dataset.py`
- Create: `configs/datasets/habr-ir.yaml`
- Modify: `justatom/tooling/ir_dataset/__init__.py`

**Interfaces:**
- Produces CLI stages `prepare`, `embed`, `neighbors`, `inspect`, `run`.
- Produces `neighbors.parquet` with BM25, dense, and RRF diagnostics.

- [ ] **Step 1: Write failing config/orchestration tests**

```python
def test_checked_in_config_resolves_local_defaults():
    cfg = load_ir_dataset_config("configs/datasets/habr-ir.yaml")
    assert cfg.source.repo_id == "justatom/habr-ds"
    assert cfg.chunking.accepted_max_tokens == 504
    assert cfg.retrieval.bm25_k == 20
    assert cfg.retrieval.dense_k == 20

def test_rrf_union_excludes_self_and_retains_source_ranks():
    rows = merge_neighbors("p1", bm25_hits(), dense_hits(), rrf_k=60, limit=5)
    assert all(row.candidate_id != "p1" for row in rows)
    assert rows[0].bm25_rank is not None or rows[0].dense_rank is not None
```

- [ ] **Step 2: Verify RED**

Run `conda run -n justatom pytest tests/test_ir_dataset_cli.py -q`.

- [ ] **Step 3: Add checked-in config**

```yaml
source:
  repo_id: justatom/habr-ds
  config: default
  split: train
  revision: main
chunking:
  tokenizer_name: intfloat/multilingual-e5-small
  min_chars: 600
  target_chars: 1200
  max_chars: 1800
  overlap_max_chars: 250
  model_max_tokens: 512
  safety_reserve_tokens: 8
preparation:
  seed: 42
  max_articles: null
  max_passages: 100000
  max_passages_per_article: 8
retrieval:
  model_name: intfloat/multilingual-e5-small
  device: mps
  batch_size: 64
  bm25_k: 20
  dense_k: 20
  union_k: 30
  rrf_k: 60
  query_passages: 200
output:
  root: .tmp_runs/datasets/habr-ir/local-100k
```

- [ ] **Step 4: Implement CLI and RRF**

Load YAML into validated dataclasses and apply existing dotted overrides. Load
`.env` without logging secrets. Commands:

```text
prepare   -> passages.parquet + manifest.json
embed     -> build bm25/ and dense/ indices
neighbors -> stable sampled passage-to-passage neighbors.parquet
inspect   -> target and top-k text/scores for passage ID or free query
run       -> prepare, embed, neighbors
```

RRF is `sum(1 / (rrf_k + source_rank))`. Preserve raw ranks/scores, same-article
flag, IDs, and text previews. Stable-sort ties by candidate ID.

- [ ] **Step 5: Verify milestone tests and commit**

Run:

```bash
conda run -n justatom pytest \
  tests/test_ir_chunking.py tests/test_ir_source_and_artifacts.py \
  tests/test_ir_sparse_retrieval.py tests/test_ir_dense_retrieval.py \
  tests/test_ir_dataset_cli.py -q
```

Then:

```bash
git add configs/datasets/habr-ir.yaml tests/test_ir_dataset_cli.py justatom/api/ir_dataset.py justatom/tooling/ir_dataset
git commit -m "feat: add resumable Habr retrieval preparation CLI"
```

---

### Task 6: Regression Suite and Real 100-Article Pilot

**Files:**
- Generate ignored artifacts only under `.tmp_runs/datasets/habr-ir/pilot-100/`.
- Modify milestone code only for a defect proven by a failing test.

- [ ] **Step 1: Run baseline regression**

Run `conda run -n justatom pytest -m "not integration" -q`.

Expected: exit `0`. Record any pre-existing failure before changing code.

- [ ] **Step 2: Run authenticated pilot**

```bash
set -a
source .env
set +a
conda run -n justatom python -m justatom.api.ir_dataset \
  --config configs/datasets/habr-ir.yaml \
  --preparation.max_articles 100 \
  --preparation.max_passages 1000 \
  --retrieval.query_passages 25 \
  --output.root .tmp_runs/datasets/habr-ir/pilot-100 \
  run
```

Expected artifacts: manifest, passages Parquet, BM25 index, dense memmap, and
neighbors Parquet. No token value appears in logs.

- [ ] **Step 3: Inspect pilot invariants and ten neighbor groups**

Programmatically assert:

```text
max token_count <= 504
duplicate passage IDs == 0
non-Russian articles == 0
self-neighbors == 0
distinct flows > 1
BM25 contribution count > 0
dense contribution count > 0
```

Run `inspect --sample 10` and record same-topic, same-article, duplicate-like,
and plausible hard-negative neighbor counts.

- [ ] **Step 4: Fresh final verification**

```bash
conda run -n justatom pytest -m "not integration" -q
git diff --check
git status --short
```

Do not commit `.tmp_runs`. Commit a pilot fix only when a new failing test proves
the defect and the focused/full suites pass afterward.

---

## Deferred Plan

After this milestone is measured, create a separate plan for topic x intent
scheduling, Terra Batch generation, deterministic query gates, Luna collision
validation, bounded retries, 10k pairs, nested 250k/500k/1M corpora, qrels,
human audit, and private Hugging Face publication. That plan must consume these
passage and retrieval contracts instead of reimplementing them.
