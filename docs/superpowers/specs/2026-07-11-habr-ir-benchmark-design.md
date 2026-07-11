# Habr Grounded IR Benchmark Design

**Date:** 2026-07-11  
**Status:** Approved for implementation planning  
**Source dataset:** `justatom/habr-ds`  
**Output dataset:** `justatom/habr-ir` (private by default)

## 1. Purpose

Build a reproducible Russian technical information-retrieval dataset from Habr
articles. The first release contains 10,000 grounded query-passage pairs and a
family of nested corpora:

```text
Habr-100k subset Habr-250k subset Habr-500k subset Habr-1M
```

All corpus variants share the same queries, positives, and relevance labels.
Only distractors are added as corpus size grows. This supports two goals:

1. compare `vanilla`, `atom_gate`, and `atomic` training on a recognizable
   Russian technical domain;
2. measure how retrieval quality, neighbor density, hubness, collision rate,
   and memory-bank difficulty change as corpus size increases.

The benchmark targets at least 95% one-target answerability in a stratified
human audit. This is a measured release criterion relative to the frozen
benchmark corpus, not a claim of global uniqueness across the internet.

## 2. Non-goals

- Do not generate questions from comments in v1.
- Do not use whole articles as positive documents.
- Do not use semantic chunk boundaries produced by another embedding model.
- Do not require Weaviate to prepare or validate the dataset.
- Do not force an intent when a passage cannot support it.
- Do not publish the source-derived corpus publicly without a separate license
  and redistribution review.
- Do not change the current training recipes as part of this feature.

## 3. Source contract

Read the private Hugging Face dataset `justatom/habr-ds`, configuration
`default`, split `train`. Use these source fields:

| Field | Use |
| --- | --- |
| `id` | Stable `article_id` |
| `language` | Keep Russian rows only |
| `type` | Keep articles only |
| `url` | Provenance |
| `title` | Passage context and metadata |
| `text_markdown` | Chunking source |
| `flows` | Coarse multi-label topic |
| `hubs` | Fine multi-label topic |
| `tags` | Additional metadata only |
| `statistics` | Selection/reporting metadata |

An eligible article has non-empty `title` and `text_markdown`, is Russian, and
produces at least two valid passages after normalization and chunking.

## 4. Markdown-aware passage construction

### 4.1 Parsing and normalization

Use a declared Markdown parser dependency and its token tree. Do not implement
Markdown parsing with regular expressions.

Convert Markdown into plain, wiki-like text while retaining semantic content:

- headings become plain section titles;
- emphasis markers are removed while text is retained;
- links retain anchor text and drop the URL;
- images retain non-empty descriptive alt text and otherwise disappear;
- lists retain item text without Markdown bullets;
- tables become plain rows with a stable cell separator;
- blockquotes retain their text without quote markers;
- fenced and indented code retain code content but drop fences and language
  markers;
- HTML-only formatting and empty blocks are removed;
- repeated whitespace and blank lines are normalized.

Comments are ignored. A code block remains associated with the section and
paragraphs around it.

### 4.2 Structural units

Parse each article into ordered units:

```text
section heading -> paragraph | list | table | code block
```

Pack adjacent units from the same section. Do not cut a normal paragraph unless
it exceeds the available token budget. Split an oversized prose block by
sentences and an oversized code block by lines.

### 4.3 Length contract

Character counts guide packing:

```yaml
min_chars: 600
target_chars: 1200
max_chars: 1800
overlap_max_chars: 250
```

The tokenizer is the source of truth. Tokenize the final serialized passage
with `intfloat/multilingual-e5-small`, including special tokens:

```text
passage: {article_title}
{section_title}

{plain_passage_text}
```

Length limits:

```yaml
model_max_tokens: 512
safety_reserve_tokens: 8
accepted_max_tokens: 504
```

The chunker dynamically subtracts prefix, title, section, and special-token
costs from the content budget. It must never rely on later tokenizer truncation.

One previous structural unit may overlap with the next passage, capped at 250
characters and by the token budget. A generated evidence span may not live
entirely inside overlap text. Adjacent passages participate in collision
validation, so overlap cannot create two accepted positives for one query.

### 4.4 Passage identity

Create stable identifiers from normalized source identity and boundaries:

```text
passage_id = sha256(article_id, section_path, start_unit, end_unit, text_hash)
```

Each passage stores:

```text
passage_id, article_id, url, title, section, content,
char_count, token_count, overlap_prefix_chars,
flows, hubs, tags, source_hash
```

## 5. Topic-balanced article selection

Use source metadata rather than an LLM-generated topic ontology:

- `flows` are coarse topics;
- `hubs` are fine topics;
- `tags` are retained for analysis but do not define quotas.

Selection is deterministic with seed `42`. Compute coarse-flow quotas from the
square root of eligible article counts, normalize them, and cap any one primary
flow at 30% of selected articles. Within each flow, down-weight frequent hubs by
inverse square-root frequency. Articles without a flow or hub use `other`.

Normalize flow and hub values by trimming whitespace and lowercasing. Define
`primary_flow` as the lexicographically first normalized flow, or `other` when
the list is empty. Define `primary_hub` the same way. Preserve the complete
multi-label lists for reporting and downstream analysis.

Select 5,000 target articles. Each target article must yield two accepted
query-passage pairs from different passages. When an article has valid passages
from at least two sections, the two positives must come from different sections.
If two pairs cannot be accepted after the retry policy, replace the article.

Split target articles before generation:

| Split | Articles | Queries |
| --- | ---: | ---: |
| train | 4,000 | 8,000 |
| dev | 500 | 1,000 |
| test | 500 | 1,000 |

No `article_id` may cross splits.

## 6. Query-intent coverage

Use these target proportions:

| Intent | Share | Meaning |
| --- | ---: | --- |
| `how_to` | 0.25 | Procedure or implementation |
| `why` | 0.15 | Cause or rationale |
| `troubleshooting` | 0.15 | Diagnosis or remediation |
| `concept` | 0.15 | Technical concept in context |
| `comparison` | 0.10 | Difference or trade-off |
| `requirements` | 0.08 | Preconditions or dependencies |
| `limitations` | 0.07 | Boundaries, risks, or failure modes |
| `factual` | 0.05 | Specific reported value or detail |

Balance intents inside each coarse topic rather than only across the complete
dataset. The two queries for one article must use different intents.

The scheduler requests an intent. The generator may reject the request when the
passage does not support it. Rejection is preferable to inventing a cause,
comparison, failure mode, or factual detail.

## 7. One-target query contract

An accepted query must be:

- answerable from the target passage alone;
- self-contained and free of references such as "in the article", "the author",
  "this method", or "the example above";
- scoped by all details required to remove temporal, version, component, mode,
  or scenario ambiguity;
- associated with one stable interpretation and one concise answer;
- supported by an exact evidence substring from the target passage;
- phrased naturally rather than copied from a heading or sentence;
- independent of external knowledge;
- not answerable merely because another passage mentions the same entity.

Necessary product names, APIs, versions, identifiers, and domain terms may be
copied. Long phrasing may not be copied. The deterministic lexical gate rejects
a query with more than eight consecutive normalized content tokens copied from
the target, excluding required identifiers and stop words.

For a frozen corpus `C`, the desired condition is:

```text
A(query, positive) = 1
sum(A(query, passage) for passage in C - {positive}) = 0
```

Here `A` means that a passage independently contains enough information to
answer the query. Automated validation approximates this condition over dense
and lexical nearest candidates; the human audit estimates the residual error.

## 8. Nested corpus construction

All 10,000 positive passages must appear in `Habr-100k`. Fill the remaining
90,000 rows with topic-stratified distractors from non-positive passages. Extend
the corpus deterministically:

```text
100k + 150k distractors = 250k
250k + 250k distractors = 500k
500k + 500k distractors = 1M
```

Assign every passage a stable `corpus_rank`. Logical corpora are prefixes of the
same ranked corpus. Never pad with duplicates. If fewer than one million valid
passages exist, finalization fails clearly instead of silently producing a
smaller `1M` variant.

Distractor selection preserves coarse topic coverage and limits domination by
one article or hub. Passages from target articles are allowed as distractors;
they are especially important collision candidates.

The same query IDs and qrels are used for every corpus size. Each query has one
positive qrel with relevance `1`. If another passage can independently answer
the query, reject or regenerate the query instead of adding a second qrel.

## 9. Candidate retrieval and hard negatives

Preparation must work without Weaviate:

1. encode all corpus passages with normalized `multilingual-e5-small`
   embeddings and store them in a memory-mapped array;
2. retrieve exact dense top-k candidates in blocks on MPS, with a CPU fallback;
3. retrieve lexical candidates with a local sparse index;
4. union and deduplicate dense and lexical candidates;
5. always include adjacent and same-article passages.

Weaviate hybrid retrieval is an optional integration check, not a pipeline
dependency.

Before generation, provide the generator with the target and up to three close
non-target passages. After generation, retrieve candidates using the generated
query and provide the validator with the target and the five strongest distinct
competitors. Keep the larger top-10 candidate IDs and scores in diagnostics.

Passages that are semantically close but judged unable to answer the query are
stored as `hard_negative_ids`. These labels support later analysis of memory-bank
selection, collision risk, and query-conditional margin behavior.

## 10. LLM generation

Use OpenAI Batch API with `/v1/responses`, strict Structured Outputs, and model:

```yaml
model: gpt-5.6-terra
reasoning_effort: low
```

Shard input JSONL before either 1,000 requests or 100 MB, whichever comes first.
Use stable `custom_id` values derived from article, passage, intent, prompt hash,
and attempt number.

### 10.1 Generator system prompt

```text
Ты создаёшь один пример для русскоязычного benchmark по information retrieval.

Дан TARGET PASSAGE и несколько похожих, но нецелевых passages. Сформулируй
естественный вопрос пользователя, для которого TARGET является единственным
самодостаточным источником ответа среди показанных passages.

Требования:
1. Используй только информацию из TARGET. Не добавляй внешние знания.
2. Вопрос должен соответствовать REQUESTED INTENT.
3. Включи в вопрос технологию, компонент, версию, режим, условие или момент,
   если без них возможны разные ответы.
4. Не используй выражения "в статье", "автор", "этот подход", "выше" или
   другие ссылки на невидимый контекст.
5. Вопрос должен иметь одну устойчивую интерпретацию и краткий ответ.
6. Сохрани необходимые технические названия, но не копируй предложение,
   заголовок или длинную фразу из TARGET.
7. EVIDENCE должен быть точной непрерывной цитатой из TARGET.
8. Если TARGET не поддерживает REQUESTED INTENT или вопрос нельзя сделать
   однозначным, верни usable=false. Ничего не выдумывай.
9. Верни только объект, соответствующий JSON schema.
```

### 10.2 Generator user template

```text
LANGUAGE: ru
REQUESTED INTENT: {{requested_intent}}

TARGET PASSAGE ID: {{target_id}}
TARGET:
{{target_text}}

NEARBY NON-TARGET PASSAGES:
{{neighbor_passages}}
```

### 10.3 Generator output contract

All fields are required by the strict schema:

```json
{
  "usable": true,
  "reason": "ok",
  "query": "...",
  "answer": "...",
  "evidence": "...",
  "requested_intent": "how_to",
  "actual_intent": "how_to",
  "disambiguators": ["technology", "condition"]
}
```

When `usable=false`, `query`, `answer`, and `evidence` are empty strings;
`reason` is one of `unsupported_intent`, `insufficient_context`,
`ambiguous_target`, `duplicate_with_neighbor`, or `malformed_source`. Successful
generation uses `reason=ok`. `actual_intent` must equal `requested_intent` when
`usable=true`.

## 11. Validation

### 11.1 Deterministic gates

Reject before an LLM validation call when:

- output does not satisfy the schema;
- `usable=false`;
- query, answer, or evidence is empty;
- evidence is not an exact substring of normalized target content;
- normalized query duplicates an accepted query;
- query is shorter than 5 or longer than 30 whitespace-delimited words;
- query contains a banned context-dependent phrase;
- copied contiguous phrasing exceeds the lexical threshold;
- requested and actual intents differ;
- target passage exceeds its length contract;
- target IDs or source hashes do not match the prepared manifest.

### 11.2 Independent LLM validator

Use OpenAI Batch API, strict Structured Outputs, and:

```yaml
model: gpt-5.6-luna
reasoning_effort: low
```

The validator must not see the generator's free-form rationale.

Validator system prompt:

```text
Ты независимо проверяешь один пример русского information-retrieval benchmark.

Определи, является ли TARGET единственным passage среди TARGET и COMPETITORS,
который самодостаточно отвечает на QUERY.

Проверяй:
1. Ответ и EVIDENCE полностью подтверждаются TARGET.
2. QUERY понятен без статьи, заголовка страницы и внешнего контекста.
3. QUERY имеет одну устойчивую интерпретацию.
4. Ни один COMPETITOR не содержит достаточного ответа на тот же QUERY.
5. QUERY не является длинной копией TARGET.
6. Уточнения в QUERY естественны и действительно устраняют неоднозначность.

Совпадение темы или терминов не делает COMPETITOR ответом. Но если COMPETITOR
позволяет дать тот же содержательный ответ, укажи его ID и отклони пример.
Верни только объект, соответствующий JSON schema.
```

Validator input contains query, proposed answer, exact evidence, target passage,
and five competitors.

Validator output:

```json
{
  "accept": true,
  "grounded": true,
  "self_contained": true,
  "single_answer": true,
  "target_answerable": true,
  "lexical_copy": false,
  "competing_passage_ids": [],
  "reason_codes": ["ok"],
  "confidence": 0.98
}
```

Accept only when every positive boolean is true, `lexical_copy=false`, no
competitor is listed, and confidence is at least `0.90`.

### 11.3 Retry policy

For each candidate article:

1. attempt the scheduled intent once;
2. if ambiguity or a competing answer is found, regenerate once while including
   the identified competitors and requiring an explicit natural disambiguator;
3. if the passage cannot support the intent, schedule one different underfilled
   intent;
4. if the passage still fails, try another passage from the same article;
5. if the article cannot produce two accepted pairs, replace the article.

Never edit or silently accept a failed model output in place.

## 12. Human audit and release criteria

Before publication, draw a deterministic stratified sample of 500 accepted
examples across split, coarse topic, and intent. A human labels:

- target answers the query;
- evidence supports the answer;
- query is self-contained;
- query has one stable interpretation;
- no displayed competitor answers the query;
- wording is natural and not copied.

Have a second annotator independently label a 100-example subset and report
agreement. Publish counts, point estimates, Wilson 95% confidence intervals,
and disagreement resolution rules in the dataset card.

Release requires:

- at least 95% of the 500 audited examples pass all one-target criteria;
- at least 98% pass groundedness;
- zero article leakage across train/dev/test;
- zero missing positives in any nested corpus;
- zero duplicate IDs or exact duplicate normalized queries;
- all automatic validation and reproducibility checks pass.

## 13. Output contracts

### 13.1 Pair rows

```text
pair_id, query_id, article_id, passage_id, split,
query, answer, evidence, positive,
requested_intent, actual_intent, topic_flows, topic_hubs, tags,
difficulty, hard_negative_ids,
generator_model, validator_model,
generator_prompt_hash, validator_prompt_hash,
generation_attempt, validator_confidence
```

`positive` is the exact serialized passage consumed by training and evaluation.
`difficulty` is derived after retrieval from positive rank, score gap, and hard
negative scores; it is not accepted solely from an LLM self-assessment.

### 13.2 Corpus rows

```text
passage_id, article_id, title, section, content, serialized_passage,
url, flows, hubs, tags, char_count, token_count,
corpus_rank, is_positive, source_hash
```

### 13.3 Qrels

```text
query_id, passage_id, relevance
```

Every accepted query has exactly one row with `relevance=1`.

### 13.4 Manifest

Record:

- source dataset, config, split, and revision;
- source schema and row counts at every filter stage;
- random seed and deterministic selection algorithm version;
- chunker version and all character/token limits;
- tokenizer and embedding model revisions;
- generator and validator model IDs returned by the API;
- complete prompt hashes and JSON-schema hashes;
- Batch IDs, request/output checksums, and retry counts;
- git SHA and dirty-worktree flag;
- topic, hub, intent, length, rejection, and collision distributions;
- corpus IDs/checksums and nesting verification;
- human-audit statistics.

Never write API tokens or authorization headers to an artifact.

## 14. Pipeline architecture

Keep the legacy `justatom.api.datasets` behavior compatible. Add a focused
grounded-IR API and small modules with single responsibilities:

```text
justatom/api/ir_dataset.py              CLI and stage orchestration
justatom/tooling/ir_dataset/chunking.py Markdown normalization and packing
justatom/tooling/ir_dataset/selection.py Topics, intents, splits, corpus rank
justatom/tooling/ir_dataset/retrieval.py Embedding and local candidate search
justatom/tooling/ir_dataset/prompts.py   Prompt rendering and strict schemas
justatom/tooling/ir_dataset/batch.py     Batch submit/status/collect
justatom/tooling/ir_dataset/validation.py Deterministic and LLM gates
justatom/tooling/ir_dataset/artifacts.py State, checksums, export, manifest
configs/datasets/habr-ir.yaml           Reproducible experiment config
```

Run from the repository root in the `justatom` Conda environment:

```bash
python -m justatom.api.ir_dataset --config configs/datasets/habr-ir.yaml prepare
python -m justatom.api.ir_dataset --config configs/datasets/habr-ir.yaml embed
python -m justatom.api.ir_dataset --config configs/datasets/habr-ir.yaml generate
python -m justatom.api.ir_dataset --config configs/datasets/habr-ir.yaml collect --stage generation
python -m justatom.api.ir_dataset --config configs/datasets/habr-ir.yaml validate
python -m justatom.api.ir_dataset --config configs/datasets/habr-ir.yaml collect --stage validation
python -m justatom.api.ir_dataset --config configs/datasets/habr-ir.yaml finalize
python -m justatom.api.ir_dataset --config configs/datasets/habr-ir.yaml publish
```

`generate` submits the Terra Batch after local preparation. `validate` applies
deterministic gates, retrieves query-level competitors, and submits the Luna
Batch. Each `collect` invocation is stage-specific and idempotent.

`run` executes every currently possible stage and exits successfully with a
clear message when an asynchronous Batch is pending. Re-running the same command
continues from saved state.

## 15. Artifacts and resumability

Default run root:

```text
.tmp_runs/datasets/habr-ir/<run-id>/
```

Store:

```text
state.json
manifest.json
passages.parquet
embeddings.f32
targets.parquet
batches/*.jsonl
generations.jsonl
validations.jsonl
rejections.jsonl
final/train.parquet
final/dev.parquet
final/test.parquet
final/corpus-100k.parquet
final/corpus-250k.parquet
final/corpus-500k.parquet
final/corpus-1m.parquet
final/qrels.parquet
```

Each stage fingerprints its inputs and config. Write stage output to a temporary
path, fsync it, validate it, and atomically rename it. A completed stage is reused
only when its fingerprint and output checksum match. Changed upstream inputs
invalidate dependent stages with an explicit explanation.

Batch state records request file checksum, OpenAI file ID, Batch ID, status,
output file ID, and collected checksum. Collection is idempotent by `custom_id`.
Malformed, missing, duplicate, or refused responses go to `rejections.jsonl` and
remain eligible for the bounded retry policy.

## 16. Testing strategy

Unit tests run without network, Hugging Face, OpenAI, or Weaviate. Cover:

- Markdown normalization for headings, links, images, tables, lists, quotes,
  inline code, and fenced code;
- paragraph, sentence, and code-line splitting;
- dynamic token budgets and the 504/512-token invariant;
- stable passage IDs and source hashes;
- overlap limits and evidence outside overlap-only text;
- deterministic topic selection and intent balancing;
- article-grouped splits;
- exact nested corpus membership and positive preservation;
- prompt rendering and strict output schemas;
- deterministic evidence, lexical-copy, context-dependence, and duplicate gates;
- validator acceptance logic and retry transitions;
- Batch sharding, `custom_id` stability, response reconciliation, and resume;
- atomic artifact writes, fingerprints, invalidation, and manifest completeness.

Add a checked-in synthetic Markdown fixture and mocked Batch responses. Mark
network/API and Weaviate checks as `integration`.

Before the full run, execute a pilot on 100 articles and 200 accepted pairs. The
pilot must exercise all stages, produce a small nested corpus family, and report
acceptance, rejection, collision, topic, intent, and estimated-cost statistics.

## 17. Resource expectations

A local M5 Max benchmark with `multilingual-e5-small`, 512 tokens, and MPS
measured approximately 207-208 passages per second at batch sizes 32 and 64.
Expected local wall time, including tokenization and indexing, is:

| Corpus | Raw embedding | Practical preparation/index estimate |
| ---: | ---: | ---: |
| 100k | about 8 minutes | 20-45 minutes |
| 250k | about 20 minutes | 40-90 minutes |
| 500k | about 40 minutes | 1-3 hours |
| 1M | about 80 minutes | 2-5 hours |

One million 384-dimensional float32 vectors require about 1.5 GB before index
overhead. A 128 GB machine has ample memory. OpenAI generation and validation
are asynchronous and may dominate wall time; they do not increase with corpus
size because the number of labeled queries remains 10,000.

## 18. Publication and downstream benchmark

Publish privately by default to `justatom/habr-ir`. Include dataset card sections
for source provenance, generation method, prompts, models, automatic filters,
human audit, known limitations, and intended use.

Add a JustAtom dataset preset only after finalization succeeds. The existing
benchmark runner must be able to compare:

```text
vanilla   = InfoNCE
atom_gate = alpha(q), no bank
atomic    = alpha(q) + adaptive bank + m(q)
```

Evaluate the same checkpoints and query set against each nested corpus size.
Report retrieval metrics together with geometry and collision diagnostics. The
dataset-building feature prepares the contracts and artifacts; changes to the
training method remain a separate implementation scope.
