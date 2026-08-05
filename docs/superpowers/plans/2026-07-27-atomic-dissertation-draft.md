# ATOMIC Dissertation Draft Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a detailed, evidence-controlled Russian candidate-dissertation draft about ATOMIC and generate `phd.paper/main.pdf` beside its modular LaTeX sources.

**Architecture:** Replace the inherited single-file template with a thin root document and focused chapter files. Use `biblatex-gost` with a verified shared bibliography, import only traceable claims from repository research notes and benchmark artifacts, and represent unfinished empirical obligations with an explicit evidence marker rather than invented results.

**Tech Stack:** LaTeX `report`, `biblatex-gost`, Biber, `latexmk`, BibTeX data, Poppler rendering and extraction.

## Global Constraints

- Working title: "Модель ATOMIC: адаптивное контрастивное обучение моделей плотного информационного поиска с динамическим отбором негативных примеров".
- Specialty: 1.2.1, "Искусственный интеллект и машинное обучение".
- Degree: кандидат физико-математических наук.
- Central comparison: `vanilla`, `atom_gate`, `atomic`.
- `tau(q)` is an investigated regime-dependent extension, not the central ATOMIC contribution.
- Confirmed and unconfirmed evidence must be visibly distinct.
- Do not state theorem-level conclusions where repository notes provide only an interpretation or approximation.
- Do not reuse Kirill's bibliography, appendix, title, or research text.
- Write the latest successful PDF to `phd.paper/main.pdf`.

---

## File Structure

- Modify `phd.paper/main.tex`: preamble, title page, contents, chapter assembly, bibliography.
- Create `phd.paper/chapters/00-introduction.tex`: dissertation introduction.
- Create `phd.paper/chapters/01-related-work.tex`: dense-IR literature review.
- Create `phd.paper/chapters/02-problem-geometry.tex`: formal problem and geometry.
- Create `phd.paper/chapters/03-atomic-method.tex`: method.
- Create `phd.paper/chapters/04-data-methodology.tex`: datasets and experimental protocol.
- Create `phd.paper/chapters/05-experiments.tex`: results and ablations.
- Create `phd.paper/chapters/06-discussion.tex`: interpretation and limitations.
- Create `phd.paper/chapters/07-conclusion.tex`: conclusions.
- Create `phd.paper/chapters/appendices.tex`: reproducibility appendices.
- Create `phd.paper/references.bib`: verified bibliography.
- Create `phd.paper/sources/README.md`: provenance rules for source manuscripts.
- Create `phd.paper/check_dissertation.sh`: static and build checks.
- Generate `phd.paper/main.pdf`: current draft.

### Task 1: Replace the Inherited Template With a Modular Build

**Files:**
- Modify: `phd.paper/main.tex`
- Create: `phd.paper/chapters/00-introduction.tex`
- Create: `phd.paper/chapters/01-related-work.tex`
- Create: `phd.paper/chapters/02-problem-geometry.tex`
- Create: `phd.paper/chapters/03-atomic-method.tex`
- Create: `phd.paper/chapters/04-data-methodology.tex`
- Create: `phd.paper/chapters/05-experiments.tex`
- Create: `phd.paper/chapters/06-discussion.tex`
- Create: `phd.paper/chapters/07-conclusion.tex`
- Create: `phd.paper/chapters/appendices.tex`
- Create: `phd.paper/sources/README.md`

**Interfaces:**
- Consumes: chapter files and `references.bib`.
- Produces: one compilable dissertation root with explicit evidence markers.

- [ ] **Step 1: Create chapter and source directories**

Run:

```bash
mkdir -p phd.paper/chapters phd.paper/sources phd.paper/figures
```

Expected: all directories exist.

- [ ] **Step 2: Rewrite the preamble and title page**

Use:

```latex
\documentclass[a4paper,14pt,oneside]{extreport}
\usepackage[T2A]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[russian,english]{babel}
\usepackage{amsmath,amssymb,amsfonts,mathtools}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{microtype}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage[
  backend=biber,
  style=gost-numeric,
  sorting=none,
  language=auto,
  autolang=other
]{biblatex}
\addbibresource{references.bib}

\geometry{left=30mm,right=10mm,top=20mm,bottom=20mm}
\linespread{1.3}
\hypersetup{hidelinks}

\newcommand{\evidencepending}[1]{%
  \par\smallskip
  \noindent\textcolor{red!65!black}{%
    \textbf{Требует подтверждения:} #1%
  }%
  \par\smallskip
}
```

The title page must contain the approved title, author, MSU, department,
specialty 1.2.1, degree, supervisor, Moscow, and `20__` until the defense year
is confirmed.

- [ ] **Step 3: Assemble chapters**

The root must contain:

```latex
\tableofcontents

\input{chapters/00-introduction}
\input{chapters/01-related-work}
\input{chapters/02-problem-geometry}
\input{chapters/03-atomic-method}
\input{chapters/04-data-methodology}
\input{chapters/05-experiments}
\input{chapters/06-discussion}
\input{chapters/07-conclusion}

\appendix
\input{chapters/appendices}

\printbibliography[heading=bibintoc,title={Список литературы}]
```

- [ ] **Step 4: Create compiling chapter shells with final section names**

Each chapter file must contain its final `\chapter{...}` and section hierarchy,
not generic provisional headings. For example:

```latex
\chapter{Модель ATOMIC}
\label{ch:atomic}

\section{Общая схема метода}
\section{Query-conditional регуляризация \texorpdfstring{$\alpha(q)$}{alpha(q)}}
\section{Динамический банк негативных примеров}
\section{Диагностика коллизий \texorpdfstring{$g(q)$}{g(q)}}
\section{Адаптивный отбор и обучаемая граница \texorpdfstring{$m(q)$}{m(q)}}
\section{Взвешенный контрастивный знаменатель}
\section{Алгоритм обучения}
\section{Вычислительная сложность}
```

- [ ] **Step 5: Document source provenance**

Write `sources/README.md` with:

```markdown
# Dissertation Sources

Files in this directory are source material, not automatically authoritative.
Before text or numbers are moved into the dissertation:

1. identify the exact source and version;
2. distinguish published claims from working hypotheses;
3. link numeric claims to a benchmark artifact;
4. avoid verbatim reuse when a paraphrase and citation are required;
5. record unresolved conflicts with `\evidencepending{...}`.
```

- [ ] **Step 6: Build the empty modular structure**

Run:

```bash
cd phd.paper
latexmk -C main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: exit 0 and `phd.paper/main.pdf`.

- [ ] **Step 7: Commit the modular scaffold**

```bash
git add phd.paper/main.tex phd.paper/chapters phd.paper/sources/README.md
git commit -m "docs: scaffold modular ATOMIC dissertation"
```

### Task 2: Build the Verified Dissertation Bibliography

**Files:**
- Create: `phd.paper/references.bib`
- Modify: all chapter files as citations are introduced.

**Interfaces:**
- Consumes: publisher/DOI/primary-paper metadata.
- Produces: unique BibTeX keys resolvable by Biber and `biblatex-gost`.

- [ ] **Step 1: Add the dense-IR and contrastive-learning core**

Create verified entries for:

```text
robertson2009bm25       BM25 and probabilistic relevance
karpukhin2020dpr        Dense Passage Retrieval
khattab2020colbert      late-interaction retrieval
wang2022e5              E5 embeddings
muennighoff2023mteb     MTEB
oord2018cpc             InfoNCE/CPC
gao2021simcse           SimCSE
yeh2022dcl              Decoupled Contrastive Learning
wang2020alignment       alignment and uniformity
```

- [ ] **Step 2: Add memory-bank and negative-mining sources**

Create verified entries for:

```text
he2020moco              momentum contrast and queue
xiong2021ance           ANN hard-negative mining for dense retrieval
qu2021rocketqa          cross-batch negatives and denoised hard negatives
chuang2020debiased      false-negative/debiased contrastive learning
malkov2018hnsw          HNSW
```

- [ ] **Step 3: Add geometry and evaluation sources**

Add primary mathematical or archival sources for:

```text
high-dimensional concentration on the sphere
spherical-cap measure
effective rank
anisotropy in sentence embeddings
IR metrics and evaluation practice
```

Do not cite a blog or Wikipedia for a formula.

- [ ] **Step 4: Add software and dataset records**

Include standard entries for JustAtom, Weaviate, Hugging Face datasets used in
the experiments, and the future Habr IR release. The Habr entry must remain
described as an internal benchmark until an immutable public version exists.

- [ ] **Step 5: Validate bibliography**

Run:

```bash
cd phd.paper
latexmk -C main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
rg -n 'WARN.*(not found|undefined)|undefined citation' main.blg main.log
```

Expected: build exit 0 and no unresolved citation match.

- [ ] **Step 6: Commit bibliography**

```bash
git add phd.paper/references.bib phd.paper/chapters phd.paper/main.tex
git commit -m "docs: add verified ATOMIC dissertation bibliography"
```

### Task 3: Write the Dissertation Introduction

**Files:**
- Modify: `phd.paper/chapters/00-introduction.tex`

**Interfaces:**
- Consumes: approved thesis identity and confirmed project evidence.
- Produces: all standard candidate-dissertation introduction elements.

- [ ] **Step 1: Write relevance and prior-work framing**

Explain:

- dense retrieval's role;
- the fixed in-batch denominator;
- query heterogeneity;
- the tension between hard and false negatives;
- why fixed global controls do not transfer uniformly across domains.

Avoid saying memory banks are universally beneficial.

- [ ] **Step 2: State the goal**

Use:

```latex
\textbf{Цель исследования} состоит в разработке и экспериментальном
обосновании метода адаптивного контрастивного обучения моделей плотного
информационного поиска, который при фиксированном размере мини-пакета
динамически расширяет множество негативных примеров и регулирует их вклад с
учетом свойств отдельного запроса.
```

- [ ] **Step 3: State the research tasks**

Include exactly:

1. formalize fixed-batch contrastive retrieval and bank expansion;
2. study high-dimensional query-passage geometry;
3. develop `alpha(q)`-gated auxiliary regularization;
4. develop collision-aware memory-bank mining and `m(q)`;
5. implement ATOMIC reproducibly;
6. construct the cross-domain evaluation protocol including Habr IR;
7. compare `vanilla`, `atom_gate`, and `atomic`;
8. analyze retrieval and geometric diagnostics.

- [ ] **Step 4: Define object, subject, hypothesis, and methods**

Use:

```text
Object: neural dense information-retrieval systems.
Subject: query-conditional control of contrastive training and bank-negative
admission.
Hypothesis: per-query control can preserve useful hard-negative signal while
reducing domain-dependent false-negative damage better than one global margin.
```

- [ ] **Step 5: State novelty conservatively**

Proposed novelty items:

1. unified query-conditional control of auxiliary pressure and bank admission;
2. collision statistic `g(q)=max_j s(q,b_j)-s(q,p^+)`;
3. differentiable bank admission through weighted log-sum-exp;
4. an empirical protocol linking IR quality to hypersphere diagnostics.

Prefix claims with `предлагается` until novelty search and publication review
are complete.

- [ ] **Step 6: Draft defense propositions**

Write propositions as testable statements, not conclusions already assumed:

1. query-conditional auxiliary weighting improves the fixed-batch baseline
   under specified experimental conditions;
2. bank usefulness depends on the distribution of collision risk;
3. collision-aware soft admission can reduce the cross-domain instability of
   fixed hard-negative selection;
4. retrieval changes should be interpreted jointly with alignment,
   uniformity, effective rank, and anisotropy.

- [ ] **Step 7: Add explicit evidence obligations**

Add markers for:

- two-seed or repeated-run confirmation;
- complete mMARCO and Habr evaluations;
- publication and conference presentation counts;
- exact personal-contribution wording agreed with the supervisor.

- [ ] **Step 8: Build and commit**

Run:

```bash
cd phd.paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: exit 0.

```bash
git add phd.paper/chapters/00-introduction.tex phd.paper/main.pdf
git commit -m "docs: draft ATOMIC dissertation introduction"
```

### Task 4: Write the Related-Work Chapter

**Files:**
- Modify: `phd.paper/chapters/01-related-work.tex`

**Interfaces:**
- Consumes: `references.bib`.
- Produces: a critical literature review ending in the research gap.

- [ ] **Step 1: Write sparse and dense retrieval foundations**

Compare inverted-index BM25, bi-encoders, cross-encoders, and late interaction
by quality, index cost, and inference cost.

- [ ] **Step 2: Explain contrastive objectives**

Present InfoNCE:

```latex
\mathcal{L}_{\mathrm{NCE}}=
-\frac{1}{B}\sum_{i=1}^{B}
\log
\frac{\exp(s(q_i,p_i^+)/\tau)}
{\sum_{j=1}^{B}\exp(s(q_i,p_j^+)/\tau)}.
```

Distinguish vanilla InfoNCE from DCL and auxiliary SimCSE.

- [ ] **Step 3: Review negative sampling**

Cover in-batch negatives, mined hard negatives, queues/memory banks,
cross-batch negatives, stale embeddings, and false-negative risk.

- [ ] **Step 4: Review query-conditional controls**

Separate:

- global temperature/margin/weight;
- dataset-level tuning;
- sample/query-dependent functions.

End with the gap: existing controls do not jointly adapt auxiliary
regularization and bank admission to per-query collision risk in this
experimental setting.

- [ ] **Step 5: Build and commit**

```bash
cd phd.paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: exit 0.

```bash
git add phd.paper/chapters/01-related-work.tex phd.paper/main.pdf
git commit -m "docs: write dense retrieval literature review"
```

### Task 5: Write the Formal Problem and Geometry Chapter

**Files:**
- Modify: `phd.paper/chapters/02-problem-geometry.tex`

**Interfaces:**
- Consumes: normalized query/document embeddings and fixed-batch notation.
- Produces: the formal objects used by the ATOMIC chapter.

- [ ] **Step 1: Define the retrieval problem**

Define `Q`, `D`, relevance relation, ranking function, and HR/MRR/NDCG with
indicator-based formulas.

- [ ] **Step 2: Define batch and bank score matrices**

Use:

```latex
S_{\mathrm{batch}}=QP^\top\in\mathbb{R}^{B\times B},
\qquad
S_{\mathrm{bank}}=QM^\top\in\mathbb{R}^{B\times K}.
```

State that the effective candidate denominator has `B + K` columns before
masking/selection, while physical batch size remains `B`.

- [ ] **Step 3: Define normalized hypersphere geometry**

Use `q,p,m in S^{d-1}`, cosine similarity, angular distance, and:

```latex
\|q-p\|_2^2=2(1-\langle q,p\rangle).
```

- [ ] **Step 4: Present spherical-cap interpretation cautiously**

Give the exact cap-measure formula with a primary citation and label the
small-angle proportionality as an asymptotic relation. Do not equate a
training-loss change directly with retrieval improvement.

- [ ] **Step 5: Define diagnostics**

Present alignment, uniformity, effective rank, anisotropy, positive similarity,
negative similarity, and similarity gap. Explain what each can and cannot
prove.

- [ ] **Step 6: Formalize query heterogeneity**

Introduce:

```latex
g(q_i)=\max_{j\in\mathcal{B}_{i}^{\mathrm{valid}}}
s(q_i,b_j)-s(q_i,p_i^+).
```

Interpret `g(q)>0` as a collision-risk signal, not proof that the candidate is
a false negative.

- [ ] **Step 7: Build and commit**

```bash
cd phd.paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: exit 0 and no overfull displayed equations.

```bash
git add phd.paper/chapters/02-problem-geometry.tex phd.paper/main.pdf
git commit -m "docs: formalize ATOMIC retrieval geometry"
```

### Task 6: Write the ATOMIC Method Chapter Against Current Code

**Files:**
- Modify: `phd.paper/chapters/03-atomic-method.tex`
- Read: `justatom/running/trainer.py`
- Read: `justatom/running/encoders.py`
- Read: `justatom/training/memory_bank.py`
- Read: `justatom/training/loss.py`
- Read: `scripts/run_benchmark.sh`

**Interfaces:**
- Consumes: chapter 2 notation and current implementation.
- Produces: an implementation-faithful method specification.

- [ ] **Step 1: Define the three public variants**

Use:

```text
vanilla: InfoNCE fine-tuning without alpha or bank.
atom_gate: query-conditional alpha(q), no bank.
atomic: alpha(q) plus adaptive bank and m(q).
```

- [ ] **Step 2: Define `alpha(q)` training**

Describe the actual current loss form from `trainer.py`, including
train-only behavior, auxiliary term, and whether the deployed encoder uses the
head at inference.

- [ ] **Step 3: Define bank lifecycle**

Document FIFO accumulation, detached stored passage vectors, key-collision
masking, warmup, hard-count ramp, hard/random mixed selection, and the
`B x K` bank score matrix.

- [ ] **Step 4: Define adaptive hard admission**

Use:

```latex
w_{\mathrm{hard}}(q)=
\sigma\!\left(
\frac{t_g-g(q)}{\beta_g}
\right).
```

Explain that it weights selected hard-bank columns and does not change
in-batch negatives.

- [ ] **Step 5: Define learned margin admission**

Use:

```latex
m_{\mathrm{raw}}(q)=
m_0+s_m\tanh(h_m(q)),
\qquad
m(q)=\operatorname{clip}(m_{\mathrm{raw}}(q),m_{\min},m_{\max}),
```

and:

```latex
w_m(q,b)=
\sigma\!\left(
\frac{s(q,p^+)-m(q)-s(q,b)}{\beta_m}
\right).
```

- [ ] **Step 6: Show weighted denominator**

Use:

```latex
Z_i=
\sum_{j\ne i}\exp(\ell_{ij})
+
\sum_{b\in\mathcal{B}_i}
w_{\mathrm{hard}}(q_i,b)\,
w_m(q_i,b)\,
\exp(\ell_{ib}).
```

Explain code equivalence:

```text
logit <- logit + log(weight.clamp_min(eps))
```

and why the gate must be computed on the live query graph in the loss rather
than solely from detached bank vectors.

- [ ] **Step 7: Document regularization and degeneracy**

Describe the observed `m(q)` upper-cap collapse and the raw-margin L2 anchor:

```latex
\mathcal{L}_m=
\lambda_m\mathbb{E}_q
\left(m_{\mathrm{raw}}(q)-m_0\right)^2.
```

Label the current `m(q)` evidence as incomplete where standard deviation or
gradient remains near zero.

- [ ] **Step 8: Add pseudocode and complexity**

State encoder cost, `O(BKd)` bank similarity cost, top-k selection cost, memory
`O(Kd)`, and no bank/head inference cost in the plain encoder evaluation path.

- [ ] **Step 9: Build and commit**

```bash
cd phd.paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: exit 0.

```bash
git add phd.paper/chapters/03-atomic-method.tex phd.paper/main.pdf
git commit -m "docs: write implementation-faithful ATOMIC method"
```

### Task 7: Write Data and Experimental Methodology

**Files:**
- Modify: `phd.paper/chapters/04-data-methodology.tex`
- Read: `docs/research/atom-gate-memory-bank-validation.md`
- Read: `docs/superpowers/specs/2026-07-11-habr-ir-benchmark-design.md`
- Read: benchmark `COMMANDS.md` and result manifests.

**Interfaces:**
- Consumes: dataset contracts and run artifacts.
- Produces: a reproducible, leakage-aware protocol.

- [ ] **Step 1: Describe each dataset separately**

For JustAtom, Meme Russian IR, mMARCO-ru-selected, and Habr IR, report only
counts and splits traceable to manifests. Describe OOD datasets separately from
in-domain adaptation.

- [ ] **Step 2: Describe Habr IR construction**

Cover markdown-aware passage extraction, topic balancing, query-intent
coverage, one-target contract, deterministic gates, LLM validation, human
audit, qrels, nested corpora, and release criteria.

- [ ] **Step 3: Define the controlled comparison**

Fix:

```text
model = intfloat/multilingual-e5-small
batch size = 32
temperature = 0.05
variants = vanilla, atom_gate, atomic
```

Record all other values from `COMMANDS.md` rather than from memory.

- [ ] **Step 4: Define metrics and statistics**

Report HR@1/5/10, MRR@10, NDCG@10 and geometric diagnostics. Require at least
two seeds or repeated runs for confirmatory tables; label single-run tables
exploratory.

- [ ] **Step 5: Define leakage controls**

Require disjoint train/dev/test queries and document passage overlap,
near-duplicate detection, parameter tuning split, baseline cache provenance,
and identical evaluation collection across variants.

- [ ] **Step 6: Build and commit**

```bash
cd phd.paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: exit 0.

```bash
git add phd.paper/chapters/04-data-methodology.tex phd.paper/main.pdf
git commit -m "docs: define ATOMIC evaluation methodology"
```

### Task 8: Write Results, Negative Results, and Discussion

**Files:**
- Modify: `phd.paper/chapters/05-experiments.tex`
- Modify: `phd.paper/chapters/06-discussion.tex`
- Read: `TABLE_RESULTS.md`
- Read: `VERDIKT.md`
- Read: `.tmp_runs/benchmark_runs/*/BENCHMARK_RESULTS.md`
- Read: `.tmp_runs/benchmark_runs/*/GEOMETRY_RESULTS.md`

**Interfaces:**
- Consumes: traceable benchmark tables.
- Produces: exploratory and confirmatory results clearly separated.

- [ ] **Step 1: Add an evidence ledger table**

For every included run, record:

```text
date, run root, datasets, variants, epochs, batch size, bank settings,
evaluation status, and evidence class (exploratory/confirmatory).
```

- [ ] **Step 2: Report alpha-gate evidence**

Include the confirmed in-domain and OOD tables from the report, but qualify
single-seed transfer deltas as descriptive.

- [ ] **Step 3: Report fixed-bank negative results**

Include:

- fixed 8+8 mixed bank;
- 4+12 mixed bank;
- random-only control;
- hard similarity caps;
- dataset-dependent sign changes.

Use these failures to motivate per-query admission without claiming they prove
the mechanism.

- [ ] **Step 4: Report learned-margin behavior**

Include upper-cap collapse, raw-margin anchoring, `MMean/MStd`, `MRaw*`,
regularization, and gradient telemetry. State when the query-conditional
hypothesis was not honestly tested because the head collapsed.

- [ ] **Step 5: Report adaptive-hard results**

For the 2026-06-23 run, report:

```text
justatom:
  atom_gate HR@1 = 0.5159143519
  atomic HR@1 = 0.5355179398
  descriptive delta = +1.9604 percentage points

meme-russian-ir:
  atom_gate HR@1 = 0.7596986214
  atomic HR@1 = 0.7582558512
  descriptive delta = -0.1443 percentage points
```

Record that the historical artifact names the second variant
`atom_gate_bank`, while the public current name is `atomic`.

- [ ] **Step 6: Report failed mMARCO evidence honestly**

The 2026-06-27 `mmarco-ru-selected` run has valid base metrics but failed tuned
evaluation for all three variants. Do not use it as method evidence. Add an
explicit rerun obligation.

- [ ] **Step 7: Connect geometry to retrieval without causal overclaim**

Compare alignment, uniformity, effective rank, anisotropy, sim gap, collision
statistics, hard weights, and margin distributions. Use "associated with" or
"consistent with", not "caused", unless an intervention isolates causality.

- [ ] **Step 8: Write threats to validity**

Cover single-seed noise, dataset size, synthetic questions, evaluation backend,
cache provenance, bank staleness, false-negative labels, model-family scope,
and incomplete Habr/mMARCO confirmation.

- [ ] **Step 9: Build and commit**

```bash
cd phd.paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: exit 0 and all numeric tables fit within margins.

```bash
git add phd.paper/chapters/05-experiments.tex \
  phd.paper/chapters/06-discussion.tex \
  phd.paper/main.pdf
git commit -m "docs: document ATOMIC evidence and limitations"
```

### Task 9: Write Conclusion and Reproducibility Appendices

**Files:**
- Modify: `phd.paper/chapters/07-conclusion.tex`
- Modify: `phd.paper/chapters/appendices.tex`

**Interfaces:**
- Consumes: completed chapters.
- Produces: conclusions bounded by evidence and reproducibility material.

- [ ] **Step 1: Map conclusions to research tasks**

For each task from the introduction, state what was completed, what evidence
supports it, and what remains open.

- [ ] **Step 2: Separate contributions from future work**

Contributions may include the method, differentiable admission, telemetry, and
benchmark protocol. Future work includes multi-seed confirmation, additional
models, larger datasets, and stronger false-negative labels.

- [ ] **Step 3: Add reproducibility appendix**

Include:

- canonical benchmark command;
- configuration table;
- public variant mapping;
- telemetry dictionary;
- artifact layout;
- software/hardware environment.

- [ ] **Step 4: Add supplementary tables appendix**

Move full exploratory tables out of the main narrative while preserving run
roots and settings.

- [ ] **Step 5: Build and commit**

```bash
cd phd.paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: exit 0.

```bash
git add phd.paper/chapters/07-conclusion.tex \
  phd.paper/chapters/appendices.tex \
  phd.paper/main.pdf
git commit -m "docs: complete dissertation conclusions and appendices"
```

### Task 10: Add Final Validation and Inspect the PDF

**Files:**
- Create: `phd.paper/check_dissertation.sh`
- Generate: `phd.paper/main.pdf`

**Interfaces:**
- Consumes: all dissertation sources.
- Produces: a cleanly built and visually inspected draft.

- [ ] **Step 1: Add the dissertation check**

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if rg -n 'Каймаков|Малышев|Нижний Новгород|Манхэттенского расстояния' \
  main.tex chapters references.bib; then
  echo "Inherited template content remains" >&2
  exit 1
fi

duplicate_labels="$(
  rg -o '\\label\{[^}]+\}' main.tex chapters \
    | sed 's/.*\\label{//; s/}//' \
    | sort \
    | uniq -d
)"
if [[ -n "$duplicate_labels" ]]; then
  echo "Duplicate labels:" >&2
  echo "$duplicate_labels" >&2
  exit 1
fi

latexmk -C main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

if rg -n 'undefined references|undefined citations|There were undefined' \
  main.log main.blg; then
  echo "Unresolved references remain" >&2
  exit 1
fi

test -s main.pdf
pdfinfo main.pdf | rg '^Pages:'
```

- [ ] **Step 2: Run validation**

Run:

```bash
chmod +x phd.paper/check_dissertation.sh
phd.paper/check_dissertation.sh
```

Expected: exit 0.

- [ ] **Step 3: Render all pages**

Run:

```bash
rm -rf /tmp/atomic-dissertation-pages
mkdir -p /tmp/atomic-dissertation-pages
pdftoppm -png -r 120 phd.paper/main.pdf /tmp/atomic-dissertation-pages/page
```

Expected: one image per PDF page.

- [ ] **Step 4: Visually inspect**

Check title page, contents, every displayed equation, every table, bibliography,
and chapter transitions. Fix clipping, overlap, unresolved LaTeX references,
broken Cyrillic glyphs, and unreadable figures.

- [ ] **Step 5: Produce an evidence-marker inventory**

Run:

```bash
rg -n '\\evidencepending' phd.paper/chapters
```

Expected: a finite, reviewable list of outstanding research obligations. These
markers are allowed in the working draft and must not disappear without
supporting evidence.

- [ ] **Step 6: Commit final draft**

```bash
git add phd.paper
git commit -m "docs: produce detailed ATOMIC dissertation draft"
```
