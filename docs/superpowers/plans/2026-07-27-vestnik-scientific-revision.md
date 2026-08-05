# Vestnik Scientific Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a scientifically revised, internally consistent, submission-ready Vestnik manuscript and a clean PDF without changing unsupported experimental results.

**Architecture:** Treat `main_for_vestnik.tex` as the only canonical submission source. Keep figures and result tables as separate assets, replace the oversized bibliography with a focused scholarly bibliography, and validate the manuscript with clean LaTeX/BibTeX builds plus static source checks and rendered-page inspection.

**Tech Stack:** LaTeX, BibTeX, `latexmk`, Poppler (`pdfinfo`, `pdftoppm`, `pdftotext`), shell static checks.

## Global Constraints

- The article describes retrieval-time contextual-keyword reranking, not the dissertation's training-time ATOMIC memory-bank method.
- Numeric result cells may change only when a deterministic recalculation from existing table cells proves the correction.
- Do not claim statistical significance because the manuscript does not provide seed-level estimates.
- Use absolute percentage points for differences between HitRate values; label relative percentages explicitly if any are retained.
- Replace Wikipedia with original papers, publisher pages, or authoritative software/repository records.
- Preserve `main.tex` and `main_for_pmi_grey.tex` as reference variants; edit only the Vestnik source and its shared assets.
- Write the final PDF to `phd.paper/VestnikPaper/main_for_vestnik.pdf`.

---

## File Structure

- Modify `phd.paper/VestnikPaper/main_for_vestnik.tex`: canonical manuscript.
- Modify `phd.paper/VestnikPaper/pmi-bibliography.bib`: focused article bibliography.
- Modify `phd.paper/VestnikPaper/tables/hitrate-small.tex`: precise columns and caption.
- Modify `phd.paper/VestnikPaper/tables/hitrate-base.tex`: precise columns and caption.
- Modify `phd.paper/VestnikPaper/tables/hitrate-large.tex`: precise columns and caption.
- Modify `phd.paper/VestnikPaper/tables/queries-DU.tex`: neutral examples and table caption where needed.
- Modify `phd.paper/VestnikPaper/tables/queries-DV.tex`: neutral examples and table caption where needed.
- Create `phd.paper/VestnikPaper/check_manuscript.sh`: reproducible static and build validation.
- Generate `phd.paper/VestnikPaper/main_for_vestnik.pdf`: latest verified build.

### Task 1: Establish a Reproducible Manuscript Check

**Files:**
- Create: `phd.paper/VestnikPaper/check_manuscript.sh`
- Read: `phd.paper/VestnikPaper/main_for_vestnik.tex`

**Interfaces:**
- Consumes: the canonical `.tex`, shared table files, figures, and `pmi-bibliography.bib`.
- Produces: exit code 0 only when forbidden legacy content is absent and the PDF builds.

- [ ] **Step 1: Add the validation script**

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

forbidden=(
  'en.wikipedia.org'
  'w.wiki'
  'glavatsky\\_st@mail.ru'
  'Introduction to ``{This} is {Watson}'
  'Performance issues and error analysis'
)

for pattern in "${forbidden[@]}"; do
  if rg -n -F "$pattern" main_for_vestnik.tex pmi-bibliography.bib tables; then
    echo "Forbidden legacy content found: $pattern" >&2
    exit 1
  fi
done

duplicate_labels="$(
  rg -o '\\label\{[^}]+\}' main_for_vestnik.tex tables \
    | sed 's/.*\\label{//; s/}//' \
    | sort \
    | uniq -d
)"
if [[ -n "$duplicate_labels" ]]; then
  echo "Duplicate LaTeX labels:" >&2
  echo "$duplicate_labels" >&2
  exit 1
fi

latexmk -C main_for_vestnik.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main_for_vestnik.tex

if rg -n 'undefined references|undefined citations|There were undefined' main_for_vestnik.log; then
  echo "Unresolved references remain" >&2
  exit 1
fi

test -s main_for_vestnik.pdf
pdfinfo main_for_vestnik.pdf | rg '^Pages:'
```

- [ ] **Step 2: Make the script executable**

Run:

```bash
chmod +x phd.paper/VestnikPaper/check_manuscript.sh
```

Expected: `check_manuscript.sh` has executable mode.

- [ ] **Step 3: Run the check and capture the expected initial failure**

Run:

```bash
phd.paper/VestnikPaper/check_manuscript.sh
```

Expected: FAIL on at least one forbidden Wikipedia URL or the obsolete supervisor email.

- [ ] **Step 4: Commit the validation harness**

```bash
git add phd.paper/VestnikPaper/check_manuscript.sh
git commit -m "test: add Vestnik manuscript validation"
```

### Task 2: Replace the Bibliography With Focused Primary Sources

**Files:**
- Modify: `phd.paper/VestnikPaper/pmi-bibliography.bib`
- Modify: `phd.paper/VestnikPaper/main_for_vestnik.tex:106-149`
- Modify: `phd.paper/VestnikPaper/main_for_vestnik.tex:298-324`
- Modify: `phd.paper/VestnikPaper/main_for_vestnik.tex:494-523`

**Interfaces:**
- Consumes: citation keys used by the manuscript.
- Produces: one unique BibTeX record for each cited work and no unused QA-history bulk bibliography.

- [ ] **Step 1: Inventory cited keys**

Run:

```bash
rg -o '\\cite\{[^}]+\}' phd.paper/VestnikPaper/main_for_vestnik.tex \
  phd.paper/VestnikPaper/tables \
  | sed 's/.*\\cite{//; s/}//' \
  | tr ',' '\n' \
  | sed 's/^ *//; s/ *$//' \
  | sort -u
```

Expected: a finite list including the retrieval, embedding, clustering, and repository sources actually used in the text.

- [ ] **Step 2: Rewrite `pmi-bibliography.bib` around verified keys**

Keep or add exactly these conceptual records, using metadata verified against DOI/publisher/primary-paper pages:

```text
robertson2009probabilistic  BM25
karpukhin2020dpr            Dense Passage Retrieval
wang2022e5                  E5
malkov2018hnsw              HNSW
lewis2020rag                Retrieval-Augmented Generation
nogueira2019passage         BERT passage reranking
muennighoff2023mteb         MTEB
ruMTEB                      Russian embedding benchmark, only if discussed
mcinnes2018umap             UMAP
mcinnes2017hdbscan          HDBSCAN implementation/method
campos2020yake              YAKE
weaviate                    Weaviate software
repojustatom                JustAtom repository
```

Use standard BibTeX entry types. In particular:

```bibtex
@misc{repojustatom,
  author       = {Tarlinskiy, Igor},
  title        = {JustAtom: Neural Information Retrieval Toolkit},
  year         = {2026},
  howpublished = {\url{https://github.com/atomicai/justatom}},
  note         = {Software repository, accessed 2026-07-27}
}
```

Do not retain `wiki:*`, `enwiki:hnsw`, `ferrucci2012introduction`, or
`moldovan2003performance`.

- [ ] **Step 3: Replace inline web links with citations**

Use plain terminology in the prose:

```latex
информационный поиск (information retrieval, IR)
```

and cite the relevant primary source at the claim, for example:

```latex
Для разреженного ранжирования используется вероятностная модель BM25
\cite{robertson2009probabilistic}, а плотный поиск реализуется двухбашенным
энкодером \cite{karpukhin2020dpr}.
```

- [ ] **Step 4: Verify bibliography uniqueness**

Run:

```bash
rg -n '^@' phd.paper/VestnikPaper/pmi-bibliography.bib
```

Expected: one entry per retained key and no Wikipedia entries.

- [ ] **Step 5: Commit bibliography cleanup**

```bash
git add phd.paper/VestnikPaper/pmi-bibliography.bib \
  phd.paper/VestnikPaper/main_for_vestnik.tex
git commit -m "docs: focus Vestnik bibliography on primary IR sources"
```

### Task 3: Rewrite the Front Matter, Abstracts, and Problem Statement

**Files:**
- Modify: `phd.paper/VestnikPaper/main_for_vestnik.tex:60-187`

**Interfaces:**
- Consumes: existing method and table values.
- Produces: Russian and English abstracts that state the same scoped contribution.

- [ ] **Step 1: Correct identity and contact information**

Replace the supervisor email with:

```latex
\textit{e-mail: sergey.glavatsky@math.msu.ru}
```

Replace the unrelated UDC line with an IR/AI classification verified for the
target journal, or leave the field visibly marked for editorial confirmation
without asserting the inherited `512;511.823`.

- [ ] **Step 2: Rewrite the Russian abstract**

The abstract must contain, in order:

1. the retrieval problem;
2. the limitation of one global dense/sparse interpolation weight;
3. the proposed two-stage top-p then top-k reranking method;
4. the two datasets and three E5 sizes;
5. the observed deltas without significance claims.

State the existing table evidence exactly:

```text
Compared with dense retrieval, the proposed method improves HitRate by
2.0-7.0 percentage points across the six model/domain cells. Compared with
the best reported hybrid configuration, the gain is 1.0-4.0 percentage
points.
```

- [ ] **Step 3: Rewrite the English abstract as a faithful translation**

Use `percentage points`, not `%`, for the same ranges and use
`contextual-keyword reranking` consistently.

- [ ] **Step 4: Replace the introduction's unsupported dimensionality claim**

Remove the statement that index-size degradation is caused by the curse of
dimensionality. Replace it with the narrower, testable motivation:

```latex
При росте коллекции увеличивается число лексически и семантически близких
кандидатов, поэтому при фиксированном размере выдачи возрастает цена ошибки
первичного ранжирования. В работе исследуется двухэтапная схема, в которой
плотный поиск обеспечивает полноту множества кандидатов, а дополнительная
лексическая функция изменяет их порядок.
```

- [ ] **Step 5: Remove the Watson detour and define system components with citations**

Describe only:

```text
retriever -> optional reranker -> answer generator
```

The article evaluates the retriever/reranker and does not evaluate answer
generation quality.

- [ ] **Step 6: Correct set notation and HitRate**

Use:

```latex
\[
D=\{d_1,\ldots,d_{L_D}\}, \qquad
Q=\{q_1,\ldots,q_{L_Q}\}.
\]

\[
\operatorname{HitRate@}k =
\frac{1}{|Q|}
\sum_{i=1}^{|Q|}
\mathbf{1}\!\left[d_i^+\in R_k(q_i,D)\right].
\]
```

- [ ] **Step 7: Replace ambiguous fictional questions**

Use neutral examples such as:

```text
«Какая команда Git создает новую ветвь и сразу переключается на нее?»
```

and a passage that contains one unambiguous command. Preserve the one-positive
evaluation assumption explicitly.

- [ ] **Step 8: Compile the rewritten front matter**

Run:

```bash
cd phd.paper/VestnikPaper
latexmk -pdf -interaction=nonstopmode -halt-on-error main_for_vestnik.tex
```

Expected: exit 0; no overfull title line caused by the English abstract header.

- [ ] **Step 9: Commit the front-matter rewrite**

```bash
git add phd.paper/VestnikPaper/main_for_vestnik.tex
git commit -m "docs: clarify Vestnik problem statement and claims"
```

### Task 4: Make the Ranking Method Mathematically Consistent

**Files:**
- Modify: `phd.paper/VestnikPaper/main_for_vestnik.tex:288-461`

**Interfaces:**
- Consumes: dense similarity, contextual keyword coverage, and learned global coefficients.
- Produces: one definition of each score used identically in formula, prose, and pseudocode.

- [ ] **Step 1: Normalize score notation**

Define:

```latex
s_{\mathrm{sem}}(q,d)=
\left\langle
\frac{F_\omega(q)}{\|F_\omega(q)\|_2},
\frac{F_\omega(d)}{\|F_\omega(d)\|_2}
\right\rangle
```

and:

```latex
s_{\mathrm{lex}}(q,d)=
\frac{
  \sum_{(k,e)\in K(d)}
  \omega(k,d)\,\mathbf{1}[k\in q]
}{
  \sum_{(k,e)\in K(d)} \omega(k,d)
},
\qquad
\omega(k,d)=\frac{1}{\ln(1+\operatorname{tf}(k,d))}.
```

If explanations `e` are included in matching by the implementation, state the
normalization and match rule explicitly; otherwise do not claim that the
formula uses them.

- [ ] **Step 2: Define the final ranking score**

Use one equation:

```latex
s_\gamma(q,d)=
\gamma_{\mathrm{sem}}s_{\mathrm{sem}}(q,d)+
\gamma_{\mathrm{lex}}s_{\mathrm{lex}}(q,d),
\qquad
\gamma_{\mathrm{sem}},\gamma_{\mathrm{lex}}\geq 0.
```

Call them `весовые коэффициенты`, not an unexplained "гипер-параметр gamma".

- [ ] **Step 3: Correct the coefficient-training loss**

For batch `S`, use:

```latex
\mathcal{L}_{\gamma}(S)=
-\frac{1}{|S|}
\sum_{i\in S}
\log
\frac{\exp(s_\gamma(q_i,d_i^+)/\tau)}
{\sum_{j\in S}\exp(s_\gamma(q_i,d_j^+)/\tau)}.
```

State whether the implementation constrains or parameterizes the two
coefficients. If the code does not guarantee non-negativity, remove the
constraint from the displayed equation rather than inventing it.

- [ ] **Step 4: Explain the two-stage algorithm**

The pseudocode must perform:

```text
1. Encode query.
2. Retrieve top-p candidates by dense similarity.
3. Compute lexical score for each candidate.
4. Compute combined score with learned coefficients.
5. Sort candidates by combined score.
6. Return the first top-k candidates.
```

Use Russian comments and names in the displayed algorithm.

- [ ] **Step 5: State complexity honestly**

State:

```text
ANN retrieval retains its index complexity. The method adds O(p * C_lex)
online reranking work and offline keyword/explanation extraction and storage.
```

Do not claim unchanged online performance without measured latency.

- [ ] **Step 6: Remove duplicate labels and redundant metric subsection**

Give the content-vs-keywords figure and the top-p/top-k figure distinct labels.
Merge repeated prose under one subsection.

- [ ] **Step 7: Compile and inspect method pages**

Run:

```bash
cd phd.paper/VestnikPaper
latexmk -pdf -interaction=nonstopmode -halt-on-error main_for_vestnik.tex
pdftoppm -f 7 -l 11 -png -r 144 main_for_vestnik.pdf /tmp/vestnik-method
```

Expected: exit 0; formulas fit page width; algorithm does not overflow.

- [ ] **Step 8: Commit the method correction**

```bash
git add phd.paper/VestnikPaper/main_for_vestnik.tex
git commit -m "docs: correct contextual reranking formulation"
```

### Task 5: Correct Tables, Figures, Results, and Declarations

**Files:**
- Modify: `phd.paper/VestnikPaper/main_for_vestnik.tex:188-287`
- Modify: `phd.paper/VestnikPaper/main_for_vestnik.tex:463-642`
- Modify: `phd.paper/VestnikPaper/tables/hitrate-small.tex`
- Modify: `phd.paper/VestnikPaper/tables/hitrate-base.tex`
- Modify: `phd.paper/VestnikPaper/tables/hitrate-large.tex`
- Modify: `phd.paper/VestnikPaper/tables/queries-DU.tex`
- Modify: `phd.paper/VestnikPaper/tables/queries-DV.tex`

**Interfaces:**
- Consumes: existing table cells and manuscript figures.
- Produces: readable assets and claims traceable to those cells.

- [ ] **Step 1: Make all HitRate tables explicit**

Each table must identify the cutoff:

```latex
\textbf{Метод} & \textbf{HitRate@2, $D_u$} & \textbf{HitRate@2, $D_v$}
```

If the experiment used a different `k`, use that actual value consistently in
all three captions and in the abstract.

- [ ] **Step 2: Rewrite captions**

Use captions such as:

```latex
\caption{Качество поиска для модели multilingual-E5-small:
HitRate@2 на наборах $D_u$ и $D_v$.}
```

Keep citations in surrounding prose, not inside every caption.

- [ ] **Step 3: Reduce figure sizes**

Use:

```latex
\includegraphics[width=0.72\textwidth,keepaspectratio]{...}
```

for clustering and ranking plots, and at most `0.82\textwidth` for diagrams
whose labels become unreadable at `0.72`.

- [ ] **Step 4: Add a compact delta summary**

Derive, without changing source table values:

```text
R_gamma - R_e: 2.0 to 7.0 percentage points.
R_gamma - R_h: 1.0 to 4.0 percentage points.
```

State that these are descriptive single-run differences.

- [ ] **Step 5: Add validity limitations**

Explicitly state:

- the absence of seed-level confidence intervals;
- whether coefficient tuning and evaluation use disjoint queries;
- the dependence on LLM-generated keyword quality;
- the limited domain and model family coverage.

If data separation cannot be verified, write:

```latex
Разделение данных, использованное для настройки коэффициентов и итоговой
оценки, требует дополнительной верификации по экспериментальным артефактам;
поэтому представленные различия интерпретируются как описательные.
```

- [ ] **Step 6: Correct model-name consistency**

Use the actual extraction model supported by artifacts. If the artifact cannot
distinguish GPT-4 Turbo from GPT-4o, describe it conservatively as an OpenAI
language model and record model identification as a limitation.

- [ ] **Step 7: Rewrite acknowledgments and declarations**

Remove the template footnote and use:

```latex
\section*{Благодарности}
Авторы благодарят Е. М. Крейнес за обсуждение рукописи и рекомендации.
```

Do not thank a listed co-author for co-authoring the same manuscript.

- [ ] **Step 8: Commit result and presentation corrections**

```bash
git add phd.paper/VestnikPaper/main_for_vestnik.tex \
  phd.paper/VestnikPaper/tables
git commit -m "docs: tighten Vestnik evidence and presentation"
```

### Task 6: Produce and Inspect the Submission PDF

**Files:**
- Generate: `phd.paper/VestnikPaper/main_for_vestnik.pdf`
- Verify: `phd.paper/VestnikPaper/main_for_vestnik.log`

**Interfaces:**
- Consumes: all revised article sources.
- Produces: final article PDF and passing validation.

- [ ] **Step 1: Run the complete manuscript check**

Run:

```bash
phd.paper/VestnikPaper/check_manuscript.sh
```

Expected: exit 0 and a non-empty `main_for_vestnik.pdf`.

- [ ] **Step 2: Render every page**

Run:

```bash
rm -rf /tmp/vestnik-final-pages
mkdir -p /tmp/vestnik-final-pages
pdftoppm -png -r 144 \
  phd.paper/VestnikPaper/main_for_vestnik.pdf \
  /tmp/vestnik-final-pages/page
```

Expected: one PNG per PDF page.

- [ ] **Step 3: Visually inspect**

Inspect all rendered pages for:

- clipped title or abstract text;
- unreadable plots;
- equations outside margins;
- algorithm overflow;
- tables split incorrectly;
- unresolved `?` references;
- bibliography line overflow.

- [ ] **Step 4: Run final text checks**

Run:

```bash
pdftotext phd.paper/VestnikPaper/main_for_vestnik.pdf - \
  | rg '\[\?\]|\?\?|Wikipedia|glavatsky_st@mail'
```

Expected: no matches.

- [ ] **Step 5: Commit the final article artifact**

```bash
git add phd.paper/VestnikPaper/main_for_vestnik.tex \
  phd.paper/VestnikPaper/main_for_vestnik.pdf \
  phd.paper/VestnikPaper/pmi-bibliography.bib \
  phd.paper/VestnikPaper/tables \
  phd.paper/VestnikPaper/check_manuscript.sh
git commit -m "docs: finalize revised Vestnik manuscript"
```
