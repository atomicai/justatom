# Dissertation Draft and Vestnik Article Revision Design

**Date:** 2026-07-27  
**Status:** approved direction, pending implementation plan  
**Primary language:** Russian

## 1. Objectives

This work has two connected outputs:

1. Turn `phd.paper/main.tex` into a detailed, modular working draft of a
   candidate dissertation in specialty 1.2.1, "Artificial Intelligence and
   Machine Learning" (physical and mathematical sciences).
2. Perform a full scientific revision of the earlier Vestnik manuscript about
   contextual-keyword reranking without conflating that method with the newer
   ATOMIC contrastive-training method.

The dissertation remains an evidence-controlled working draft. Confirmed
results may be stated directly. Unconfirmed experimental claims, missing
replications, and incomplete statistical checks must be visibly marked as
requiring confirmation rather than filled with invented values.

## 2. Dissertation Identity

Working Russian title:

> Модель ATOMIC: адаптивное контрастивное обучение моделей плотного
> информационного поиска с динамическим отбором негативных примеров

Working English title:

> The ATOMIC Model: Adaptive Contrastive Training of Dense Information
> Retrieval Models with Dynamic Negative Selection

Author: Тарлинский Игорь Викторович.  
Organization: Московский государственный университет имени
М. В. Ломоносова, кафедра теоретической информатики.  
Supervisor: кандидат физико-математических наук Главацкий Сергей Тимофеевич.  
Specialty: 1.2.1, "Искусственный интеллект и машинное обучение".  
Degree: кандидат физико-математических наук.

## 3. Dissertation Structure

The dissertation uses a method-first structure:

1. Introduction: relevance, prior work, goal, tasks, object, subject,
   hypothesis, methods, novelty, significance, defense propositions,
   reliability, personal contribution, publications, and structure.
2. Dense information retrieval: sparse and dense retrieval, bi-encoders,
   InfoNCE, in-batch negatives, memory banks, hard-negative mining, false
   negatives, query-conditional learning, and IR evaluation.
3. Mathematical problem statement: the `Q P^T` score matrix, the extended
   `B + K` denominator, unit-hypersphere geometry, concentration effects,
   spherical caps, alignment, uniformity, effective rank, and query
   heterogeneity.
4. ATOMIC: the `vanilla -> atom_gate -> atomic` progression, `alpha(q)`,
   dynamic mixed memory-bank mining, collision statistic `g(q)`, adaptive hard
   admission, learned `m(q)`, weighted log-sum-exp, gradient flow, schedules,
   complexity, and train-only inference behavior.
5. Data and methodology: JustAtom, Meme Russian IR, mMARCO-ru-selected, Habr
   IR, OOD datasets, qrels, leakage prevention, metrics, fixed batch size,
   ablations, seeds, uncertainty, and reproducibility.
6. Experimental study: retrieval results, geometric diagnostics, bank
   telemetry, ablations, negative results, domain transfer, and error analysis.
7. Discussion: dataset-dependent bank behavior, the conditions under which
   query-conditional control helps, geometric interpretation, limitations,
   threats to validity, and applicability.
8. Conclusion and future work.
9. Appendices: configurations, pseudocode, full tables, commands, telemetry,
   and Habr IR construction details.

`tau(q)` is documented as an investigated extension and regime-dependent
result, not silently promoted to the central ATOMIC contribution.

## 4. Dissertation File Layout

The canonical files will be:

- `phd.paper/main.tex`: document preamble and chapter assembly;
- `phd.paper/chapters/*.tex`: one source file per major chapter;
- `phd.paper/references.bib`: verified bibliography;
- `phd.paper/figures/`: dissertation figures;
- `phd.paper/sources/`: source manuscripts and supporting material;
- `phd.paper/main.pdf`: latest successful build.

The inherited Kirill bibliography and appendix are removed. Bibliography is
generated from `references.bib`. The build must resolve citations and
cross-references and must leave the final PDF next to `main.tex`.

## 5. Distinction Between the Two ATOMIC Lines

The Vestnik article describes an earlier retrieval-time method:

- semantic top-p retrieval;
- contextual keyword extraction;
- keyword-aware reranking;
- learned global coefficients `gamma_1` and `gamma_2`.

The dissertation's central ATOMIC method describes training-time adaptation:

- query-conditional `alpha(q)`;
- a memory bank that expands the negative candidate set;
- query-dependent collision control and margin `m(q)`;
- soft weighting in the contrastive denominator.

The article is treated as a preceding stage that motivates query-dependent
control. Its terminology must not imply that its keyword reranker is identical
to the later ATOMIC training algorithm.

## 6. Vestnik Canonical Source

The canonical submission source is
`phd.paper/VestnikPaper/main_for_vestnik.tex`.

`main.tex` and `main_for_pmi_grey.tex` remain reference variants and are not
silently synchronized. The revised PDF is written as
`phd.paper/VestnikPaper/main_for_vestnik.pdf`.

## 7. Vestnik Revision Scope

The revision includes:

1. Reduce oversized figures while preserving legibility.
2. Add appropriate literature citations for the retrieval and generation
   components.
3. Replace Wikipedia links with primary or authoritative scholarly sources.
4. Remove the Watson and open-domain QA references that do not support the
   article's argument.
5. Rewrite the ranking subsection so `phi_s`, `phi_r`, `gamma_1`, and
   `gamma_2` have one consistent meaning in prose, formulas, pseudocode, and
   tables.
6. Use Russian scientific terminology for coefficients, optimization, score,
   pipeline, and related concepts where a stable Russian equivalent exists.
7. Correct the HitRate definition using an indicator function.
8. Replace ambiguous literary examples with neutral, unambiguous retrieval
   examples.
9. Correct table and figure captions and eliminate duplicate labels.
10. Replace the supervisor email with `sergey.glavatsky@math.msu.ru`.
11. Rewrite acknowledgments without template instructions or ambiguous
    attribution.
12. Correct the JustAtom repository reference and use a standard BibTeX entry
    type.
13. Remove duplicate and unused bibliography records.
14. Resolve the GPT-4 Turbo versus GPT-4o inconsistency using only information
    supported by the implementation or experiment artifacts.
15. Qualify the online-complexity claim to include top-p reranking cost.
16. State improvements as absolute percentage-point or relative changes
    explicitly, based on the existing result tables.
17. Avoid claims of statistical significance unless seed-level evidence is
    present.
18. Mark unresolved train/validation/test separation as a validity limitation
    if it cannot be reconstructed from artifacts.

No result table is altered merely to improve the narrative. Any numeric change
must be traceable to an existing experiment artifact or a reproducible
recalculation from the existing tables.

## 8. Bibliography Policy

Bibliographic entries are stored as BibTeX and checked against original
publisher pages, DOI records, or primary preprints. The core bibliography
covers:

- BM25 and probabilistic retrieval;
- DPR and dense passage retrieval;
- E5 embeddings;
- InfoNCE and contrastive representation learning;
- SimCSE and decoupled contrastive learning;
- alignment and uniformity;
- memory-bank and momentum-encoder methods;
- hard-negative mining and false-negative handling;
- HNSW and approximate nearest-neighbor search;
- RAG and retrieval evaluation;
- high-dimensional concentration relevant to dense IR.

Wikipedia may be useful during exploration but is not used as a scholarly
reference in either final document.

## 9. Verification

Implementation is accepted only when:

1. the dissertation and Vestnik sources compile from clean auxiliary state;
2. citations and cross-references resolve;
3. generated PDFs are saved in their requested source directories;
4. all PDF pages are rendered and visually inspected for clipping, overlap,
   unreadable figures, and broken tables;
5. a text scan finds no inherited author, title, bibliography, or appendix from
   the original dissertation template;
6. a text scan finds no Wikipedia links, duplicate labels, template
   acknowledgments, or obsolete supervisor email in the Vestnik source;
7. unsupported claims remain explicitly qualified.

