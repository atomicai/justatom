# Vestnik Monochrome Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the six active color figures in the Vestnik manuscript with four reproducible black-and-white figures that explain the method and its measured result.

**Architecture:** A single article-local Python entry point parses the existing LaTeX result tables, computes the lexical-score example with the repository's exact tokenization semantics, renders all four figures with one Matplotlib style, and exports PDF, EPS, and true-grayscale PNG files. Pytest protects the parser, lexical-score computation, output contract, and manuscript integration; the existing manuscript checker enforces the four-figure and 15-page journal constraints.

**Tech Stack:** Python 3.12, Matplotlib 3.10.6, Pillow, pytest, LaTeX/latexmk, Poppler, qpdf.

## Global Constraints

- The active manuscript must contain exactly four numbered figures.
- Figures use a white background and black foreground only.
- Semantic distinctions use marker shape, line style, and hatching, never color alone.
- All labels are Russian; mathematical notation matches `main_for_vestnik.tex`.
- Final figure text is at least 8 pt at the official `cmcherald` text width of 170 mm.
- Export each figure as vector PDF, vector EPS, and true-grayscale PNG at 600 dpi.
- Figure 4 reads metrics from the three LaTeX tables; metric constants are not duplicated in drawing code.
- No missing raw curve may be reconstructed from an existing raster image.
- Do not invent confidence intervals when repeated-run measurements are unavailable.
- The manuscript remains at no more than 15 pages.
- The broader migration to `cmcherald` is not part of this change.
- Do not modify or commit unrelated dirty worktree files.

---

## File Map

- Create `phd.paper/VestnikPaper/figures/src/build_monochrome_figures.py`
  - Owns result-table parsing, lexical-score breakdown, shared style, four renderers, and export.
- Create `tests/test_vestnik_figures.py`
  - Tests source-data parsing, agreement with the production lexical helper, output formats, grayscale mode, and active manuscript references.
- Generate `phd.paper/VestnikPaper/figures/method-pipeline-bw.{pdf,eps,png}`
- Generate `phd.paper/VestnikPaper/figures/qlex-git-example-bw.{pdf,eps,png}`
- Generate `phd.paper/VestnikPaper/figures/gamma-training-matrix-bw.{pdf,eps,png}`
- Generate `phd.paper/VestnikPaper/figures/hitrate-comparison-bw.{pdf,eps,png}`
- Modify `phd.paper/VestnikPaper/main_for_vestnik.tex`
  - Removes obsolete figure prose and integrates the four new illustrations.
- Modify `phd.paper/VestnikPaper/check_manuscript.sh`
  - Enforces four active figures, rejects obsolete active assets, and changes the page rule from exactly 12 to 1--15.
- Regenerate `phd.paper/VestnikPaper/main_for_vestnik.pdf`
  - Stores the verified manuscript artifact matching the changed source.

---

### Task 1: Parse Source Metrics and Compute the Git Lexical Example

**Files:**
- Create: `phd.paper/VestnikPaper/figures/src/build_monochrome_figures.py`
- Create: `tests/test_vestnik_figures.py`

**Interfaces:**
- Produces: `ResultRow(model: str, dataset: str, dense: float, hybrid: float, reranker: float)`
- Produces: `TokenContribution(token: str, count: int, weight: float, matched: bool)`
- Produces: `parse_hitrate_table(path: Path) -> dict[str, tuple[float, float]]`
- Produces: `load_result_rows(tables_dir: Path) -> list[ResultRow]`
- Produces: `qlex_breakdown(query: str, passage: str) -> tuple[list[TokenContribution], float]`

- [ ] **Step 1: Write failing parser and lexical-score tests**

Create `tests/test_vestnik_figures.py` with the loader and these tests:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
from justatom.tooling.nlp import keywords_metrics


ROOT = Path(__file__).resolve().parents[1]
ARTICLE_DIR = ROOT / "phd.paper" / "VestnikPaper"
GENERATOR_PATH = (
    ARTICLE_DIR / "figures" / "src" / "build_monochrome_figures.py"
)

spec = importlib.util.spec_from_file_location(
    "vestnik_monochrome_figures", GENERATOR_PATH
)
assert spec is not None and spec.loader is not None
figures = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = figures
spec.loader.exec_module(figures)


def test_load_result_rows_reads_the_three_latex_tables() -> None:
    rows = figures.load_result_rows(ARTICLE_DIR / "tables")

    assert [
        (row.model, row.dataset, row.dense, row.hybrid, row.reranker)
        for row in rows
    ] == [
        ("small", "D_u", 0.53, 0.56, 0.60),
        ("small", "D_v", 0.89, 0.91, 0.93),
        ("base", "D_u", 0.58, 0.61, 0.633),
        ("base", "D_v", 0.90, 0.92, 0.93),
        ("large", "D_u", 0.64, 0.66, 0.681),
        ("large", "D_v", 0.92, 0.93, 0.94),
    ]


def test_qlex_breakdown_matches_the_production_helper() -> None:
    query = (
        "Какая команда Git создает новую ветвь и сразу "
        "переключается на нее?"
    )
    passage = (
        "Команда git switch -c new-branch создает новую ветвь "
        "и сразу переключает рабочую копию на нее."
    )

    contributions, score = figures.qlex_breakdown(query, passage)
    production_score = keywords_metrics._compute_inverse_recall(
        query, passage
    )

    assert score == pytest.approx(production_score)
    assert score == pytest.approx(9 / 11)
    assert [item.token for item in contributions if not item.matched] == [
        "какая",
        "переключается",
    ]
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run:

```bash
conda run -n justatom pytest tests/test_vestnik_figures.py -v
```

Expected: collection fails because
`figures/src/build_monochrome_figures.py` does not exist.

- [ ] **Step 3: Implement immutable data objects and the LaTeX table parser**

Create `phd.paper/VestnikPaper/figures/src/build_monochrome_figures.py` with:

```python
from __future__ import annotations

import math
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_STOPS = "«»\":'" + string.punctuation


@dataclass(frozen=True)
class ResultRow:
    model: str
    dataset: str
    dense: float
    hybrid: float
    reranker: float


@dataclass(frozen=True)
class TokenContribution:
    token: str
    count: int
    weight: float
    matched: bool


def _metric_pair(line: str) -> tuple[float, float]:
    cells = line.split("&")
    if len(cells) < 3:
        raise ValueError(f"Malformed result row: {line!r}")
    values: list[float] = []
    for cell in cells[1:3]:
        cleaned = "".join(
            char for char in cell if char.isdigit() or char == "."
        )
        if not cleaned:
            raise ValueError(f"Missing metric value in row: {line!r}")
        values.append(float(cleaned))
    return values[0], values[1]


def parse_hitrate_table(
    path: Path,
) -> dict[str, tuple[float, float]]:
    text = path.read_text(encoding="utf-8")
    tokens = {
        "dense": r"\(R_e\)",
        "hybrid": r"\(R_h\)",
        "reranker": r"\(R_{\gamma}\)",
    }
    parsed: dict[str, tuple[float, float]] = {}
    for method, token in tokens.items():
        matches = [line for line in text.splitlines() if token in line]
        if len(matches) != 1:
            raise ValueError(
                f"{path}: expected one {method} row, found {len(matches)}"
            )
        parsed[method] = _metric_pair(matches[0])
    return parsed


def load_result_rows(tables_dir: Path) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for model in ("small", "base", "large"):
        parsed = parse_hitrate_table(
            tables_dir / f"hitrate-{model}.tex"
        )
        for index, dataset in enumerate(("D_u", "D_v")):
            rows.append(
                ResultRow(
                    model=model,
                    dataset=dataset,
                    dense=parsed["dense"][index],
                    hybrid=parsed["hybrid"][index],
                    reranker=parsed["reranker"][index],
                )
            )
    return rows
```

- [ ] **Step 4: Implement lexical token contributions with production semantics**

Append:

```python
def _normalize_query(text: str) -> list[str]:
    return (
        "".join(char for char in text if char not in DEFAULT_STOPS)
        .lower()
        .strip()
        .split()
    )


def _passage_counter(text: str) -> Counter[str]:
    return Counter(
        "".join(
            char
            for char in word.lower().strip()
            if char not in DEFAULT_STOPS
        )
        for word in text.split()
    )


def qlex_breakdown(
    query: str,
    passage: str,
) -> tuple[list[TokenContribution], float]:
    query_tokens = _normalize_query(query)
    passage_counts = _passage_counter(passage)
    contributions = [
        TokenContribution(
            token=token,
            count=passage_counts.get(token, 0),
            weight=1.0
            / math.log(1 + passage_counts.get(token, 1)),
            matched=token in passage_counts,
        )
        for token in query_tokens
    ]
    denominator = sum(item.weight for item in contributions)
    numerator = sum(
        item.weight for item in contributions if item.matched
    )
    score = numerator / denominator if denominator else 0.0
    return contributions, score
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
conda run -n justatom pytest tests/test_vestnik_figures.py -v
```

Expected: `2 passed`.

- [ ] **Step 6: Commit the tested data layer**

```bash
git add \
  phd.paper/VestnikPaper/figures/src/build_monochrome_figures.py \
  tests/test_vestnik_figures.py
git commit -m "test: define reproducible Vestnik figure data"
```

---

### Task 2: Render and Export the Four Monochrome Figures

**Files:**
- Modify: `phd.paper/VestnikPaper/figures/src/build_monochrome_figures.py`
- Modify: `tests/test_vestnik_figures.py`
- Generate: `phd.paper/VestnikPaper/figures/*-bw.pdf`
- Generate: `phd.paper/VestnikPaper/figures/*-bw.eps`
- Generate: `phd.paper/VestnikPaper/figures/*-bw.png`

**Interfaces:**
- Consumes: `load_result_rows()` and `qlex_breakdown()` from Task 1.
- Produces: `build_all(output_dir: Path, tables_dir: Path) -> list[Path]`
- Produces exactly twelve assets with these stems:
  `method-pipeline-bw`, `qlex-git-example-bw`,
  `gamma-training-matrix-bw`, `hitrate-comparison-bw`.

- [ ] **Step 1: Add failing output-contract tests**

Append to `tests/test_vestnik_figures.py`:

```python
from PIL import Image


EXPECTED_STEMS = {
    "method-pipeline-bw",
    "qlex-git-example-bw",
    "gamma-training-matrix-bw",
    "hitrate-comparison-bw",
}


def test_build_all_exports_pdf_eps_and_true_grayscale_png(
    tmp_path: Path,
) -> None:
    outputs = figures.build_all(tmp_path, ARTICLE_DIR / "tables")

    assert {path.stem for path in outputs} == EXPECTED_STEMS
    assert {path.suffix for path in outputs} == {".pdf", ".eps", ".png"}
    assert len(outputs) == 12
    for path in outputs:
        assert path.stat().st_size > 1_000
    for png in tmp_path.glob("*.png"):
        with Image.open(png) as image:
            assert image.mode == "L"
            assert image.info.get("dpi", (0, 0))[0] >= 599


def test_every_result_row_improves_over_dense_and_hybrid() -> None:
    for row in figures.load_result_rows(ARTICLE_DIR / "tables"):
        assert row.reranker > row.dense
        assert row.reranker > row.hybrid
```

- [ ] **Step 2: Run tests and verify the missing renderer failure**

Run:

```bash
conda run -n justatom pytest \
  tests/test_vestnik_figures.py::test_build_all_exports_pdf_eps_and_true_grayscale_png \
  -v
```

Expected: FAIL with `AttributeError` for missing `build_all`.

- [ ] **Step 3: Add the shared print style and deterministic exporter**

Add imports and helpers to the generator:

```python
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image


TEXT_WIDTH_IN = 170 / 25.4
QUERY = (
    "Какая команда Git создает новую ветвь и сразу "
    "переключается на нее?"
)
PASSAGE = (
    "Команда git switch -c new-branch создает новую ветвь "
    "и сразу переключает рабочую копию на нее."
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "text.color": "black",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _new_canvas(height: float) -> tuple[Figure, Axes]:
    figure, axis = plt.subplots(
        figsize=(TEXT_WIDTH_IN, height),
        constrained_layout=True,
    )
    axis.set_axis_off()
    return figure, axis


def _box(
    axis: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    linewidth: float = 0.8,
    linestyle: str = "-",
) -> None:
    axis.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            fill=False,
            edgecolor="black",
            linewidth=linewidth,
            linestyle=linestyle,
        )
    )
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
    )


def _arrow(
    axis: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    linestyle: str = "-",
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=8,
            linewidth=0.8,
            linestyle=linestyle,
            color="black",
        )
    )


def _save_figure(figure: Figure, stem: Path) -> list[Path]:
    pdf = stem.with_suffix(".pdf")
    eps = stem.with_suffix(".eps")
    png = stem.with_suffix(".png")
    raster = stem.with_name(stem.name + ".rgba.png")
    figure.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"Creator": "JustAtom Vestnik figure builder"},
    )
    figure.savefig(eps, bbox_inches="tight")
    figure.savefig(raster, dpi=600, bbox_inches="tight")
    with Image.open(raster) as image:
        image.convert("L").save(png, dpi=(600, 600))
    raster.unlink()
    plt.close(figure)
    return [pdf, eps, png]
```

- [ ] **Step 4: Implement Figure 1 and Figure 2 renderers**

Add two functions. Keep all coordinates in normalized axis space so the
rendered result is independent of screen DPI:

```python
def render_pipeline(stem: Path) -> list[Path]:
    figure, axis = _new_canvas(2.55)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.text(0.01, 0.82, "Индексация", weight="bold")
    axis.text(0.01, 0.30, "Поиск", weight="bold")

    offline = [
        (0.13, 0.72, 0.10, 0.16, r"документ $d$"),
        (0.31, 0.72, 0.18, 0.16, r"$F_\theta(d)$, $K(d)$"),
        (0.58, 0.72, 0.15, 0.16, "индекс"),
    ]
    online = [
        (0.05, 0.18, 0.10, 0.16, r"запрос $q$"),
        (0.22, 0.18, 0.16, 0.16, r"ANN по $s_0$"),
        (0.45, 0.18, 0.12, 0.16, r"$C_p$"),
        (0.64, 0.18, 0.18, 0.16, r"$s_\gamma$"),
        (0.88, 0.18, 0.10, 0.16, r"$top_k$"),
    ]
    for x, y, width, height, text in offline + online:
        _box(axis, x, y, width, height, text)
    for row in (offline, online):
        for left, right in zip(row, row[1:]):
            _arrow(
                axis,
                (left[0] + left[2], left[1] + left[3] / 2),
                (right[0], right[1] + right[3] / 2),
            )
    axis.text(
        0.70,
        0.48,
        r"$s_\gamma=\gamma_0s_0+\gamma_{\mathrm{qlex}}"
        r"s_{\mathrm{qlex}}$",
        ha="center",
    )
    _arrow(axis, (0.65, 0.72), (0.70, 0.34), linestyle="--")
    axis.text(0.58, 0.50, r"$K(d)$", ha="center")
    return _save_figure(figure, stem)


def render_qlex_example(stem: Path) -> list[Path]:
    contributions, score = qlex_breakdown(QUERY, PASSAGE)
    figure, axis = _new_canvas(3.05)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.text(0.02, 0.92, r"$q$:", weight="bold")
    axis.text(0.07, 0.92, QUERY, va="top", wrap=True)
    axis.text(0.02, 0.78, r"$d$:", weight="bold")
    axis.text(0.07, 0.78, PASSAGE, va="top", wrap=True)

    labels = [item.token for item in contributions]
    matches = ["+" if item.matched else "−" for item in contributions]
    counts = [str(item.count) for item in contributions]
    table = axis.table(
        cellText=[labels, matches, counts],
        rowLabels=["токен", "в числителе", r"$c_X$"],
        bbox=[0.08, 0.30, 0.88, 0.31],
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.6)
    for cell in table.get_celld().values():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.5)
        cell.set_facecolor("white")

    numerator = sum(
        item.weight for item in contributions if item.matched
    )
    denominator = sum(item.weight for item in contributions)
    axis.text(
        0.50,
        0.13,
        rf"$s_{{\mathrm{{qlex}}}}="
        rf"\frac{{{numerator:.3f}}}{{{denominator:.3f}}}"
        rf"={score:.3f}$",
        ha="center",
        fontsize=9,
    )
    return _save_figure(figure, stem)
```

- [ ] **Step 5: Implement Figure 3 and Figure 4 renderers**

Add:

```python
def render_gamma_training(stem: Path) -> list[Path]:
    figure, axis = _new_canvas(2.75)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    boxes = [
        (0.03, 0.60, 0.14, 0.18, r"$S_0\in\mathbb{R}^{N\times N}$", "-"),
        (
            0.03,
            0.24,
            0.14,
            0.18,
            r"$S_{\mathrm{qlex}}\in\mathbb{R}^{N\times N}$",
            "--",
        ),
        (
            0.29,
            0.42,
            0.24,
            0.22,
            r"$S_\gamma=\gamma_0S_0+"
            r"\gamma_{\mathrm{qlex}}S_{\mathrm{qlex}}$",
            "-",
        ),
        (0.66, 0.42, 0.14, 0.22, "cross-entropy\nпо строкам", "-"),
        (0.87, 0.42, 0.11, 0.22, r"$\nabla_\gamma\mathcal{L}$", "-"),
    ]
    for x, y, width, height, text, linestyle in boxes:
        _box(
            axis,
            x,
            y,
            width,
            height,
            text,
            linestyle=linestyle,
        )
    _arrow(axis, (0.17, 0.69), (0.29, 0.55))
    _arrow(axis, (0.17, 0.33), (0.29, 0.51), linestyle="--")
    _arrow(axis, (0.53, 0.53), (0.66, 0.53))
    _arrow(axis, (0.80, 0.53), (0.87, 0.53))

    for row in range(4):
        for column in range(4):
            axis.add_patch(
                Rectangle(
                    (0.555 + column * 0.019, 0.36 + row * 0.035),
                    0.019,
                    0.035,
                    facecolor="black"
                    if row == 3 - column
                    else "white",
                    edgecolor="black",
                    linewidth=0.35,
                )
            )
    axis.text(
        0.593,
        0.31,
        "диагональ: положительные пары",
        ha="center",
        fontsize=7,
    )
    return _save_figure(figure, stem)


def render_hitrate(stem: Path, tables_dir: Path) -> list[Path]:
    rows = load_result_rows(tables_dir)
    figure, axis = plt.subplots(
        figsize=(TEXT_WIDTH_IN, 3.15),
        constrained_layout=True,
    )
    positions = list(range(len(rows) - 1, -1, -1))
    labels = [
        rf"{row.model}, ${row.dataset}$"
        for row in rows
    ]
    for y, row in zip(positions, rows):
        axis.plot(
            [row.dense, row.reranker],
            [y, y],
            color="black",
            linewidth=0.55,
            zorder=1,
        )
    axis.scatter(
        [row.dense for row in rows],
        positions,
        marker="o",
        facecolors="white",
        edgecolors="black",
        label=r"$R_e$",
        zorder=2,
    )
    axis.scatter(
        [row.hybrid for row in rows],
        positions,
        marker="^",
        facecolors="white",
        edgecolors="black",
        label=r"$R_h$",
        zorder=2,
    )
    axis.scatter(
        [row.reranker for row in rows],
        positions,
        marker="s",
        color="black",
        label=r"$R_\gamma$",
        zorder=3,
    )
    axis.set_yticks(positions, labels)
    axis.set_xlim(0.50, 0.95)
    axis.set_xlabel("HitRate@2")
    axis.grid(axis="x", color="black", linewidth=0.35, linestyle=":")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    return _save_figure(figure, stem)
```

- [ ] **Step 6: Add the single entry point**

Append:

```python
FIGURE_STEMS = (
    "method-pipeline-bw",
    "qlex-git-example-bw",
    "gamma-training-matrix-bw",
    "hitrate-comparison-bw",
)


def build_all(output_dir: Path, tables_dir: Path) -> list[Path]:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    outputs.extend(
        render_pipeline(output_dir / FIGURE_STEMS[0])
    )
    outputs.extend(
        render_qlex_example(output_dir / FIGURE_STEMS[1])
    )
    outputs.extend(
        render_gamma_training(output_dir / FIGURE_STEMS[2])
    )
    outputs.extend(
        render_hitrate(output_dir / FIGURE_STEMS[3], tables_dir)
    )
    return outputs


def main() -> None:
    article_dir = Path(__file__).resolve().parents[2]
    build_all(article_dir / "figures", article_dir / "tables")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the full figure test module**

Run:

```bash
conda run -n justatom pytest tests/test_vestnik_figures.py -v
```

Expected: `4 passed`.

- [ ] **Step 8: Generate repository assets**

Run:

```bash
SOURCE_DATE_EPOCH=1767225600 \
conda run -n justatom python \
  phd.paper/VestnikPaper/figures/src/build_monochrome_figures.py
```

Expected: twelve non-empty files under
`phd.paper/VestnikPaper/figures/`, four per format.

- [ ] **Step 9: Render the vector outputs and inspect all four figures**

Run:

```bash
mkdir -p tmp/pdfs/vestnik-monochrome
for pdf in phd.paper/VestnikPaper/figures/*-bw.pdf; do
  name="$(basename "$pdf" .pdf)"
  pdftoppm -png -r 150 -singlefile \
    "$pdf" "tmp/pdfs/vestnik-monochrome/$name"
done
```

Inspect:

```bash
open tmp/pdfs/vestnik-monochrome
```

Expected: no clipped text; all marks remain distinguishable without color;
the Git token table and six result rows are readable at 100%.

- [ ] **Step 10: Commit the renderer and generated assets**

```bash
git add \
  phd.paper/VestnikPaper/figures/src/build_monochrome_figures.py \
  phd.paper/VestnikPaper/figures/method-pipeline-bw.* \
  phd.paper/VestnikPaper/figures/qlex-git-example-bw.* \
  phd.paper/VestnikPaper/figures/gamma-training-matrix-bw.* \
  phd.paper/VestnikPaper/figures/hitrate-comparison-bw.* \
  tests/test_vestnik_figures.py
git commit -m "docs: add monochrome Vestnik figures"
```

---

### Task 3: Integrate the Figures and Enforce Journal Constraints

**Files:**
- Modify: `tests/test_vestnik_figures.py`
- Modify: `phd.paper/VestnikPaper/main_for_vestnik.tex`
- Modify: `phd.paper/VestnikPaper/check_manuscript.sh`
- Regenerate: `phd.paper/VestnikPaper/main_for_vestnik.pdf`

**Interfaces:**
- Consumes: the four PDF figure assets from Task 2.
- Produces: an active manuscript with exactly four figure references.
- Produces: `check_manuscript.sh` output containing
  `WORKTREE_PDF_PAGES=<1..15>` and no figure-compliance error.

- [ ] **Step 1: Add a failing manuscript integration test**

Append:

```python
def _active_tex(source: str) -> str:
    active_lines: list[str] = []
    for line in source.splitlines():
        visible: list[str] = []
        escaped = False
        for char in line:
            if char == "%" and not escaped:
                break
            visible.append(char)
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
        active_lines.append("".join(visible))
    return "\n".join(active_lines)


def test_manuscript_references_exactly_four_new_figures() -> None:
    active = _active_tex(
        (ARTICLE_DIR / "main_for_vestnik.tex").read_text(
            encoding="utf-8"
        )
    )
    assert active.count(r"\includegraphics") == 4
    for stem in EXPECTED_STEMS:
        assert f"figures/{stem}.pdf" in active
    for obsolete in (
        "clustering-e5-base.png",
        "clustering_model=[e5]_dataset=[universe].png",
        "sample-hunger-games.png",
        "Loss-BiGamma-vs-Gamma.png",
        "idf-recall-content-vs-keywords.png",
        "top-k-top-p.png",
    ):
        assert obsolete not in active
```

- [ ] **Step 2: Run the integration test and verify it fails**

Run:

```bash
conda run -n justatom pytest \
  tests/test_vestnik_figures.py::test_manuscript_references_exactly_four_new_figures \
  -v
```

Expected: FAIL because the active source still contains six old figures.

- [ ] **Step 3: Remove the two cluster figures and their unsupported visual narrative**

In `main_for_vestnik.tex`, keep the dataset counts and lexical statistics
tables, but remove both UMAP/HDBSCAN paragraphs and figure environments.
Replace the transition after Table `tab:tokensdist` with:

```tex
Различия между наборами проявляются в длине текстов, доле
лексического пересечения и характере предметной лексики. Эти
характеристики мотивируют использование второго этапа, который
учитывает совпадения значимых токенов внутри кандидатов,
предварительно найденных плотным поиском.
```

- [ ] **Step 4: Insert Figure 1 at the start of the reranking subsection**

Immediately after the first paragraph of
`\subsection{Контекстно-ключевое повторное ранжирование}`, add:

```tex
\begin{figure}[ht]
    \centering
    \includegraphics[width=\textwidth,keepaspectratio]
        {figures/method-pipeline-bw.pdf}
    \caption{Двухэтапная схема поиска: индексирование представлений
    документов и ключевых выражений, выбор множества кандидатов
    $C_p$ и повторное ранжирование по $s_{\gamma}$.}
    \label{fig:method-pipeline}
\end{figure}
```

Remove the old `top-k-top-p.png` environment and the `\clearpage` immediately
before it, then rewrite its following sentence to reference
`fig:method-pipeline`.

- [ ] **Step 5: Replace the Hunger Games illustration with Figure 2**

Replace its introductory paragraph and figure with:

```tex
Покомпонентное вычисление оценки для технического примера из
постановки показано на рис.~\ref{fig:qlex-git-example}.
Знак ``$+$'' означает, что нормализованный токен запроса найден
в параграфе; несовпавшие позиции участвуют только в знаменателе.

\begin{figure}[ht]
    \centering
    \includegraphics[width=\textwidth,keepaspectratio]
        {figures/qlex-git-example-bw.pdf}
    \caption{Вычисление $s_{\mathrm{qlex}}$ для запроса о создании
    ветви Git: совпавшие позиции входят в числитель, а все позиции
    запроса --- в знаменатель оценки.}
    \label{fig:qlex-git-example}
\end{figure}
```

- [ ] **Step 6: Replace the empirical loss raster with Figure 3**

After Equation `eq:gamma-loss`, replace the old loss-comparison prose and
figure with:

```tex
Вычисление функции потерь в матричной форме показано на
рис.~\ref{fig:gamma-training-matrix}. Матрицы $S_0$ и
$S_{\mathrm{qlex}}$ строятся для всех пар запросов и документов
батча. Положительные пары расположены на диагонали
$S_{\gamma}$, а градиент cross-entropy обновляет
$\gamma_0$ и $\gamma_{\mathrm{qlex}}$.

\begin{figure}[ht]
    \centering
    \includegraphics[width=\textwidth,keepaspectratio]
        {figures/gamma-training-matrix-bw.pdf}
    \caption{Формирование матрицы $S_{\gamma}$ и обучение
    коэффициентов $\gamma_0$, $\gamma_{\mathrm{qlex}}$ по
    диагональным положительным парам батча.}
    \label{fig:gamma-training-matrix}
\end{figure}
```

- [ ] **Step 7: Remove the content-versus-keywords trace**

Delete the `idf-recall-content-vs-keywords.png` environment. Retain the
opening sentence of the subsection as prose and replace the figure
reference with:

```tex
В качестве $X(d)$ можно выбрать токены ключевых слов, их
объяснений, полного текста или настроенной комбинации полей.
В проведённых экспериментах выбор поля задавался одинаково для
всех сравниваемых методов.
```

- [ ] **Step 8: Add Figure 4 after the three result tables**

After the three `\input{tables/hitrate-*.tex}` statements, add:

```tex
\begin{figure}[ht]
    \centering
    \includegraphics[width=\textwidth,keepaspectratio]
        {figures/hitrate-comparison-bw.pdf}
    \caption{HitRate@2 плотного поиска $R_e$, гибридного поиска
    $R_h$ и предложенного повторного ранжирования $R_{\gamma}$
    для моделей multilingual-E5 трёх размеров на $D_u$ и $D_v$.}
    \label{fig:hitrate-comparison}
\end{figure}
```

Add this sentence before the numerical difference paragraph:

```tex
Взаимное расположение результатов в шести конфигурациях
показано на рис.~\ref{fig:hitrate-comparison}.
```

- [ ] **Step 9: Run the integration test**

Run:

```bash
conda run -n justatom pytest tests/test_vestnik_figures.py -v
```

Expected: all tests pass.

- [ ] **Step 10: Extend the manuscript checker**

In `static_source_checks()`, after `visible_source` is assigned, add:

```bash
  local figure_count obsolete_figure
  figure_count="$(
    printf '%s\n' "$visible_source" |
      {
        rg -o -F '\includegraphics' || true
      } |
      wc -l |
      tr -d ' '
  )"
  [[ "$figure_count" -eq 4 ]] ||
    fail "Vestnik manuscript must contain exactly four active figures"

  for obsolete_figure in \
    clustering-e5-base.png \
    'clustering_model=[e5]_dataset=[universe].png' \
    sample-hunger-games.png \
    Loss-BiGamma-vs-Gamma.png \
    idf-recall-content-vs-keywords.png \
    top-k-top-p.png
  do
    if printf '%s\n' "$visible_source" | rg -F -q "$obsolete_figure"; then
      fail "obsolete active figure remains: $obsolete_figure"
    fi
  done
```

In `validate_pdf()`, replace the exact-page assertion with:

```bash
  [[ "$pages" -ge 1 && "$pages" -le 15 ]] ||
    fail "$label PDF must contain between 1 and 15 pages"
```

- [ ] **Step 11: Build the manuscript and inspect every page containing a figure**

Run:

```bash
cd phd.paper/VestnikPaper
SOURCE_DATE_EPOCH=1767225600 FORCE_SOURCE_DATE=1 TZ=UTC \
  latexmk -pdf -interaction=nonstopmode -halt-on-error \
  main_for_vestnik.tex
cd ../../
mkdir -p tmp/pdfs/vestnik-manuscript-bw
pdftoppm -png -r 150 \
  phd.paper/VestnikPaper/main_for_vestnik.pdf \
  tmp/pdfs/vestnik-manuscript-bw/page
open tmp/pdfs/vestnik-manuscript-bw
```

Expected:

- at most 15 pages;
- figures appear in order 1--4;
- no figure text is clipped or smaller than surrounding footnote text;
- the three result markers remain distinct;
- no blank page or excessive vertical gap is introduced.

- [ ] **Step 12: Stage only article-related files and run the full checker**

Stage the exact scope before running the checker because it compares the
working PDF with the Git index and a fresh archive:

```bash
git add \
  phd.paper/VestnikPaper/main_for_vestnik.tex \
  phd.paper/VestnikPaper/main_for_vestnik.pdf \
  phd.paper/VestnikPaper/check_manuscript.sh \
  tests/test_vestnik_figures.py
phd.paper/VestnikPaper/check_manuscript.sh
```

Expected final line: `VESTNIK_CHECK=PASS`.

- [ ] **Step 13: Verify no unrelated path was staged**

Run:

```bash
git diff --cached --name-only
```

Expected paths are limited to:

```text
phd.paper/VestnikPaper/check_manuscript.sh
phd.paper/VestnikPaper/main_for_vestnik.pdf
phd.paper/VestnikPaper/main_for_vestnik.tex
tests/test_vestnik_figures.py
```

- [ ] **Step 14: Commit manuscript integration**

```bash
git commit -m "docs: integrate monochrome Vestnik figures"
```

---

### Task 4: Final Reproducibility and Scientific-Consistency Audit

**Files:**
- Verify only; no planned modifications.

**Interfaces:**
- Consumes: all commits from Tasks 1--3.
- Produces: a clean verification report in command output and a submission-ready four-figure manuscript artifact.

- [ ] **Step 1: Re-run focused tests from the committed tree**

```bash
conda run -n justatom pytest tests/test_vestnik_figures.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Rebuild figures twice and compare content hashes**

```bash
SOURCE_DATE_EPOCH=1767225600 \
conda run -n justatom python \
  phd.paper/VestnikPaper/figures/src/build_monochrome_figures.py
find phd.paper/VestnikPaper/figures -maxdepth 1 \
  -type f -name '*-bw.*' -print0 |
  sort -z |
  xargs -0 shasum -a 256 > /tmp/vestnik-figures-first.sha256

SOURCE_DATE_EPOCH=1767225600 \
conda run -n justatom python \
  phd.paper/VestnikPaper/figures/src/build_monochrome_figures.py
find phd.paper/VestnikPaper/figures -maxdepth 1 \
  -type f -name '*-bw.*' -print0 |
  sort -z |
  xargs -0 shasum -a 256 > /tmp/vestnik-figures-second.sha256

diff -u \
  /tmp/vestnik-figures-first.sha256 \
  /tmp/vestnik-figures-second.sha256
```

Expected: no diff. If EPS metadata prevents byte-identical hashes, inspect the
only differing lines; remove volatile EPS creation metadata in the exporter
rather than weakening the check.

- [ ] **Step 3: Run the manuscript checker once more**

```bash
phd.paper/VestnikPaper/check_manuscript.sh
```

Expected: `VESTNIK_CHECK=PASS`.

- [ ] **Step 4: Confirm repository scope**

```bash
git status --short
git log -4 --oneline
```

Expected: pre-existing unrelated dirty files may remain, but no uncommitted
changes remain in the files listed in this plan. The latest commits cover the
figure data layer, generated assets, and manuscript integration.
