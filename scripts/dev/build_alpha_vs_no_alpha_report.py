from __future__ import annotations

import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from justatom.tooling.dataset import DatasetRecordAdapter

OUTPUT_DIR = REPO_ROOT / "artifacts" / "alpha_vs_no_alpha_report"
REPORT_PATH = REPO_ROOT / "alpha_vs_no_alpha_report.md"

METRIC_KEYS = ["HR@1", "HR@5", "HR@10", "MRR@10", "NDCG@10"]
TOKEN_RE = re.compile(r"\w+", re.UNICODE)

sns.set_theme(style="whitegrid", context="talk")


@dataclass(frozen=True)
class MetricsSpec:
    baseline: dict[str, float]
    no_alpha: dict[str, float]
    alpha: dict[str, float]
    source_report: str
    scope_label: str
    alpha_label: str
    no_alpha_label: str
    note: str


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    dataset_name_or_path: str
    content_field: str
    labels_field: str
    chunk_id_col: str | None = None
    split: str | None = None
    stats_limit: int | None = None
    stats_scope_label: str = "full dataset"
    metrics: MetricsSpec | None = None


DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        dataset_id="justatom",
        dataset_name_or_path="justatom",
        content_field="content",
        labels_field="queries",
        chunk_id_col="chunk_id",
        stats_scope_label="full repo-local dataset",
        metrics=MetricsSpec(
            baseline={"HR@1": 0.3954, "HR@5": 0.6502, "HR@10": 0.7359, "MRR@10": 0.5053, "NDCG@10": 0.5607},
            no_alpha={"HR@1": 0.5840, "HR@5": 0.8353, "HR@10": 0.8981, "MRR@10": 0.6908, "NDCG@10": 0.7411},
            alpha={"HR@1": 0.5861, "HR@5": 0.8384, "HR@10": 0.9000, "MRR@10": 0.6931, "NDCG@10": 0.7434},
            source_report="justatom.md",
            scope_label="full dataset",
            alpha_label="best alpha sweep t=0.025, mix=0.1",
            no_alpha_label="contrastive t=0.03, alpha off",
            note="Alpha wins on all reported retrieval metrics.",
        ),
    ),
    DatasetSpec(
        dataset_id="electrical-engineering-ru",
        dataset_name_or_path="hf://d0rj/Electrical-engineering-ru",
        content_field="output",
        labels_field="input",
        split="train",
        stats_scope_label="full train split",
        metrics=MetricsSpec(
            baseline={"HR@1": 0.8276, "HR@5": 0.9655, "HR@10": 0.9761, "MRR@10": 0.8868, "NDCG@10": 0.9090},
            no_alpha={"HR@1": 0.8417, "HR@5": 0.9637, "HR@10": 0.9752, "MRR@10": 0.8946, "NDCG@10": 0.9147},
            alpha={"HR@1": 0.8462, "HR@5": 0.9655, "HR@10": 0.9761, "MRR@10": 0.8978, "NDCG@10": 0.9174},
            source_report="electrical-engineering-ru.md",
            scope_label="full dataset",
            alpha_label="transferred alpha t=0.025, mix=0.1",
            no_alpha_label="contrastive t=0.03, alpha off",
            note="Alpha wins on all reported retrieval metrics.",
        ),
    ),
    DatasetSpec(
        dataset_id="meme-russian-ir",
        dataset_name_or_path="justatom/meme-russian-ir",
        content_field="description",
        labels_field="generated",
        chunk_id_col="id",
        split="train",
        stats_scope_label="full train split",
        metrics=MetricsSpec(
            baseline={"HR@1": 0.6310, "HR@5": 0.7647, "HR@10": 0.8034, "MRR@10": 0.6881, "NDCG@10": 0.7160},
            no_alpha={"HR@1": 0.7874, "HR@5": 0.9125, "HR@10": 0.9423, "MRR@10": 0.8423, "NDCG@10": 0.8667},
            alpha={"HR@1": 0.7913, "HR@5": 0.9143, "HR@10": 0.9427, "MRR@10": 0.8453, "NDCG@10": 0.8690},
            source_report="meme-russian-ir.md",
            scope_label="full dataset",
            alpha_label="transferred alpha t=0.025, mix=0.1",
            no_alpha_label="contrastive t=0.03, alpha off",
            note="Alpha wins on all reported retrieval metrics.",
        ),
    ),
    DatasetSpec(
        dataset_id="boolq-ru",
        dataset_name_or_path="hf://d0rj/boolq-ru",
        content_field="passage",
        labels_field="question",
        split="train",
        stats_scope_label="full train split",
        metrics=MetricsSpec(
            baseline={"HR@1": 0.5165, "HR@5": 0.6971, "HR@10": 0.7348, "MRR@10": 0.5942, "NDCG@10": 0.6284},
            no_alpha={"HR@1": 0.5602, "HR@5": 0.7423, "HR@10": 0.7765, "MRR@10": 0.6383, "NDCG@10": 0.6721},
            alpha={"HR@1": 0.5599, "HR@5": 0.7435, "HR@10": 0.7785, "MRR@10": 0.6395, "NDCG@10": 0.6736},
            source_report="boolq-ru.md",
            scope_label="full dataset",
            alpha_label="transferred alpha t=0.025, mix=0.1",
            no_alpha_label="contrastive t=0.03, alpha off",
            note="Alpha is nearly tied on HR@1 and slightly better on broader retrieval metrics.",
        ),
    ),
    DatasetSpec(
        dataset_id="miracl-ru",
        dataset_name_or_path=str(REPO_ROOT / ".data" / "retrieval" / "miracl-ru-train.jsonl"),
        content_field="content",
        labels_field="queries",
        chunk_id_col="chunk_id",
        stats_scope_label="full materialized train dataset",
        metrics=None,
    ),
    DatasetSpec(
        dataset_id="mMARCO-russian",
        dataset_name_or_path="hf://unicamp-dl/mmarco?config=russian",
        content_field="positive",
        labels_field="query",
        split="train",
        stats_limit=10000,
        stats_scope_label="fixed first 10k train slice used in experiments",
        metrics=MetricsSpec(
            baseline={"HR@1": 0.7971, "HR@5": 0.9381, "HR@10": 0.9583, "MRR@10": 0.8586, "NDCG@10": 0.8829},
            no_alpha={"HR@1": 0.8022, "HR@5": 0.9502, "HR@10": 0.9725, "MRR@10": 0.8679, "NDCG@10": 0.8934},
            alpha={"HR@1": 0.8068, "HR@5": 0.9524, "HR@10": 0.9713, "MRR@10": 0.8711, "NDCG@10": 0.8957},
            source_report="mMARCO-russian.md",
            scope_label="fixed 10k train slice",
            alpha_label="best alpha sweep t=0.02, mix=0.2",
            no_alpha_label="contrastive t=0.03, alpha off",
            note="Alpha wins on HR@1, HR@5, MRR@10, NDCG@10 and slightly trails on HR@10.",
        ),
    ),
]


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return float(ordered[lower])
    weight = pos - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(mean(values)),
        "median": quantile(values, 0.5),
        "p10": quantile(values, 0.10),
        "p25": quantile(values, 0.25),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "max": float(max(values)),
    }


def load_documents(spec: DatasetSpec) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "dataset_name_or_path": spec.dataset_name_or_path,
        "lazy": True,
        "content_col": spec.content_field,
        "queries_col": spec.labels_field,
    }
    if spec.chunk_id_col is not None:
        kwargs["chunk_id_col"] = spec.chunk_id_col
    if spec.split is not None:
        kwargs["split"] = spec.split
    if spec.stats_limit is not None:
        kwargs["limit"] = spec.stats_limit

    adapter = DatasetRecordAdapter.from_source(**kwargs)
    return list(adapter.iterator())


def compute_dataset_stats(spec: DatasetSpec) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    documents = load_documents(spec)
    query_lengths: list[int] = []
    content_lengths: list[int] = []
    labels_per_doc: list[int] = []
    overlap_counts: list[int] = []
    overlap_ratios: list[float] = []
    jaccards: list[float] = []
    query_ttr_values: list[float] = []
    content_ttr_values: list[float] = []
    query_to_content_ratio_values: list[float] = []
    query_coverage_values: list[float] = []
    content_coverage_values: list[float] = []
    length_plot_rows: list[dict[str, Any]] = []
    overlap_plot_rows: list[dict[str, Any]] = []

    doc_count = 0
    pair_count = 0

    for doc in documents:
        content = str(doc.get("content", "") or "")
        content_tokens = tokenize(content)
        content_len = len(content_tokens)
        content_token_set = set(content_tokens)
        content_lengths.append(content_len)
        content_ttr_values.append(len(content_token_set) / content_len if content_len else 0.0)
        length_plot_rows.append({"dataset": spec.dataset_id, "kind": "content", "words": content_len})
        doc_count += 1

        labels = doc.get("meta", {}).get("labels", []) or []
        if isinstance(labels, str):
            labels = [labels]
        labels_per_doc.append(len(labels))

        for query in labels:
            query_text = str(query or "")
            query_tokens = tokenize(query_text)
            query_len = len(query_tokens)
            query_set = set(query_tokens)
            overlap = len(query_set & content_token_set)
            union = len(query_set | content_token_set)
            overlap_ratio = overlap / query_len if query_len else 0.0
            jaccard = overlap / union if union else 0.0
            query_ttr = len(query_set) / query_len if query_len else 0.0
            query_to_content_ratio = query_len / content_len if content_len else 0.0
            query_coverage = overlap / len(query_set) if query_set else 0.0
            content_coverage = overlap / len(content_token_set) if content_token_set else 0.0

            query_lengths.append(query_len)
            overlap_counts.append(overlap)
            overlap_ratios.append(overlap_ratio)
            jaccards.append(jaccard)
            query_ttr_values.append(query_ttr)
            query_to_content_ratio_values.append(query_to_content_ratio)
            query_coverage_values.append(query_coverage)
            content_coverage_values.append(content_coverage)
            pair_count += 1

            length_plot_rows.append({"dataset": spec.dataset_id, "kind": "query", "words": query_len})
            overlap_plot_rows.append({"dataset": spec.dataset_id, "metric": "query_overlap_ratio", "value": overlap_ratio})
            overlap_plot_rows.append({"dataset": spec.dataset_id, "metric": "jaccard", "value": jaccard})

    stats = {
        "dataset": spec.dataset_id,
        "stats_scope": spec.stats_scope_label,
        "documents": doc_count,
        "pairs": pair_count,
        "labels_per_doc_mean": summarize([float(v) for v in labels_per_doc])["mean"],
        "query_words_mean": summarize([float(v) for v in query_lengths])["mean"],
        "query_words_median": summarize([float(v) for v in query_lengths])["median"],
        "query_words_p90": summarize([float(v) for v in query_lengths])["p90"],
        "content_words_mean": summarize([float(v) for v in content_lengths])["mean"],
        "content_words_median": summarize([float(v) for v in content_lengths])["median"],
        "content_words_p90": summarize([float(v) for v in content_lengths])["p90"],
        "overlap_count_mean": summarize([float(v) for v in overlap_counts])["mean"],
        "overlap_count_median": summarize([float(v) for v in overlap_counts])["median"],
        "overlap_ratio_mean": summarize(overlap_ratios)["mean"],
        "overlap_ratio_median": summarize(overlap_ratios)["median"],
        "overlap_ratio_p90": summarize(overlap_ratios)["p90"],
        "overlap_nonzero_share": (
            float(sum(1 for value in overlap_counts if value > 0) / len(overlap_counts)) if overlap_counts else 0.0
        ),
        "jaccard_mean": summarize(jaccards)["mean"],
        "jaccard_median": summarize(jaccards)["median"],
        "query_ttr_mean": summarize(query_ttr_values)["mean"],
        "content_ttr_mean": summarize(content_ttr_values)["mean"],
        "query_to_content_ratio_mean": summarize(query_to_content_ratio_values)["mean"],
        "query_coverage_mean": summarize(query_coverage_values)["mean"],
        "content_coverage_mean": summarize(content_coverage_values)["mean"],
    }
    return stats, pd.DataFrame(length_plot_rows), pd.DataFrame(overlap_plot_rows)


def build_metric_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    absolute_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []

    for spec in DATASETS:
        if spec.metrics is None:
            continue
        for metric in METRIC_KEYS:
            absolute_rows.append(
                {
                    "dataset": spec.dataset_id,
                    "run": "no-alpha",
                    "metric": metric,
                    "value": spec.metrics.no_alpha[metric],
                }
            )
            absolute_rows.append(
                {
                    "dataset": spec.dataset_id,
                    "run": "alpha",
                    "metric": metric,
                    "value": spec.metrics.alpha[metric],
                }
            )
            delta_rows.append(
                {
                    "dataset": spec.dataset_id,
                    "metric": metric,
                    "delta": spec.metrics.alpha[metric] - spec.metrics.no_alpha[metric],
                }
            )
    return pd.DataFrame(absolute_rows), pd.DataFrame(delta_rows)


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_grid(metric_df: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "alpha_vs_noalpha_metric_grid.png"
    fig, axes = plt.subplots(1, len(METRIC_KEYS), figsize=(28, 6), sharey=False)
    palette = {"no-alpha": "#5B8FF9", "alpha": "#F08A24"}

    for axis, metric in zip(axes, METRIC_KEYS, strict=True):
        subset = metric_df[metric_df["metric"] == metric]
        sns.barplot(data=subset, x="dataset", y="value", hue="run", palette=palette, ax=axis)
        axis.set_title(metric)
        axis.set_xlabel("")
        axis.set_ylabel("score")
        axis.tick_params(axis="x", rotation=45)
        if axis is not axes[0]:
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Alpha vs no-alpha across completed tuning datasets", y=1.06, fontsize=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_delta_heatmap(delta_df: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "alpha_minus_noalpha_heatmap.png"
    pivot = delta_df.pivot(index="dataset", columns="metric", values="delta").reindex(columns=METRIC_KEYS)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn", center=0.0, linewidths=0.5, ax=ax)
    ax.set_title("Alpha - no-alpha delta")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_gain_over_baseline() -> Path:
    output_path = OUTPUT_DIR / "gain_over_baseline.png"
    rows: list[dict[str, Any]] = []
    for spec in DATASETS:
        if spec.metrics is None:
            continue
        baseline_mrr = spec.metrics.baseline["MRR@10"]
        rows.append({"dataset": spec.dataset_id, "run": "no-alpha", "gain": spec.metrics.no_alpha["MRR@10"] - baseline_mrr})
        rows.append({"dataset": spec.dataset_id, "run": "alpha", "gain": spec.metrics.alpha["MRR@10"] - baseline_mrr})
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=frame, x="dataset", y="gain", hue="run", palette={"no-alpha": "#5B8FF9", "alpha": "#F08A24"}, ax=ax)
    ax.set_title("MRR@10 gain over baseline")
    ax.set_xlabel("")
    ax.set_ylabel("delta MRR@10")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_length_distributions(length_df: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "dataset_length_distributions.png"
    datasets = list(length_df["dataset"].drop_duplicates())
    fig, axes = plt.subplots(2, 3, figsize=(21, 12), sharex=False, sharey=False)
    axes_flat = list(axes.flatten())

    for axis, dataset in zip(axes_flat, datasets, strict=False):
        subset = length_df[length_df["dataset"] == dataset]
        sns.histplot(
            data=subset, x="words", hue="kind", bins=30, stat="density", common_norm=False, element="step", fill=False, ax=axis
        )
        axis.set_title(dataset)
        axis.set_xlabel("word count")
        axis.set_ylabel("density")
    for axis in axes_flat[len(datasets) :]:
        axis.axis("off")

    fig.suptitle("Query and content length distributions", y=1.02, fontsize=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_overlap_distributions(overlap_df: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "dataset_overlap_distributions.png"
    subset = overlap_df[overlap_df["metric"] == "query_overlap_ratio"]
    datasets = list(subset["dataset"].drop_duplicates())
    fig, axes = plt.subplots(2, 3, figsize=(21, 12), sharex=True, sharey=True)
    axes_flat = list(axes.flatten())

    for axis, dataset in zip(axes_flat, datasets, strict=False):
        ds_subset = subset[subset["dataset"] == dataset]
        sns.histplot(data=ds_subset, x="value", bins=30, color="#2A9D8F", ax=axis)
        axis.set_title(dataset)
        axis.set_xlabel("query overlap ratio")
        axis.set_ylabel("pairs")
    for axis in axes_flat[len(datasets) :]:
        axis.axis("off")

    fig.suptitle("Query-paragraph lexical overlap distributions", y=1.02, fontsize=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_dataset_profile_bars(stats_df: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "dataset_profile_bars.png"
    metric_specs = [
        ("labels_per_doc_mean", "labels per doc"),
        ("query_to_content_ratio_mean", "query/content ratio"),
        ("query_ttr_mean", "query TTR"),
        ("content_ttr_mean", "content TTR"),
        ("query_coverage_mean", "query coverage"),
        ("content_coverage_mean", "content coverage"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharey=False)
    axes_flat = list(axes.flatten())
    palette = sns.color_palette("crest", n_colors=len(stats_df))

    for axis, (column, title) in zip(axes_flat, metric_specs, strict=True):
        subset = stats_df.sort_values(column, ascending=False)
        sns.barplot(data=subset, x="dataset", y=column, hue="dataset", palette=palette, dodge=False, legend=False, ax=axis)
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="x", rotation=45)

    fig.suptitle("More discriminative dataset-profile metrics", y=1.02, fontsize=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_dataset_signature_scatter(stats_df: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "dataset_signature_scatter.png"
    fig, ax = plt.subplots(figsize=(11, 8))
    sizes = stats_df["labels_per_doc_mean"].clip(lower=0.8) * 220
    scatter = ax.scatter(
        stats_df["query_to_content_ratio_mean"],
        stats_df["query_coverage_mean"],
        s=sizes,
        c=stats_df["content_ttr_mean"],
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.8,
    )
    for _, row in stats_df.iterrows():
        ax.annotate(
            row["dataset"],
            (row["query_to_content_ratio_mean"], row["query_coverage_mean"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
        )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("content TTR")
    ax.set_title("Dataset lexical signature: brevity vs lexical grounding")
    ax.set_xlabel("query/content length ratio")
    ax.set_ylabel("unique-query token coverage")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def comparison_table_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in DATASETS:
        if spec.metrics is None:
            continue
        row: dict[str, Any] = {
            "dataset": spec.dataset_id,
            "scope": spec.metrics.scope_label,
            "alpha_label": spec.metrics.alpha_label,
            "no_alpha_label": spec.metrics.no_alpha_label,
            "source_report": spec.metrics.source_report,
            "note": spec.metrics.note,
        }
        for metric in METRIC_KEYS:
            row[f"no_alpha_{metric}"] = spec.metrics.no_alpha[metric]
            row[f"alpha_{metric}"] = spec.metrics.alpha[metric]
            row[f"delta_{metric}"] = spec.metrics.alpha[metric] - spec.metrics.no_alpha[metric]
        rows.append(row)
    return rows


def format_metric_row(values: dict[str, float]) -> str:
    return " | ".join(f"{values[key]:.4f}" for key in METRIC_KEYS)


def build_report(
    stats_rows: list[dict[str, Any]],
    metric_plot: Path,
    heatmap_plot: Path,
    baseline_gain_plot: Path,
    length_plot: Path,
    overlap_plot: Path,
    profile_bars_plot: Path,
    signature_scatter_plot: Path,
) -> str:
    completed = [spec for spec in DATASETS if spec.metrics is not None]
    wins_all = [
        spec.dataset_id for spec in completed if all(spec.metrics.alpha[key] > spec.metrics.no_alpha[key] for key in METRIC_KEYS)
    ]
    wins_most = [
        spec.dataset_id
        for spec in completed
        if sum(spec.metrics.alpha[key] > spec.metrics.no_alpha[key] for key in METRIC_KEYS) >= 4
    ]

    stats_by_dataset = {row["dataset"]: row for row in stats_rows}
    stats_df = pd.DataFrame(stats_rows)
    highest_alignment = stats_df.sort_values("query_coverage_mean", ascending=False).iloc[0]["dataset"]
    most_multiquery = stats_df.sort_values("labels_per_doc_mean", ascending=False).iloc[0]["dataset"]
    most_repetitive_surface = stats_df.sort_values("query_ttr_mean", ascending=True).iloc[0]["dataset"]
    lines: list[str] = []
    lines.append("# Alpha vs No-Alpha Report")
    lines.append("")
    lines.append(
        "Этот отчёт агрегирует все non-MLNavigator retrieval-датасеты, которые сейчас присутствуют в репозитории: `justatom`, `electrical-engineering-ru`, `meme-russian-ir`, `boolq-ru`, `miracl-ru`, `mMARCO-russian`."
    )
    lines.append("")
    lines.append(
        "Сравнение alpha vs no-alpha доступно для пяти наборов: четыре полноразмерных датасета из `TABLE_RESULTS.md` и контролируемый `mMARCO-russian` fixed-10k slice. Для `miracl-ru` в репозитории пока нет завершённой пары no-alpha/alpha tune, поэтому он включён в статистический профиль, но исключён из quality-comparison графиков."
    )
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Alpha выигрывает все 5 из 5 метрик на: `{', '.join(wins_all)}`.")
    lines.append(
        f"- Если считать `mMARCO-russian` fixed-10k slice, alpha выигрывает минимум 4 из 5 метрик на: `{', '.join(sorted(set(wins_most)))}`."
    )
    lines.append(
        "- Самый слабый transfer-case: `boolq-ru`, где `HR@1` почти не меняется (`-0.0003`), но `HR@5`, `HR@10`, `MRR@10`, `NDCG@10` всё равно слегка лучше у alpha."
    )
    lines.append(
        "- На `mMARCO-russian` alpha улучшает top-rank метрики (`HR@1`, `HR@5`, `MRR@10`, `NDCG@10`), но немного уступает по `HR@10`, поэтому это скорее targeted improvement, чем безусловная победа по всей кривой recall."
    )
    lines.append(
        f"- По более различающим признакам датасеты уже не выглядят одинаково: `{highest_alignment}` даёт самую сильную lexical grounding, `{most_multiquery}` содержит больше всего query variants на документ, а `{most_repetitive_surface}` сильнее остальных уходит в повторяющиеся query surface forms."
    )
    lines.append("")
    lines.append("## Approach And Intuition")
    lines.append("")
    lines.append("Используемая идея взята из уже зафиксированной рабочей гипотезы в `PhD (train).md` и `TRAIN.md`:")
    lines.append("")
    lines.append("- обычный no-alpha tune учит encoder только по semantic contrastive objective;")
    lines.append("- alpha-режим включает `gamma_joint=true` и учит query-dependent gate `alpha(q)`;")
    lines.append("- mixed score строится как `alpha(q) * semantic + (1 - alpha(q)) * lexical`;")
    lines.append(
        "- в `alpha_train_only=true` lexical branch используется как мягкий train-time guide, а не как обязательный inference-time dependency."
    )
    lines.append("")
    lines.append("Интуиция такая:")
    lines.append("")
    lines.append(
        "1. Не все запросы одинаково semantic-heavy. Короткие, keyword-like или entity-heavy запросы часто выигрывают, если во время обучения encoder видит мягкий lexical anchor."
    )
    lines.append(
        "2. `alpha(q)` делает это не глобально, а query-wise: одна и та же модель может больше доверять semantic ветке для одного запроса и lexical ветке для другого."
    )
    lines.append(
        "3. В режиме `alpha_train_only` это помогает encoder не потерять semantic purity на inference, но всё равно получить дополнительный supervision signal во время train."
    )
    lines.append(
        "4. Практический вывод из уже проведённых экспериментов: alpha не является автоматическим win. Он начинает стабильно помогать только после калибровки `contrastive_temperature` и `alpha_mix_weight`."
    )
    lines.append("")
    lines.append(
        "С инженерной точки зрения это хорошо согласуется с наблюдением из thesis notes: alpha полезен не как замена обычному contrastive, а как calibrated auxiliary mechanism, который особенно помогает top-rank retrieval quality."
    )
    lines.append("")
    lines.append("## Shared Experimental Setup")
    lines.append("")
    lines.append(
        "Для полноразмерных датасетов сводка опирается на уже зафиксированные runs из `TABLE_RESULTS.md` и dataset-specific notes:"
    )
    lines.append("")
    lines.append("- no-alpha reference: `loss=contrastive`, `temperature=0.03`, `optimizer=adamw`, `alpha=off`;")
    lines.append(
        "- transferred/best alpha reference: `loss=contrastive`, calibrated `temperature`, `alpha_train_only=true`, `gamma_joint=true`, calibrated `alpha_mix_weight`;"
    )
    lines.append(
        "- shared train recipe for the public full datasets: `batch_size=96`, `grad_acc_steps=6`, `n_epochs=1`, `--auto-e5-prefixes`."
    )
    lines.append("")
    lines.append(
        "Для `mMARCO-russian` сравнение идёт не на полном train split (он слишком велик), а на том же fixed first `10000` rows train slice, где уже проводился alpha sweep."
    )
    lines.append("")
    lines.append("## Cross-Dataset Metrics")
    lines.append("")
    lines.append(f"![Alpha vs no-alpha absolute metrics]({metric_plot.relative_to(REPO_ROOT).as_posix()})")
    lines.append("")
    lines.append(f"![Alpha minus no-alpha heatmap]({heatmap_plot.relative_to(REPO_ROOT).as_posix()})")
    lines.append("")
    lines.append(f"![MRR gain over baseline]({baseline_gain_plot.relative_to(REPO_ROOT).as_posix()})")
    lines.append("")
    lines.append("### Comparison Table")
    lines.append("")
    lines.append(
        "| Dataset | Scope | No-alpha HR@1 | No-alpha HR@5 | No-alpha HR@10 | No-alpha MRR@10 | No-alpha NDCG@10 | Alpha HR@1 | Alpha HR@5 | Alpha HR@10 | Alpha MRR@10 | Alpha NDCG@10 | Delta HR@1 | Delta HR@5 | Delta HR@10 | Delta MRR@10 | Delta NDCG@10 |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for spec in completed:
        metrics = spec.metrics
        delta = {key: metrics.alpha[key] - metrics.no_alpha[key] for key in METRIC_KEYS}
        lines.append(
            f"| `{spec.dataset_id}` | {metrics.scope_label} | {metrics.no_alpha['HR@1']:.4f} | {metrics.no_alpha['HR@5']:.4f} | {metrics.no_alpha['HR@10']:.4f} | {metrics.no_alpha['MRR@10']:.4f} | {metrics.no_alpha['NDCG@10']:.4f} | {metrics.alpha['HR@1']:.4f} | {metrics.alpha['HR@5']:.4f} | {metrics.alpha['HR@10']:.4f} | {metrics.alpha['MRR@10']:.4f} | {metrics.alpha['NDCG@10']:.4f} | {delta['HR@1']:+.4f} | {delta['HR@5']:+.4f} | {delta['HR@10']:+.4f} | {delta['MRR@10']:+.4f} | {delta['NDCG@10']:+.4f} |"
        )
    lines.append("")
    lines.append("### Reading The Metric Story")
    lines.append("")
    lines.append(
        "- `justatom`: best calibrated alpha (`t=0.025`, `mix=0.1`) улучшает все tracked metrics; это самый чистый thesis-level positive case."
    )
    lines.append(
        "- `electrical-engineering-ru`: transfer почти без деградации, причём alpha возвращает `HR@5` и `HR@10` к уровню baseline и улучшает rank-sensitive metrics."
    )
    lines.append("- `meme-russian-ir`: alpha даёт небольшой, но консистентный прирост поверх уже сильного no-alpha tune.")
    lines.append(
        "- `boolq-ru`: alpha не помогает на самом верхнем hit-rate, зато остаётся чуть лучше на более широком top-k и на rank-sensitive metrics."
    )
    lines.append(
        "- `mMARCO-russian`: на fixed 10k slice alpha выглядит полезным именно для top-rank quality, а не для полного `HR@10` domination."
    )
    lines.append("")
    lines.append("## Dataset Statistics")
    lines.append("")
    lines.append(
        "Статистика ниже посчитана заново по сырым данным через `DatasetRecordAdapter`. Для `mMARCO-russian` она считается на том же fixed 10k slice, что и tuning/eval, чтобы не делать ложное сравнение с полным 39.8M-row train split. Старые mean-overlap числа действительно местами близки, поэтому ниже добавлены более различающие признаки: multi-query density, query/content ratio, lexical diversity (TTR) и coverage по уникальным токенам."
    )
    lines.append("")
    lines.append(f"![More discriminative dataset profile metrics]({profile_bars_plot.relative_to(REPO_ROOT).as_posix()})")
    lines.append("")
    lines.append(f"![Dataset lexical signature scatter]({signature_scatter_plot.relative_to(REPO_ROOT).as_posix()})")
    lines.append("")
    lines.append(f"![Query and content length distributions]({length_plot.relative_to(REPO_ROOT).as_posix()})")
    lines.append("")
    lines.append(f"![Query-paragraph overlap distributions]({overlap_plot.relative_to(REPO_ROOT).as_posix()})")
    lines.append("")
    lines.append("### Why These Datasets Are Not Actually The Same")
    lines.append("")
    lines.append(
        "- `justatom` выделяется не только длиной query/content, но и низким `query_ttr`: это признак более повторяющегося формата запросов и более плотного reuse surface patterns."
    )
    lines.append(
        "- `meme-russian-ir` близок к `boolq-ru` по средней длине query, но резко отличается по `labels_per_doc`: это не single-query retrieval, а multi-query paraphrase-heavy setup."
    )
    lines.append(
        "- `electrical-engineering-ru` даёт максимальные `query_coverage` и `content_coverage`, то есть лексика query гораздо сильнее зацепляется за релевантный passage, чем в остальных наборах."
    )
    lines.append(
        "- `boolq-ru`, `miracl-ru` и `mMARCO-russian` действительно похожи по грубому overlap, но расходятся по длине passage и `query/content ratio`: у `boolq-ru` и `miracl-ru` passages тяжелее и длиннее, а `mMARCO-russian` компактнее."
    )
    lines.append("")
    lines.append("### Comparative Profile Table")
    lines.append("")
    lines.append(
        "| Dataset | Labels/doc | Query/content ratio | Query TTR | Content TTR | Unique-query coverage | Unique-content coverage |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in stats_rows:
        lines.append(
            f"| `{row['dataset']}` | {row['labels_per_doc_mean']:.3f} | {row['query_to_content_ratio_mean']:.3f} | {row['query_ttr_mean']:.3f} | {row['content_ttr_mean']:.3f} | {row['query_coverage_mean']:.3f} | {row['content_coverage_mean']:.3f} |"
        )
    lines.append("")
    lines.append("### Summary Stats Table")
    lines.append("")
    lines.append(
        "| Dataset | Scope | Documents | Query-doc pairs | Query words mean | Query words p90 | Content words mean | Content words p90 | Overlap count mean | Query overlap mean | Query overlap p90 | Non-zero overlap share | Jaccard mean | Labels/doc |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in stats_rows:
        lines.append(
            f"| `{row['dataset']}` | {row['stats_scope']} | {row['documents']} | {row['pairs']} | {row['query_words_mean']:.2f} | {row['query_words_p90']:.2f} | {row['content_words_mean']:.2f} | {row['content_words_p90']:.2f} | {row['overlap_count_mean']:.2f} | {row['overlap_ratio_mean']:.3f} | {row['overlap_ratio_p90']:.3f} | {row['overlap_nonzero_share']:.3f} | {row['jaccard_mean']:.3f} | {row['labels_per_doc_mean']:.3f} |"
        )
    lines.append("")
    lines.append("### Per-Dataset Readout")
    lines.append("")
    for spec in DATASETS:
        row = stats_by_dataset[spec.dataset_id]
        lines.append(f"#### `{spec.dataset_id}`")
        lines.append("")
        lines.append(f"- Stats scope: {row['stats_scope']}")
        lines.append(f"- Documents: `{row['documents']}`")
        lines.append(f"- Query-document pairs: `{row['pairs']}`")
        lines.append(
            f"- Mean query length: `{row['query_words_mean']:.2f}` words; median `{row['query_words_median']:.2f}`; p90 `{row['query_words_p90']:.2f}`"
        )
        lines.append(
            f"- Mean content length: `{row['content_words_mean']:.2f}` words; median `{row['content_words_median']:.2f}`; p90 `{row['content_words_p90']:.2f}`"
        )
        lines.append(f"- Mean shared-token count between query and paragraph: `{row['overlap_count_mean']:.2f}`")
        lines.append(f"- Mean query-overlap ratio: `{row['overlap_ratio_mean']:.3f}`; p90 `{row['overlap_ratio_p90']:.3f}`")
        lines.append(f"- Share of pairs with at least one shared token: `{row['overlap_nonzero_share']:.3f}`")
        lines.append(f"- Mean Jaccard overlap: `{row['jaccard_mean']:.3f}`")
        lines.append(f"- Mean labels per document: `{row['labels_per_doc_mean']:.3f}`")
        lines.append(f"- Mean query/content length ratio: `{row['query_to_content_ratio_mean']:.3f}`")
        lines.append(f"- Mean query TTR: `{row['query_ttr_mean']:.3f}`; mean content TTR: `{row['content_ttr_mean']:.3f}`")
        lines.append(
            f"- Unique-query token coverage: `{row['query_coverage_mean']:.3f}`; unique-content coverage: `{row['content_coverage_mean']:.3f}`"
        )
        if spec.metrics is not None:
            lines.append(f"- Alpha/no-alpha comparison scope: {spec.metrics.scope_label}")
            lines.append(f"- Alpha note: {spec.metrics.note}")
            lines.append(f"- Metrics source: `{spec.metrics.source_report}`")
        else:
            lines.append(
                "- Alpha/no-alpha metrics are not available yet in the repository, so this dataset is profile-only in the current report."
            )
        lines.append("")
    lines.append("## Limitations And Interpretation Boundaries")
    lines.append("")
    lines.append(
        "- `miracl-ru` is included only in the data-profile section because the repository does not currently contain a completed no-alpha/alpha pair for it."
    )
    lines.append(
        "- `mMARCO-russian` comparison is intentionally slice-scoped (`first 10k train rows`) and should not be read as a full-dataset claim."
    )
    lines.append(
        "- Word statistics use a simple Unicode word tokenizer and lexical overlap is token-set based; this is good for relative comparison, but not a substitute for full morphological normalization."
    )
    lines.append(
        "- The main conclusion should therefore be phrased conservatively: calibrated query-adaptive alpha usually matches or slightly outperforms no-alpha tuning across the datasets already tested here, with the strongest gains showing up in top-rank and rank-sensitive retrieval metrics."
    )
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append(f"- Metrics and plots directory: `{OUTPUT_DIR.relative_to(REPO_ROOT).as_posix()}`")
    lines.append(f"- This report: `{REPORT_PATH.relative_to(REPO_ROOT).as_posix()}`")
    lines.append(
        "- Metric comparison sources: `TABLE_RESULTS.md`, `justatom.md`, `electrical-engineering-ru.md`, `meme-russian-ir.md`, `boolq-ru.md`, `mMARCO-russian.md`"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stats_rows: list[dict[str, Any]] = []
    length_frames: list[pd.DataFrame] = []
    overlap_frames: list[pd.DataFrame] = []

    for spec in DATASETS:
        stats, length_frame, overlap_frame = compute_dataset_stats(spec)
        stats_rows.append(stats)
        length_frames.append(length_frame)
        overlap_frames.append(overlap_frame)

    stats_rows.sort(key=lambda row: row["dataset"])
    metric_df, delta_df = build_metric_frames()
    stats_df = pd.DataFrame(stats_rows)
    length_df = pd.concat(length_frames, ignore_index=True)
    overlap_df = pd.concat(overlap_frames, ignore_index=True)

    metric_plot = plot_metric_grid(metric_df)
    heatmap_plot = plot_delta_heatmap(delta_df)
    baseline_gain_plot = plot_gain_over_baseline()
    profile_bars_plot = plot_dataset_profile_bars(stats_df)
    signature_scatter_plot = plot_dataset_signature_scatter(stats_df)
    length_plot = plot_length_distributions(length_df)
    overlap_plot = plot_overlap_distributions(overlap_df)

    comparison_rows = comparison_table_rows()
    save_csv(
        OUTPUT_DIR / "comparison_metrics.csv",
        comparison_rows,
        fieldnames=list(comparison_rows[0].keys()),
    )
    save_csv(
        OUTPUT_DIR / "dataset_statistics.csv",
        stats_rows,
        fieldnames=list(stats_rows[0].keys()),
    )

    report = build_report(
        stats_rows=stats_rows,
        metric_plot=metric_plot,
        heatmap_plot=heatmap_plot,
        baseline_gain_plot=baseline_gain_plot,
        length_plot=length_plot,
        overlap_plot=overlap_plot,
        profile_bars_plot=profile_bars_plot,
        signature_scatter_plot=signature_scatter_plot,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote report to {REPORT_PATH}")
    print(f"Artifacts directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
