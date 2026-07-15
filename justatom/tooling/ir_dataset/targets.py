from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
import uuid
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import polars as pl


INTENT_SHARES = (
    ("how_to", 0.25),
    ("why", 0.15),
    ("troubleshooting", 0.15),
    ("concept", 0.15),
    ("comparison", 0.10),
    ("requirements", 0.08),
    ("limitations", 0.07),
    ("factual", 0.05),
)
_CODE_SYMBOL_RE = re.compile(r"[\[\]{}<>_=+*/\\|;:$#@%~`^()]")
_REPEATED_CHARACTER_RE = re.compile(r"(.)\1{80,}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class TargetSelectionConfig:
    article_count: int = 50
    seed: int = 42
    output_dir: str | Path | None = None
    max_flow_share: float = 0.30
    exclude_target_roots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.article_count <= 0:
            raise ValueError("target selection article_count must be > 0")
        if not 0 < self.max_flow_share <= 1:
            raise ValueError("target selection max_flow_share must be within (0, 1]")
        roots = self.exclude_target_roots
        if isinstance(roots, (str, Path)):
            raise ValueError("target selection exclude_target_roots must be a sequence of artifact roots")
        normalized = tuple(str(Path(root)) for root in roots)
        if any(not root for root in normalized):
            raise ValueError("target selection exclude_target_roots must not contain empty paths")
        if len(normalized) != len(set(normalized)):
            raise ValueError("target selection exclude_target_roots must be unique")
        object.__setattr__(self, "exclude_target_roots", normalized)


@dataclass(frozen=True, slots=True)
class PassageQuality:
    score: float
    eligible: bool
    reason_codes: tuple[str, ...]
    token_count: int
    alpha_ratio: float
    symbol_ratio: float
    long_token_ratio: float
    line_count: int
    mean_line_length: float
    max_line_length: int
    max_repeated_run: int


def _text(row: Mapping[str, Any]) -> str:
    return str(row.get("content") or row.get("serialized_passage") or "").strip()


def _normalized_content(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(value))).strip().casefold()


def selection_quality_reason_counts(passages: pl.DataFrame) -> dict[str, int]:
    if not isinstance(passages, pl.DataFrame):
        raise TypeError("passages must be a polars DataFrame")
    rows = passages.to_dicts()
    content_counts = Counter(_normalized_content(row.get("content", "")) for row in rows)
    reasons: Counter[str] = Counter()
    for row in rows:
        quality = score_passage_quality(row)
        reasons.update(quality.reason_codes)
        if content_counts[_normalized_content(row.get("content", ""))] > 1:
            reasons["globally_duplicate_content"] += 1
    return dict(sorted(reasons.items()))


def _token_count(row: Mapping[str, Any], tokens: list[str]) -> int:
    if "token_count" not in row or row["token_count"] is None:
        return len(tokens)
    try:
        return int(row["token_count"])
    except (TypeError, ValueError):
        return len(tokens)


def score_passage_quality(row: Mapping[str, Any]) -> PassageQuality:
    text = _text(row)
    tokens = re.findall(r"\S+", text)
    token_count = _token_count(row, tokens)
    non_whitespace = [character for character in text if not character.isspace()]
    character_count = len(non_whitespace)
    alpha_ratio = sum(character.isalpha() for character in non_whitespace) / character_count if character_count else 0.0
    symbol_ratio = len(_CODE_SYMBOL_RE.findall(text)) / character_count if character_count else 1.0
    long_token_ratio = sum(len(token) > 30 for token in tokens) / len(tokens) if tokens else 1.0
    lines = text.splitlines() or [text]
    line_lengths = [len(line) for line in lines]
    repeated_runs = [len(match.group(0)) for match in _REPEATED_CHARACTER_RE.finditer(text)]
    max_repeated_run = max(repeated_runs, default=0)

    reason_codes: list[str] = []
    if token_count < 80:
        reason_codes.append("token_count_lt_80")
    if alpha_ratio < 0.55:
        reason_codes.append("alpha_ratio_lt_0.55")
    if symbol_ratio > 0.18:
        reason_codes.append("symbol_ratio_gt_0.18")
    if long_token_ratio > 0.05:
        reason_codes.append("long_token_ratio_gt_0.05")
    if max_repeated_run > 80:
        reason_codes.append("repeated_character_run_gt_80")

    score = max(
        0.0,
        min(
            1.0,
            0.40 * min(alpha_ratio / 0.55, 1.0)
            + 0.25 * max(0.0, 1.0 - symbol_ratio / 0.18)
            + 0.20 * max(0.0, 1.0 - long_token_ratio / 0.05)
            + 0.15 * min(token_count / 80.0, 1.0),
        ),
    )
    return PassageQuality(
        score=score,
        eligible=not reason_codes,
        reason_codes=tuple(reason_codes),
        token_count=token_count,
        alpha_ratio=alpha_ratio,
        symbol_ratio=symbol_ratio,
        long_token_ratio=long_token_ratio,
        line_count=len(lines),
        mean_line_length=sum(line_lengths) / len(line_lengths),
        max_line_length=max(line_lengths, default=0),
        max_repeated_run=max_repeated_run,
    )


def _normalized_labels(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    values: Sequence[Any]
    if isinstance(value, str):
        values = (value,)
    else:
        try:
            values = tuple(value)
        except TypeError:
            values = (value,)
    return tuple(sorted({label for item in values if (label := str(item).strip().casefold())}))


def _primary_label(row: Mapping[str, Any], column: str) -> str:
    labels = _normalized_labels(row.get(column))
    return labels[0] if labels else "other"


def _sha256(seed: int, *parts: str) -> str:
    return hashlib.sha256(":".join((str(seed), *parts)).encode("utf-8")).hexdigest()


def _stable_uniform(seed: int, *parts: str) -> float:
    return (int(_sha256(seed, *parts), 16) + 1) / (2**256 + 1)


def _largest_remainder(total: int, weighted_values: Sequence[tuple[str, float]], seed: int, namespace: str) -> dict[str, int]:
    if total <= 0:
        return {name: 0 for name, _ in weighted_values}
    weight_sum = sum(weight for _, weight in weighted_values)
    if weight_sum <= 0:
        raise ValueError("weighted allocation requires a positive total weight")
    raw = {name: total * weight / weight_sum for name, weight in weighted_values}
    allocated = {name: math.floor(value) for name, value in raw.items()}
    for name, _ in sorted(
        weighted_values,
        key=lambda item: (-(raw[item[0]] - allocated[item[0]]), _sha256(seed, namespace, item[0]), item[0]),
    )[: total - sum(allocated.values())]:
        allocated[name] += 1
    return allocated


def _flow_quotas(articles_by_flow: Mapping[str, list[dict[str, Any]]], config: TargetSelectionConfig) -> dict[str, int]:
    requested = config.article_count
    weights = [(flow, math.sqrt(len(articles))) for flow, articles in articles_by_flow.items()]
    raw_quotas = _largest_remainder(requested, weights, config.seed, "flow-quota")
    cap = max(1, math.ceil(requested * config.max_flow_share))
    capped_capacity = sum(min(len(articles), cap) for articles in articles_by_flow.values())
    if capped_capacity < requested:
        raise ValueError(
            f"requested {requested} target articles cannot satisfy max_flow_share={config.max_flow_share:.2f}: "
            f"only {capped_capacity} articles fit within the primary-flow cap of {cap}"
        )
    quotas = {flow: min(raw_quotas[flow], len(articles), cap) for flow, articles in articles_by_flow.items()}
    remaining = requested - sum(quotas.values())
    while remaining:
        candidates = [flow for flow, articles in articles_by_flow.items() if quotas[flow] < len(articles) and quotas[flow] < cap]
        if not candidates:
            raise ValueError("primary-flow quotas cannot satisfy target selection within the configured cap")
        flow = min(
            candidates,
            key=lambda name: (
                -(requested * math.sqrt(len(articles_by_flow[name])) / sum(weight for _, weight in weights) - quotas[name]),
                _sha256(config.seed, "flow-remainder", name),
                name,
            ),
        )
        quotas[flow] += 1
        remaining -= 1
    return quotas


def _article_rank(
    article: Mapping[str, Any], hub_frequency: Mapping[str, int], config: TargetSelectionConfig
) -> tuple[float, str, str]:
    hub = str(article["primary_hub"])
    weight = 1.0 / math.sqrt(hub_frequency[hub])
    uniform = _stable_uniform(config.seed, "article", str(article["article_id"]))
    return (-math.log(uniform) / weight, _sha256(config.seed, "article", str(article["article_id"])), str(article["article_id"]))


def _choose_passages(article: Mapping[str, Any], config: TargetSelectionConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = list(article["passages"])
    ranked = sorted(
        candidates,
        key=lambda item: (
            -item["quality"].score,
            _sha256(config.seed, "passage", str(article["article_id"]), str(item["passage_id"])),
            str(item["passage_id"]),
        ),
    )

    def section_label(candidate: Mapping[str, Any]) -> str:
        return str(candidate.get("section") or "").strip().casefold()

    non_empty_sections = {section_label(candidate) for candidate in ranked if section_label(candidate)}
    if len(non_empty_sections) >= 2:
        non_empty_ranked = [candidate for candidate in ranked if section_label(candidate)]
        first = non_empty_ranked[0]
        second = next(
            candidate
            for candidate in non_empty_ranked[1:]
            if str(candidate.get("passage_id")) != str(first.get("passage_id")) and section_label(candidate) != section_label(first)
        )
        return first, second

    first = ranked[0]
    first_section = section_label(first)
    distinct_section = [
        candidate
        for candidate in ranked[1:]
        if str(candidate.get("passage_id")) != str(first.get("passage_id")) and section_label(candidate) != first_section
    ]
    second = (
        distinct_section[0]
        if distinct_section
        else next(candidate for candidate in ranked[1:] if str(candidate.get("passage_id")) != str(first.get("passage_id")))
    )
    return first, second


def _split_counts(article_count: int, config: TargetSelectionConfig) -> dict[str, int]:
    if article_count == 5_000:
        return {"train": 4_000, "dev": 500, "test": 500}
    return _largest_remainder(article_count, (("train", 0.8), ("dev", 0.1), ("test", 0.1)), config.seed, "split")


def _assign_splits(articles: list[dict[str, Any]], config: TargetSelectionConfig) -> dict[str, str]:
    ranked = sorted(
        articles,
        key=lambda article: (
            _sha256(config.seed, "split", str(article["article_id"])),
            str(article["article_id"]),
        ),
    )
    split_by_article: dict[str, str] = {}
    start = 0
    for split, count in _split_counts(len(ranked), config).items():
        for article in ranked[start : start + count]:
            split_by_article[str(article["article_id"])] = split
        start += count
    return split_by_article


def _assign_intents(articles: list[dict[str, Any]], config: TargetSelectionConfig) -> dict[str, tuple[str, str]]:
    intent_by_article: dict[str, tuple[str, str]] = {}
    by_flow: dict[str, list[dict[str, Any]]] = {}
    for article in articles:
        by_flow.setdefault(str(article["primary_flow"]), []).append(article)
    for flow, flow_articles in by_flow.items():
        remaining = _largest_remainder(2 * len(flow_articles), INTENT_SHARES, config.seed, f"intent:{flow}")
        ordered_articles = sorted(
            flow_articles,
            key=lambda article: (
                _sha256(config.seed, "intent-article", flow, str(article["article_id"])),
                str(article["article_id"]),
            ),
        )
        for article in ordered_articles:
            article_id = str(article["article_id"])
            first = max(
                (intent for intent, count in remaining.items() if count),
                key=lambda intent: (remaining[intent], _sha256(config.seed, "intent", flow, article_id, intent)),
            )
            remaining[first] -= 1
            second = max(
                (intent for intent, count in remaining.items() if count and intent != first),
                key=lambda intent: (remaining[intent], _sha256(config.seed, "intent", flow, article_id, intent)),
            )
            remaining[second] -= 1
            intent_by_article[article_id] = (first, second)
    return intent_by_article


def _quality_columns(quality: PassageQuality) -> dict[str, Any]:
    return {
        "quality_score": quality.score,
        "quality_reason_codes": list(quality.reason_codes),
        "quality_alpha_ratio": quality.alpha_ratio,
        "quality_symbol_ratio": quality.symbol_ratio,
        "quality_long_token_ratio": quality.long_token_ratio,
        "quality_line_count": quality.line_count,
        "quality_mean_line_length": quality.mean_line_length,
        "quality_max_line_length": quality.max_line_length,
        "quality_max_repeated_run": quality.max_repeated_run,
    }


def _has_structural_sibling(passage: Mapping[str, Any], article_passages: Sequence[Mapping[str, Any]]) -> bool:
    passage_id = str(passage["passage_id"])
    passage_content = str(passage["content"])
    passage_start = int(passage["start_unit"])
    passage_end = int(passage["end_unit"])
    for sibling in article_passages:
        if str(sibling["passage_id"]) == passage_id or str(sibling["content"]) == passage_content:
            continue
        sibling_start = int(sibling["start_unit"])
        sibling_end = int(sibling["end_unit"])
        if max(passage_start - sibling_end, sibling_start - passage_end, 0) <= 1:
            return True
    return False


def select_target_slots(
    passages: pl.DataFrame,
    config: TargetSelectionConfig | None = None,
    *,
    excluded_article_ids: Collection[str] = (),
) -> pl.DataFrame:
    if not isinstance(passages, pl.DataFrame):
        raise TypeError("passages must be a polars DataFrame")
    active_config = config or TargetSelectionConfig()
    required_columns = {"article_id", "passage_id", "content", "flows", "hubs", "start_unit", "end_unit"}
    missing_columns = sorted(required_columns - set(passages.columns))
    if missing_columns:
        raise ValueError(f"passages is missing required columns: {', '.join(missing_columns)}")

    excluded = {str(article_id) for article_id in excluded_article_ids}
    passage_rows = [row for row in passages.to_dicts() if str(row["article_id"]) not in excluded]
    content_counts = Counter(_normalized_content(row["content"]) for row in passage_rows)
    full_article_passages: dict[str, list[dict[str, Any]]] = {}
    for row in passage_rows:
        full_article_passages.setdefault(str(row["article_id"]), []).append(row)

    eligible_by_article: dict[str, list[dict[str, Any]]] = {}
    quality_reason_counts: Counter[str] = Counter()
    for row in passage_rows:
        quality = score_passage_quality(row)
        if content_counts[_normalized_content(row["content"])] > 1:
            quality = replace(
                quality,
                eligible=False,
                reason_codes=tuple(dict.fromkeys((*quality.reason_codes, "globally_duplicate_content"))),
            )
        quality_reason_counts.update(quality.reason_codes)
        if not quality.eligible:
            continue
        article_id = str(row["article_id"])
        if not _has_structural_sibling(row, full_article_passages[article_id]):
            continue
        enriched = dict(row)
        enriched["quality"] = quality
        enriched["primary_flow"] = _primary_label(row, "flows")
        enriched["primary_hub"] = _primary_label(row, "hubs")
        eligible_by_article.setdefault(article_id, []).append(enriched)

    articles_by_flow: dict[str, list[dict[str, Any]]] = {}
    for article_id, article_passages in eligible_by_article.items():
        distinct_passage_ids = {str(passage["passage_id"]) for passage in article_passages}
        if len(distinct_passage_ids) < 2:
            continue
        primary_flow = str(article_passages[0]["primary_flow"])
        primary_hub = str(article_passages[0]["primary_hub"])
        articles_by_flow.setdefault(primary_flow, []).append(
            {
                "article_id": article_id,
                "primary_flow": primary_flow,
                "primary_hub": primary_hub,
                "passages": article_passages,
            }
        )

    eligible_article_count = sum(len(articles) for articles in articles_by_flow.values())
    if eligible_article_count < active_config.article_count:
        raise ValueError(
            f"requested {active_config.article_count} target articles but only {eligible_article_count} have two eligible passages"
        )

    quotas = _flow_quotas(articles_by_flow, active_config)
    selected_articles: list[dict[str, Any]] = []
    for flow, articles in articles_by_flow.items():
        hub_frequency = Counter(str(article["primary_hub"]) for article in articles)
        selected_articles.extend(
            sorted(articles, key=lambda article: _article_rank(article, hub_frequency, active_config))[: quotas[flow]]
        )

    split_by_article = _assign_splits(selected_articles, active_config)
    intent_by_article = _assign_intents(selected_articles, active_config)
    records: list[dict[str, Any]] = []
    for article in selected_articles:
        article_id = str(article["article_id"])
        selected_passages = _choose_passages(article, active_config)
        for slot_index, (passage, requested_intent) in enumerate(
            zip(selected_passages, intent_by_article[article_id], strict=True)
        ):
            record = {key: value for key, value in passage.items() if key != "quality"}
            record.update(_quality_columns(passage["quality"]))
            record.update(
                {
                    "split": split_by_article[article_id],
                    "requested_intent": requested_intent,
                    "slot_index": slot_index,
                }
            )
            records.append(record)

    targets = pl.DataFrame(records).sort(["article_id", "slot_index"])
    if active_config.output_dir is not None:
        output_dir = Path(active_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "targets.parquet"
        temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
        targets.write_parquet(temporary, compression="zstd")
        os.replace(temporary, output_path)
        summary_path = output_dir / "target_selection_summary.json"
        summary_temporary = summary_path.with_name(f".{summary_path.name}.{uuid.uuid4().hex}.tmp")
        summary_temporary.write_text(
            json.dumps(
                {
                    "version": 1,
                    "passage_count": len(passage_rows),
                    "selected_target_count": targets.height,
                    "quality_reason_counts": dict(sorted(quality_reason_counts.items())),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(summary_temporary, summary_path)
    return targets


__all__ = [
    "PassageQuality",
    "TargetSelectionConfig",
    "score_passage_quality",
    "select_target_slots",
    "selection_quality_reason_counts",
]
