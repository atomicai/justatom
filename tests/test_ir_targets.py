from __future__ import annotations

import json

import polars as pl
import pytest

from justatom.tooling.ir_dataset.targets import TargetSelectionConfig, score_passage_quality, select_target_slots


def prose_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "article_id": "article-prose",
        "passage_id": "passage-prose",
        "title": "Надежная доставка сообщений",
        "section": "Подтверждение обработки",
        "content": " ".join(
            ["Обработчик сначала сохраняет результат в устойчивом хранилище, затем подтверждает сообщение брокеру."] * 20
        ),
        "token_count": 160,
        "flows": ["develop"],
        "hubs": ["distributed_systems"],
        "tags": ["python"],
        "start_unit": 0,
        "end_unit": 0,
    }
    row.update(overrides)
    return row


def code_blob_row() -> dict[str, object]:
    return prose_row(
        article_id="article-code",
        passage_id="passage-code",
        content="\n".join(["{{{{ handler::process(item) => value_[index] += 1; }}}}" for _ in range(30)]),
        token_count=120,
    )


def target_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for article_index, flow in enumerate(("develop", "admin", "science"), start=1):
        article_id = f"article-{article_index}"
        for passage_index, section in enumerate(("Введение", "Практика", "Проверка"), start=1):
            rows.append(
                prose_row(
                    article_id=article_id,
                    passage_id=f"{article_id}-passage-{passage_index}",
                    section=section,
                    content=str(prose_row()["content"]) + f" Деталь {article_index}-{passage_index}.",
                    flows=[flow],
                    hubs=[f"{flow}-hub"],
                    start_unit=passage_index - 1,
                    end_unit=passage_index - 1,
                )
            )
    return pl.DataFrame(rows)


def test_quality_rejects_code_blob_but_keeps_explanatory_prose():
    prose = score_passage_quality(prose_row())
    code = score_passage_quality(code_blob_row())

    assert prose.eligible
    assert prose.reason_codes == ()
    assert prose.alpha_ratio >= 0.55
    assert not code.eligible
    assert "symbol_ratio_gt_0.18" in code.reason_codes
    assert code.symbol_ratio > 0.18


def test_quality_preserves_rejection_features_and_reason_codes():
    quality = score_passage_quality(
        prose_row(
            content=("А" * 81) + " " + "идентификатор_" + ("x" * 32),
            token_count=79,
        )
    )

    assert not quality.eligible
    assert "token_count_lt_80" in quality.reason_codes
    assert "long_token_ratio_gt_0.05" in quality.reason_codes
    assert "repeated_character_run_gt_80" in quality.reason_codes
    assert quality.max_repeated_run == 81
    assert quality.long_token_ratio > 0.05


def test_selection_is_article_safe_and_assigns_two_intents(tmp_path):
    selected = select_target_slots(
        target_frame(),
        TargetSelectionConfig(article_count=3, output_dir=tmp_path),
    )

    assert selected.height == 6
    assert selected.group_by("article_id").len()["len"].min() == 2
    assert selected.group_by("article_id").agg(pl.col("split").n_unique())["split"].max() == 1
    assert selected.group_by("article_id").agg(pl.col("requested_intent").n_unique())["requested_intent"].min() == 2
    assert selected.group_by("article_id").agg(pl.col("passage_id").n_unique())["passage_id"].min() == 2
    assert selected.group_by("article_id").agg(pl.col("section").n_unique())["section"].min() == 2
    assert (tmp_path / "targets.parquet").exists()
    assert pl.read_parquet(tmp_path / "targets.parquet").equals(selected)


def test_selection_uses_deterministic_flow_and_intent_balancing():
    selected = select_target_slots(target_frame(), TargetSelectionConfig(article_count=3, seed=42))
    repeated = select_target_slots(target_frame(), TargetSelectionConfig(article_count=3, seed=42))

    assert selected.equals(repeated)
    assert selected.group_by("primary_flow").len()["len"].max() <= 2
    assert set(selected["requested_intent"]) <= {
        "how_to",
        "why",
        "troubleshooting",
        "concept",
        "comparison",
        "requirements",
        "limitations",
        "factual",
    }


def test_selection_rejects_an_infeasible_primary_flow_cap():
    rows = []
    for article_index in range(10):
        for passage_index in range(2):
            rows.append(
                prose_row(
                    article_id=f"article-{article_index}",
                    passage_id=f"article-{article_index}-passage-{passage_index}",
                    content=str(prose_row()["content"]) + f" Деталь {article_index}-{passage_index}.",
                    flows=["develop"],
                    start_unit=passage_index,
                    end_unit=passage_index,
                )
            )

    with pytest.raises(ValueError, match="cannot satisfy.*max_flow_share=0.30"):
        select_target_slots(pl.DataFrame(rows), TargetSelectionConfig(article_count=10))


def test_selection_prefers_two_distinct_non_empty_sections_when_available():
    selected = select_target_slots(
        pl.DataFrame(
            [
                prose_row(article_id="article-1", passage_id="empty", section=""),
                prose_row(
                    article_id="article-1",
                    passage_id="implementation",
                    section="Implementation",
                    content=prose_row()["content"] + " (implementation)",
                ),
                prose_row(
                    article_id="article-1",
                    passage_id="validation",
                    section="Validation",
                    content=prose_row()["content"] + " (validation)",
                ),
            ]
        ),
        TargetSelectionConfig(article_count=1),
    )

    assert set(selected["section"]) == {"Implementation", "Validation"}


def test_selection_requires_each_target_to_have_a_full_article_structural_sibling():
    passages = pl.DataFrame(
        [
            prose_row(article_id="article-1", passage_id="target-1", content=str(prose_row()["content"]) + " one"),
            prose_row(
                article_id="article-1",
                passage_id="target-2",
                content=str(prose_row()["content"]) + " two",
                start_unit=4,
                end_unit=4,
            ),
        ]
    )

    with pytest.raises(ValueError, match="only 0 have two eligible passages"):
        select_target_slots(passages, TargetSelectionConfig(article_count=1))


def test_selection_accepts_noneligible_full_article_structural_siblings():
    passages = pl.DataFrame(
        [
            prose_row(article_id="article-1", passage_id="target-1", content=str(prose_row()["content"]) + " one"),
            prose_row(
                article_id="article-1",
                passage_id="sibling-1",
                content="x = 1",
                token_count=3,
                start_unit=1,
                end_unit=1,
            ),
            prose_row(
                article_id="article-1",
                passage_id="sibling-2",
                content="x = 2",
                token_count=3,
                start_unit=3,
                end_unit=3,
            ),
            prose_row(
                article_id="article-1",
                passage_id="target-2",
                content=str(prose_row()["content"]) + " two",
                start_unit=4,
                end_unit=4,
            ),
        ]
    )

    selected = select_target_slots(passages, TargetSelectionConfig(article_count=1))

    assert set(selected["passage_id"]) == {"target-1", "target-2"}
    assert not score_passage_quality(passages.filter(pl.col("passage_id") == "sibling-1").to_dicts()[0]).eligible


def test_selection_excludes_globally_duplicate_normalized_content_with_reason_count(tmp_path):
    rows = target_frame().to_dicts()
    duplicate_content = rows[0]["content"]
    rows[1]["content"] = f"  {duplicate_content}  "
    rows[1]["passage_id"] = "different-id-same-content"

    selected = select_target_slots(
        pl.DataFrame(rows),
        TargetSelectionConfig(article_count=2, output_dir=tmp_path, max_flow_share=1.0),
    )
    summary = json.loads((tmp_path / "target_selection_summary.json").read_text(encoding="utf-8"))

    assert rows[0]["passage_id"] not in selected["passage_id"].to_list()
    assert "different-id-same-content" not in selected["passage_id"].to_list()
    assert summary["quality_reason_counts"]["globally_duplicate_content"] == 2
