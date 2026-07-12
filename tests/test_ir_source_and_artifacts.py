from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from justatom.tooling.ir_dataset.artifacts import PrepareConfig, prepare_passages
from justatom.tooling.ir_dataset.chunking import CHUNKER_VERSION, ChunkingConfig, MarkdownPassageChunker
from justatom.tooling.ir_dataset.source import HABR_SOURCE_COLUMNS, HabrSource, promote_hf_token_env
from justatom.tooling.ir_dataset import source as source_module


class WhitespaceTokenizer:
    name_or_path = "test/whitespace"

    def __call__(self, text, *, add_special_tokens=True, truncation=False):
        assert truncation is False
        count = len(str(text).split()) + (2 if add_special_tokens else 0)
        return {"input_ids": list(range(count))}


def make_test_chunker() -> MarkdownPassageChunker:
    return MarkdownPassageChunker(
        config=ChunkingConfig(
            tokenizer_name="test/whitespace",
            min_chars=40,
            target_chars=140,
            max_chars=260,
            overlap_max_chars=30,
            model_max_tokens=80,
            safety_reserve_tokens=4,
        ),
        tokenizer=WhitespaceTokenizer(),
    )


def source_row(article_id: int, *, language: str = "ru", row_type: str = "article") -> dict:
    return {
        "id": article_id,
        "language": language,
        "url": f"https://habr.com/ru/articles/{article_id}/",
        "title": f"Техническая статья {article_id}",
        "text_markdown": (
            "# Раздел\n\n"
            + "Очередь обрабатывает сообщение и сохраняет результат перед подтверждением. " * 12
            + "\n\n## Диагностика\n\n"
            + "При повторной доставке обработчик проверяет сохранённое состояние. " * 10
        ),
        "type": row_type,
        "time_published": 1_700_000_000 + article_id,
        "statistics": {"commentsCount": 1, "favoritesCount": 2},
        "labels": [],
        "hubs": ["distributed_systems"],
        "flows": ["develop", "admin"],
        "tags": ["очереди", "python"],
        "reading_time": 5,
        "format": "tutorial",
        "complexity": "medium",
    }


def fixture_parquet(tmp_path: Path) -> Path:
    row = {**source_row(1), "comments": [{"message_markdown": "Не читать"}], "text_html": "<p>html</p>"}
    path = tmp_path / "train-00000-of-00001.parquet"
    pl.DataFrame([row]).write_parquet(path)
    return path


def synthetic_rows():
    yield source_row(2)
    yield source_row(1)
    yield source_row(3, language="en")
    yield source_row(4, row_type="news")


def test_habr_source_projects_required_columns_without_comments(tmp_path, monkeypatch):
    source = HabrSource(repo_id="justatom/habr-ds", cache_dir=tmp_path)
    monkeypatch.setattr(source, "_iter_parquet_paths", lambda: iter([fixture_parquet(tmp_path)]))

    row = next(source.iter_rows(limit=1))

    assert set(row) == set(HABR_SOURCE_COLUMNS)
    assert "comments" not in row
    assert "text_html" not in row


def test_habr_source_downloads_shards_lazily_when_limit_is_reached(tmp_path, monkeypatch):
    paths = {}
    for index in range(2):
        path = tmp_path / f"train-{index:05d}-of-00002.parquet"
        pl.DataFrame([source_row(index + 1)]).write_parquet(path)
        paths[f"data/{path.name}"] = path
    downloads = []
    source = HabrSource(repo_id="justatom/habr-ds", cache_dir=tmp_path)
    monkeypatch.setattr(source, "_matching_parquet_files", lambda: list(paths))

    def fake_download(*, filename, **kwargs):
        downloads.append(filename)
        return str(paths[filename])

    monkeypatch.setattr(source_module, "hf_hub_download", fake_download)

    rows = list(source.iter_rows(limit=1))

    assert len(rows) == 1
    assert downloads == ["data/train-00000-of-00002.parquet"]


def test_hf_api_key_is_promoted_for_transformers_hub_calls(monkeypatch):
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN", "HF_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_API_KEY", "hf_test_secret")

    promoted = promote_hf_token_env()

    assert promoted is True
    assert source_module.os.environ["HF_TOKEN"] == "hf_test_secret"


def test_prepare_filters_and_writes_ranked_passages(tmp_path):
    summary = prepare_passages(
        rows=synthetic_rows(),
        output_dir=tmp_path,
        chunker=make_test_chunker(),
        config=PrepareConfig(seed=42, max_passages=20, max_passages_per_article=3),
        source_fingerprint="synthetic-v1",
    )

    frame = pl.read_parquet(summary.passages_path)
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))

    assert frame["corpus_rank"].to_list() == list(range(frame.height))
    assert frame["passage_id"].n_unique() == frame.height
    assert set(frame["article_id"].to_list()) == {"1", "2"}
    assert frame.group_by("article_id").len()["len"].max() <= 3
    assert manifest["fingerprint"] == summary.fingerprint
    assert manifest["chunker_version"] == CHUNKER_VERSION
    assert manifest["counts"]["articles"] == 2
    assert manifest["counts"]["passages"] == frame.height


def test_prepare_reuses_matching_fingerprint(tmp_path):
    kwargs = {
        "output_dir": tmp_path,
        "chunker": make_test_chunker(),
        "config": PrepareConfig(seed=42, max_passages=20),
        "source_fingerprint": "synthetic-v1",
    }

    first = prepare_passages(rows=synthetic_rows(), **kwargs)
    second = prepare_passages(rows=synthetic_rows(), **kwargs)

    assert first.reused is False
    assert second.reused is True
    assert second.fingerprint == first.fingerprint
    assert second.passage_count == first.passage_count


def test_changed_config_invalidates_prepared_artifact(tmp_path):
    first = prepare_passages(
        rows=synthetic_rows(),
        output_dir=tmp_path,
        chunker=make_test_chunker(),
        config=PrepareConfig(seed=42, max_passages=20),
        source_fingerprint="synthetic-v1",
    )
    second = prepare_passages(
        rows=synthetic_rows(),
        output_dir=tmp_path,
        chunker=make_test_chunker(),
        config=PrepareConfig(seed=43, max_passages=20),
        source_fingerprint="synthetic-v1",
    )

    assert second.reused is False
    assert second.fingerprint != first.fingerprint
