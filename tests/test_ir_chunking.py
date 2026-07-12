from __future__ import annotations

from pathlib import Path

from justatom.tooling.ir_dataset.chunking import ChunkingConfig, MarkdownPassageChunker


FIXTURE = Path(__file__).parent / "fixtures" / "habr_article.md"


class WhitespaceTokenizer:
    name_or_path = "test/whitespace"

    def __call__(self, text, *, add_special_tokens=True, truncation=False):
        assert truncation is False
        count = len(str(text).split()) + (2 if add_special_tokens else 0)
        return {"input_ids": list(range(count))}


class VerboseAwareTokenizer(WhitespaceTokenizer):
    def __init__(self):
        self.verbose = None

    def __call__(self, text, *, add_special_tokens=True, truncation=False, verbose=True):
        self.verbose = verbose
        return super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
        )


def sample_article() -> dict:
    return {
        "id": 101,
        "title": "Надёжная очередь заданий",
        "text_markdown": FIXTURE.read_text(encoding="utf-8"),
        "url": "https://habr.com/ru/articles/101/",
        "flows": ["develop"],
        "hubs": ["distributed_systems"],
        "tags": ["очереди", "надёжность"],
    }


def test_markdown_becomes_plain_structural_units():
    chunker = MarkdownPassageChunker.for_tests(tokenizer=WhitespaceTokenizer())

    units = chunker.parse_units(FIXTURE.read_text(encoding="utf-8"))
    text = "\n".join(unit.text for unit in units)

    assert "[документация клиента](" not in text
    assert "https://example.invalid" not in text
    assert "```" not in text
    assert "def configure_client" in text
    assert "схема компонентов" in text
    assert "Архитектура сервиса" in {unit.section for unit in units}
    assert "Диагностика" in {unit.section for unit in units}


def test_passages_fit_complete_serialized_token_budget():
    config = ChunkingConfig(
        tokenizer_name="test/whitespace",
        min_chars=40,
        target_chars=180,
        max_chars=320,
        overlap_max_chars=40,
        model_max_tokens=40,
        safety_reserve_tokens=4,
    )
    chunker = MarkdownPassageChunker(config=config, tokenizer=WhitespaceTokenizer())

    passages = chunker.chunk_article(sample_article())

    assert passages
    assert all(row.token_count <= config.accepted_max_tokens for row in passages)
    assert all(row.token_count <= config.model_max_tokens for row in passages)
    assert all(row.serialized_passage.startswith("passage: Надёжная очередь заданий") for row in passages)


def test_passage_ids_are_stable_and_source_sensitive():
    chunker = MarkdownPassageChunker.for_tests(tokenizer=WhitespaceTokenizer())

    first = chunker.chunk_article(sample_article())
    second = chunker.chunk_article(sample_article())
    changed = chunker.chunk_article({**sample_article(), "text_markdown": "Совсем другой материал. " * 80})

    assert [row.passage_id for row in first] == [row.passage_id for row in second]
    assert [row.passage_id for row in first] != [row.passage_id for row in changed]
    assert len({row.passage_id for row in first}) == len(first)


def test_code_content_is_preserved_without_fence_or_language_marker():
    chunker = MarkdownPassageChunker.for_tests(tokenizer=WhitespaceTokenizer())

    passages = chunker.chunk_article(sample_article())
    text = "\n".join(row.content for row in passages)

    assert "def configure_client(timeout):" in text
    assert "legacy_call()" in text
    assert "[code]" not in text
    assert "[/code]" not in text
    assert "```python" not in text


def test_token_count_disables_tokenizer_length_warning():
    tokenizer = VerboseAwareTokenizer()
    chunker = MarkdownPassageChunker.for_tests(tokenizer=tokenizer)

    chunker.token_count("длинный текст " * 600)

    assert tokenizer.verbose is False
