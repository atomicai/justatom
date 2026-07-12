from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol

from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?…])\s+")
_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_HABR_CODE_MARKER_RE = re.compile(r"\[/?(?:code|source)(?:=[^\]\n]*)?\]", re.IGNORECASE)
CHUNKER_VERSION = 2


class TokenizerLike(Protocol):
    name_or_path: str

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        verbose: bool,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    tokenizer_name: str = "intfloat/multilingual-e5-small"
    min_chars: int = 600
    target_chars: int = 1200
    max_chars: int = 1800
    overlap_max_chars: int = 250
    model_max_tokens: int = 512
    safety_reserve_tokens: int = 8

    def __post_init__(self) -> None:
        if self.min_chars <= 0:
            raise ValueError("chunking.min_chars must be > 0")
        if not self.min_chars <= self.target_chars <= self.max_chars:
            raise ValueError("chunking character limits must satisfy min <= target <= max")
        if self.overlap_max_chars < 0:
            raise ValueError("chunking.overlap_max_chars must be >= 0")
        if self.model_max_tokens <= 0:
            raise ValueError("chunking.model_max_tokens must be > 0")
        if not 0 <= self.safety_reserve_tokens < self.model_max_tokens:
            raise ValueError("chunking.safety_reserve_tokens must be within the model token budget")

    @property
    def accepted_max_tokens(self) -> int:
        return self.model_max_tokens - self.safety_reserve_tokens


@dataclass(frozen=True, slots=True)
class StructuralUnit:
    section: str
    text: str
    kind: str
    source_index: int


@dataclass(frozen=True, slots=True)
class Passage:
    passage_id: str
    article_id: str
    title: str
    section: str
    content: str
    serialized_passage: str
    char_count: int
    token_count: int
    overlap_prefix_chars: int
    source_hash: str
    url: str = ""
    flows: tuple[str, ...] = ()
    hubs: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    start_unit: int = 0
    end_unit: int = 0


def _clean_text(text: str, *, preserve_newlines: bool = False) -> str:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in normalized.splitlines()]
    if preserve_newlines:
        return _BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()
    return " ".join(line for line in lines if line).strip()


def _normalize_labels(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    return tuple(text for item in values if (text := _clean_text(str(item))))


def serialize_passage(title: str, section: str, content: str) -> str:
    header = [f"passage: {_clean_text(title)}"]
    normalized_section = _clean_text(section)
    if normalized_section and normalized_section.casefold() != _clean_text(title).casefold():
        header.append(normalized_section)
    return "\n".join(header) + "\n\n" + _clean_text(content, preserve_newlines=True)


class MarkdownPassageChunker:
    def __init__(self, config: ChunkingConfig | None = None, tokenizer: TokenizerLike | None = None) -> None:
        self.config = config or ChunkingConfig()
        self.tokenizer = tokenizer or self._load_tokenizer(self.config.tokenizer_name)
        self._markdown = MarkdownIt("commonmark", {"html": False}).enable("table")

    @classmethod
    def for_tests(
        cls,
        *,
        tokenizer: TokenizerLike,
        max_tokens: int = 512,
        reserve_tokens: int = 8,
    ) -> "MarkdownPassageChunker":
        return cls(
            config=ChunkingConfig(
                tokenizer_name=getattr(tokenizer, "name_or_path", "test/tokenizer"),
                min_chars=40,
                target_chars=180,
                max_chars=320,
                overlap_max_chars=40,
                model_max_tokens=max_tokens,
                safety_reserve_tokens=reserve_tokens,
            ),
            tokenizer=tokenizer,
        )

    @staticmethod
    def _load_tokenizer(name: str) -> TokenizerLike:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name)

    def token_count(self, text: str) -> int:
        try:
            encoded = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )
        except TypeError as exc:
            if "verbose" not in str(exc):
                raise
            encoded = self.tokenizer(text, add_special_tokens=True, truncation=False)
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise ValueError("Tokenizer did not return input_ids")
        return len(input_ids)

    def _inline_text(self, node: SyntaxTreeNode) -> str:
        node_type = node.type
        if node_type in {"html_inline", "html_block"}:
            return ""
        if node_type in {"softbreak", "hardbreak"}:
            return "\n"
        if node_type in {"text", "code_inline", "code_block", "fence"}:
            return node.content
        if node_type == "image":
            return "".join(self._inline_text(child) for child in node.children)
        if node.children:
            return "".join(self._inline_text(child) for child in node.children)
        return node.content or ""

    def _block_units(self, node: SyntaxTreeNode, section: str, start_index: int) -> list[StructuralUnit]:
        if node.type in {"html_block", "hr"}:
            return []
        if node.type in {"fence", "code_block"}:
            text = _clean_text(node.content, preserve_newlines=True)
            return [StructuralUnit(section, text, "code", start_index)] if text else []
        if node.type in {"paragraph", "blockquote"}:
            text = _clean_text(self._inline_text(node), preserve_newlines=node.type == "blockquote")
            return [StructuralUnit(section, text, node.type, start_index)] if text else []
        if node.type in {"bullet_list", "ordered_list"}:
            units: list[StructuralUnit] = []
            for child in node.children:
                text = _clean_text(self._inline_text(child), preserve_newlines=True)
                if text:
                    units.append(StructuralUnit(section, text, "list_item", start_index + len(units)))
            return units
        if node.type == "table":
            rows: list[str] = []
            for child in node.walk(include_self=False):
                if child.type != "tr":
                    continue
                cells = [_clean_text(self._inline_text(cell)) for cell in child.children if cell.type in {"th", "td"}]
                row = " | ".join(cell for cell in cells if cell)
                if row:
                    rows.append(row)
            return [StructuralUnit(section, row, "table_row", start_index + idx) for idx, row in enumerate(rows)]
        units: list[StructuralUnit] = []
        for child in node.children:
            units.extend(self._block_units(child, section, start_index + len(units)))
        return units

    def parse_units(self, markdown: str) -> list[StructuralUnit]:
        source = _HABR_CODE_MARKER_RE.sub("", str(markdown or ""))
        root = SyntaxTreeNode(self._markdown.parse(source))
        section = ""
        units: list[StructuralUnit] = []
        for node in root.children:
            if node.type == "heading":
                section = _clean_text(self._inline_text(node))
                continue
            produced = self._block_units(node, section, len(units))
            units.extend(replace(unit, source_index=len(units) + idx) for idx, unit in enumerate(produced))
        return units

    def _fits(self, title: str, section: str, content: str) -> bool:
        if len(content) > self.config.max_chars:
            return False
        return self.token_count(serialize_passage(title, section, content)) <= self.config.accepted_max_tokens

    def _split_text(self, unit: StructuralUnit, title: str) -> list[StructuralUnit]:
        if self._fits(title, unit.section, unit.text):
            return [unit]
        pieces = unit.text.splitlines() if unit.kind == "code" else _SENTENCE_BOUNDARY_RE.split(unit.text)
        pieces = [_clean_text(piece, preserve_newlines=unit.kind == "code") for piece in pieces if _clean_text(piece)]
        if len(pieces) <= 1:
            words = unit.text.split()
            pieces = []
            current: list[str] = []
            for word in words:
                candidate = " ".join([*current, word])
                if current and not self._fits(title, unit.section, candidate):
                    pieces.append(" ".join(current))
                    current = [word]
                else:
                    current.append(word)
            if current:
                pieces.append(" ".join(current))

        output: list[StructuralUnit] = []
        current: list[str] = []
        separator = "\n" if unit.kind == "code" else " "
        for piece in pieces:
            candidate = separator.join([*current, piece])
            if current and not self._fits(title, unit.section, candidate):
                output.append(replace(unit, text=separator.join(current)))
                current = [piece]
            else:
                current.append(piece)
        if current:
            output.append(replace(unit, text=separator.join(current)))
        return output

    def _expanded_units(self, units: list[StructuralUnit], title: str) -> list[StructuralUnit]:
        expanded: list[StructuralUnit] = []
        for unit in units:
            expanded.extend(self._split_text(unit, title))
        return [replace(unit, source_index=index) for index, unit in enumerate(expanded)]

    def chunk_article(self, row: Mapping[str, Any]) -> list[Passage]:
        article_id = _clean_text(str(row.get("id", "")))
        title = _clean_text(str(row.get("title", "")))
        markdown = str(row.get("text_markdown", "") or "")
        if not article_id or not title or not markdown.strip():
            return []

        source_hash = hashlib.sha256(f"{title}\n{markdown}".encode("utf-8")).hexdigest()
        units = self._expanded_units(self.parse_units(markdown), title)
        groups: list[tuple[list[StructuralUnit], int]] = []
        current: list[StructuralUnit] = []
        overlap_chars = 0

        def flush() -> None:
            nonlocal current, overlap_chars
            if not current:
                return
            content = "\n\n".join(unit.text for unit in current)
            if len(content) >= self.config.min_chars:
                groups.append((list(current), overlap_chars))
            previous = current[-1]
            if self.config.overlap_max_chars > 0 and len(previous.text) <= self.config.overlap_max_chars:
                current = [previous]
                overlap_chars = len(previous.text)
            else:
                current = []
                overlap_chars = 0

        for unit in units:
            if current and unit.section != current[-1].section:
                flush()
            candidate_units = [*current, unit]
            candidate = "\n\n".join(item.text for item in candidate_units)
            current_content = "\n\n".join(item.text for item in current)
            should_flush = bool(
                current
                and (
                    not self._fits(title, unit.section, candidate)
                    or (len(current_content) >= self.config.target_chars and len(candidate) > self.config.target_chars)
                )
            )
            if should_flush:
                flush()
                if current and unit.section != current[-1].section:
                    current = []
                    overlap_chars = 0
                candidate_units = [*current, unit]
                candidate = "\n\n".join(item.text for item in candidate_units)
            if not self._fits(title, unit.section, candidate):
                current = [unit]
                overlap_chars = 0
            else:
                current = candidate_units
        flush()

        passages: list[Passage] = []
        for group, overlap_prefix_chars in groups:
            section = group[-1].section
            content = "\n\n".join(unit.text for unit in group)
            serialized = serialize_passage(title, section, content)
            token_count = self.token_count(serialized)
            if token_count > self.config.accepted_max_tokens:
                continue
            identity = json.dumps(
                [article_id, section, group[0].source_index, group[-1].source_index, hashlib.sha256(content.encode()).hexdigest()],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            passages.append(
                Passage(
                    passage_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                    article_id=article_id,
                    title=title,
                    section=section,
                    content=content,
                    serialized_passage=serialized,
                    char_count=len(content),
                    token_count=token_count,
                    overlap_prefix_chars=overlap_prefix_chars,
                    source_hash=source_hash,
                    url=_clean_text(str(row.get("url", ""))),
                    flows=_normalize_labels(row.get("flows")),
                    hubs=_normalize_labels(row.get("hubs")),
                    tags=_normalize_labels(row.get("tags")),
                    start_unit=group[0].source_index,
                    end_unit=group[-1].source_index,
                )
            )
        return passages


__all__ = [
    "CHUNKER_VERSION",
    "ChunkingConfig",
    "MarkdownPassageChunker",
    "Passage",
    "StructuralUnit",
    "serialize_passage",
]
