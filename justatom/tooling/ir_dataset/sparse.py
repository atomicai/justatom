from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import bm25s
from bm25s.tokenization import Tokenizer


TECHNICAL_TOKEN_PATTERN = r"(?u)[\w][\w.+#-]*"
RUSSIAN_STOPWORDS = (
    "а",
    "без",
    "бы",
    "был",
    "была",
    "были",
    "в",
    "во",
    "для",
    "до",
    "его",
    "ее",
    "если",
    "и",
    "из",
    "или",
    "их",
    "к",
    "как",
    "на",
    "не",
    "но",
    "о",
    "от",
    "по",
    "при",
    "с",
    "со",
    "то",
    "у",
    "что",
    "это",
)


@dataclass(frozen=True, slots=True)
class SearchHit:
    passage_id: str
    score: float
    rank: int


class BM25Index:
    def __init__(
        self,
        *,
        retriever: bm25s.BM25,
        tokenizer: Tokenizer,
        passage_ids: Sequence[str],
        output_dir: Path,
    ) -> None:
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.passage_ids = tuple(str(value) for value in passage_ids)
        self.output_dir = Path(output_dir)

    @classmethod
    def build(cls, rows: Iterable[tuple[str, str]], output_dir: Path) -> "BM25Index":
        passage_ids: list[str] = []
        texts: list[str] = []
        seen: set[str] = set()
        for raw_id, raw_text in rows:
            passage_id = str(raw_id)
            if passage_id in seen:
                raise ValueError(f"duplicate passage_id in BM25 corpus: {passage_id}")
            seen.add(passage_id)
            passage_ids.append(passage_id)
            texts.append(str(raw_text))
        if not passage_ids:
            raise ValueError("BM25 corpus must not be empty")

        tokenizer = Tokenizer(
            lower=True,
            splitter=TECHNICAL_TOKEN_PATTERN,
            stopwords=list(RUSSIAN_STOPWORDS),
            stemmer=None,
        )
        corpus_tokens = tokenizer.tokenize(texts, return_as="string", show_progress=False)
        retriever = bm25s.BM25(method="lucene", k1=1.2, b=0.75, dtype="float32")
        retriever.index(corpus_tokens, show_progress=False)

        output_dir = Path(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
        temporary.mkdir(parents=True)
        try:
            retriever.save(temporary, show_progress=False)
            tokenizer.save_stopwords(str(temporary))
            (temporary / "passage_ids.json").write_text(
                json.dumps(passage_ids, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            (temporary / "retrieval_config.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "method": "lucene",
                        "k1": 1.2,
                        "b": 0.75,
                        "token_pattern": TECHNICAL_TOKEN_PATTERN,
                        "count": len(passage_ids),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if output_dir.exists():
                shutil.rmtree(output_dir)
            os.replace(temporary, output_dir)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        return cls(retriever=retriever, tokenizer=tokenizer, passage_ids=passage_ids, output_dir=output_dir)

    @classmethod
    def load(cls, output_dir: Path, mmap: bool = True) -> "BM25Index":
        output_dir = Path(output_dir)
        passage_ids = json.loads((output_dir / "passage_ids.json").read_text(encoding="utf-8"))
        tokenizer = Tokenizer(
            lower=True,
            splitter=TECHNICAL_TOKEN_PATTERN,
            stopwords=[],
            stemmer=None,
        )
        tokenizer.load_stopwords(str(output_dir))
        retriever = bm25s.BM25.load(output_dir, mmap=bool(mmap), load_corpus=False)
        return cls(retriever=retriever, tokenizer=tokenizer, passage_ids=passage_ids, output_dir=output_dir)

    def search(self, queries: Sequence[str], k: int = 20) -> list[list[SearchHit]]:
        if k <= 0:
            raise ValueError("BM25 k must be > 0")
        query_texts = [str(query) for query in queries]
        if not query_texts:
            return []
        active_k = min(int(k), len(self.passage_ids))
        tokens = self.tokenizer.tokenize(
            query_texts,
            return_as="string",
            show_progress=False,
        )
        vocabulary = self.retriever.vocab_dict
        tokens = [[token for token in query if token and token in vocabulary] for query in tokens]
        results = self.retriever.retrieve(
            tokens,
            k=active_k,
            sorted=True,
            show_progress=False,
        )
        output: list[list[SearchHit]] = []
        for indices, scores in zip(results.documents, results.scores, strict=True):
            output.append(
                [
                    SearchHit(
                        passage_id=self.passage_ids[int(index)],
                        score=float(score),
                        rank=rank,
                    )
                    for rank, (index, score) in enumerate(zip(indices, scores, strict=True), start=1)
                ]
            )
        return output


__all__ = ["BM25Index", "RUSSIAN_STOPWORDS", "SearchHit", "TECHNICAL_TOKEN_PATTERN"]
