from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import torch

from justatom.training.config import RerankerConfig


@dataclass(frozen=True)
class RerankerScores:
    """Scores and cache telemetry for one collection of query-document pairs."""

    values: tuple[float | None, ...]
    cache_hits: int
    cache_misses: int
    skipped: int


class PairScoringBackend(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def score_pairs(self, queries: Sequence[str], documents: Sequence[str]) -> list[float]: ...

    def close(self) -> None: ...


class RerankerScoreCache:
    """Small SQLite cache keyed by the complete scoring contract and pair text."""

    def __init__(self, path: str | Path, *, read_only: bool = False):
        self.path = Path(path).expanduser()
        if read_only:
            if not self.path.is_file():
                raise FileNotFoundError(f"reranker read-only cache does not exist: {self.path}")
            self.connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
            self.connection.execute("CREATE TABLE IF NOT EXISTS scores (cache_key TEXT PRIMARY KEY, score REAL NOT NULL)")
            self.connection.commit()
        self.read_only = read_only

    def get_many(self, keys: Sequence[str]) -> dict[str, float]:
        if not keys:
            return {}
        unique = list(dict.fromkeys(keys))
        result: dict[str, float] = {}
        for start in range(0, len(unique), 900):
            chunk = unique[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT cache_key, score FROM scores WHERE cache_key IN ({placeholders})",  # noqa: S608
                chunk,
            )
            result.update((str(key), float(score)) for key, score in rows)
        return result

    def put_many(self, values: dict[str, float]) -> None:
        if self.read_only:
            raise RuntimeError("cannot write to a read-only reranker cache")
        if not values:
            return
        self.connection.executemany(
            "INSERT OR REPLACE INTO scores(cache_key, score) VALUES (?, ?)",
            values.items(),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


class TransformersPairScorer:
    """Lazy local scorer for Qwen3-style generative yes/no rerankers."""

    _SCORING_CONTRACT = "qwen3-generative-yes-no-v1"
    _PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
        'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, config: RerankerConfig):
        self.config = config
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._device: torch.device | None = None
        self._prefix_tokens: list[int] | None = None
        self._suffix_tokens: list[int] | None = None
        self._true_token_id: int | None = None
        self._false_token_id: int | None = None

    @property
    def fingerprint(self) -> str:
        return json.dumps(
            {
                "backend": self.config.backend,
                "contract": self._SCORING_CONTRACT,
                "model": self.config.model_name_or_path,
                "revision": self.config.revision,
                "instruction": self.config.instruction,
                "max_length": self.config.max_length,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _resolve_device(self) -> torch.device:
        if self.config.device != "auto":
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_dtype(self, device: torch.device) -> torch.dtype:
        values = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.config.dtype != "auto":
            return values[self.config.dtype]
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device.type == "mps":
            return torch.float16
        return torch.float32

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._resolve_device()
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            revision=self.config.revision,
            local_files_only=self.config.local_files_only,
            padding_side="left",
        )
        tokenizer.pad_token = tokenizer.eos_token
        model: Any = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            revision=self.config.revision,
            local_files_only=self.config.local_files_only,
            dtype=self._resolve_dtype(device),
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()

        prefix_tokens = tokenizer.encode(self._PREFIX, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(self._SUFFIX, add_special_tokens=False)
        if len(prefix_tokens) + len(suffix_tokens) >= self.config.max_length:
            raise ValueError("reranker.max_length is too small for the Qwen3 scoring template")
        true_tokens = tokenizer("yes", add_special_tokens=False).input_ids
        false_tokens = tokenizer("no", add_special_tokens=False).input_ids
        if len(true_tokens) != 1 or len(false_tokens) != 1:
            raise RuntimeError("reranker yes/no labels must each map to exactly one token")

        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._prefix_tokens = list(prefix_tokens)
        self._suffix_tokens = list(suffix_tokens)
        self._true_token_id = int(true_tokens[0])
        self._false_token_id = int(false_tokens[0])

    def _format_pair(self, query: str, document: str) -> str:
        return f"<Instruct>: {self.config.instruction}\n<Query>: {query}\n<Document>: {document}"

    @torch.inference_mode()
    def score_pairs(self, queries: Sequence[str], documents: Sequence[str]) -> list[float]:
        if len(queries) != len(documents):
            raise ValueError("queries and documents must have the same length")
        if not queries:
            return []
        self._load()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._device is not None
        assert self._prefix_tokens is not None
        assert self._suffix_tokens is not None
        assert self._true_token_id is not None
        assert self._false_token_id is not None

        scores: list[float] = []
        payload = [self._format_pair(str(query), str(document)) for query, document in zip(queries, documents)]
        body_length = self.config.max_length - len(self._prefix_tokens) - len(self._suffix_tokens)
        for start in range(0, len(payload), self.config.batch_size):
            encoded = self._tokenizer(
                payload[start : start + self.config.batch_size],
                padding=False,
                truncation=True,
                max_length=body_length,
                return_attention_mask=False,
            )
            encoded["input_ids"] = [
                self._prefix_tokens + list(token_ids) + self._suffix_tokens for token_ids in encoded["input_ids"]
            ]
            inputs = self._tokenizer.pad(encoded, padding=True, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            logits = self._model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
            binary_logits = torch.stack(
                (logits[:, self._false_token_id], logits[:, self._true_token_id]),
                dim=1,
            )
            batch_scores = torch.softmax(binary_logits.float(), dim=1)[:, 1].detach().cpu().tolist()
            if not all(math.isfinite(value) for value in batch_scores):
                raise RuntimeError("reranker returned a non-finite score")
            scores.extend(float(value) for value in batch_scores)
        return scores

    def close(self) -> None:
        device = self._device
        self._model = None
        self._tokenizer = None
        if device is not None and device.type == "cuda":
            torch.cuda.empty_cache()


class CachedTextReranker:
    """Pair scorer with deterministic content-addressed persistent caching."""

    def __init__(self, config: RerankerConfig, backend: PairScoringBackend | None = None):
        self.config = config
        self.backend = TransformersPairScorer(config) if backend is None else backend
        cache_config = config.cache
        self.cache = (
            None
            if cache_config.mode == "off"
            else RerankerScoreCache(cache_config.path, read_only=cache_config.mode == "read-only")
        )

    def _cache_key(self, query: str, document: str) -> str:
        payload = json.dumps(
            [self.backend.fingerprint, query, document],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def score_pairs(self, queries: Sequence[str], documents: Sequence[str]) -> RerankerScores:
        if len(queries) != len(documents):
            raise ValueError("queries and documents must have the same length")
        pairs = [(str(query), str(document)) for query, document in zip(queries, documents)]
        keys = [self._cache_key(query, document) for query, document in pairs]
        cached = {} if self.cache is None else self.cache.get_many(keys)
        values: list[float | None] = [cached.get(key) for key in keys]
        missing_indices = [index for index, value in enumerate(values) if value is None]
        hits = len(values) - len(missing_indices)

        if missing_indices and self.config.cache.on_miss == "error":
            raise RuntimeError(f"reranker cache miss for {len(missing_indices)} pair(s)")
        if missing_indices and self.config.cache.on_miss == "score":
            unique_missing: dict[str, tuple[str, str]] = {}
            for index in missing_indices:
                unique_missing.setdefault(keys[index], pairs[index])
            missing_keys = list(unique_missing)
            missing_pairs = [unique_missing[key] for key in missing_keys]
            scored = self.backend.score_pairs(
                [query for query, _ in missing_pairs],
                [document for _, document in missing_pairs],
            )
            if len(scored) != len(missing_keys):
                raise RuntimeError("reranker backend returned an unexpected number of scores")
            resolved = dict(zip(missing_keys, scored))
            if self.cache is not None:
                self.cache.put_many(resolved)
            values = [resolved.get(key, value) for key, value in zip(keys, values)]

        skipped = sum(value is None for value in values)
        return RerankerScores(
            values=tuple(values),
            cache_hits=hits,
            cache_misses=len(missing_indices),
            skipped=skipped,
        )

    def close(self) -> None:
        if self.cache is not None:
            self.cache.close()
        self.backend.close()


def build_reranker(config: RerankerConfig) -> CachedTextReranker | None:
    return CachedTextReranker(config) if config.enabled else None


__all__ = [
    "CachedTextReranker",
    "PairScoringBackend",
    "RerankerScoreCache",
    "RerankerScores",
    "TransformersPairScorer",
    "build_reranker",
]
