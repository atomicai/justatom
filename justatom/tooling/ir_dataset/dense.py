from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class TextEncoder(Protocol):
    dimension: int
    model_name: str
    device: str

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class DenseSearchHit:
    passage_id: str
    score: float
    rank: int


def _normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a rank-2 embedding matrix, got shape={values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Dense encoder returned a zero-length embedding")
    return values / norms


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class E5TextEncoder:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str = "mps",
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        self.model_name = str(model_name)
        self.device = str(device)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).eval().to(self.device)
        self.dimension = int(self.model.config.hidden_size)

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        import torch

        if batch_size <= 0:
            raise ValueError("dense batch_size must be > 0")
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = [str(text) for text in texts[start : start + batch_size]]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                outputs.append(pooled.float().cpu().numpy())
        if not outputs:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


class DenseIndex:
    def __init__(
        self,
        *,
        output_dir: Path,
        passage_ids: Sequence[str],
        count: int,
        dimension: int,
        model_name: str,
        encoder: TextEncoder | None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.embeddings_path = self.output_dir / "embeddings.f32"
        self.passage_ids = tuple(str(value) for value in passage_ids)
        self.count = int(count)
        self.dimension = int(dimension)
        self.model_name = str(model_name)
        self.encoder = encoder
        self._id_to_index = {passage_id: index for index, passage_id in enumerate(self.passage_ids)}
        self._embeddings = np.memmap(
            self.embeddings_path,
            dtype=np.float32,
            mode="r",
            shape=(self.count, self.dimension),
        )

    @classmethod
    def build(
        cls,
        rows: Iterable[tuple[str, str]],
        output_dir: Path,
        encoder: TextEncoder,
        batch_size: int = 64,
    ) -> "DenseIndex":
        if batch_size <= 0:
            raise ValueError("dense batch_size must be > 0")
        passage_ids: list[str] = []
        texts: list[str] = []
        seen: set[str] = set()
        for raw_id, raw_text in rows:
            passage_id = str(raw_id)
            if passage_id in seen:
                raise ValueError(f"duplicate passage_id in dense corpus: {passage_id}")
            seen.add(passage_id)
            passage_ids.append(passage_id)
            texts.append(str(raw_text))
        if not passage_ids:
            raise ValueError("Dense corpus must not be empty")

        output_dir = Path(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
        temporary.mkdir(parents=True)
        embeddings_path = temporary / "embeddings.f32"
        matrix: np.memmap | None = None
        dimension: int | None = None
        try:
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = _normalize(encoder.encode(batch, batch_size=batch_size))
                if encoded.shape[0] != len(batch):
                    raise ValueError("Dense encoder returned a different number of rows than requested")
                if dimension is None:
                    dimension = int(encoded.shape[1])
                    expected_dimension = int(getattr(encoder, "dimension", dimension))
                    if dimension != expected_dimension:
                        raise ValueError(f"Dense encoder dimension mismatch: declared={expected_dimension}, returned={dimension}")
                    matrix = np.memmap(
                        embeddings_path,
                        dtype=np.float32,
                        mode="w+",
                        shape=(len(texts), dimension),
                    )
                elif encoded.shape[1] != dimension:
                    raise ValueError("Dense encoder changed embedding dimension between batches")
                assert matrix is not None
                matrix[start : start + len(batch)] = encoded
            assert matrix is not None and dimension is not None
            matrix.flush()
            del matrix

            (temporary / "passage_ids.json").write_text(
                json.dumps(passage_ids, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            metadata = {
                "version": 1,
                "count": len(passage_ids),
                "dimension": dimension,
                "dtype": "float32",
                "model_name": str(getattr(encoder, "model_name", "unknown")),
                "build_device": str(getattr(encoder, "device", "unknown")),
                "embeddings_sha256": _sha256_file(embeddings_path),
            }
            (temporary / "dense_config.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if output_dir.exists():
                shutil.rmtree(output_dir)
            os.replace(temporary, output_dir)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

        return cls.load(output_dir, encoder=encoder)

    @classmethod
    def load(cls, output_dir: Path, encoder: TextEncoder | None = None) -> "DenseIndex":
        output_dir = Path(output_dir)
        metadata = json.loads((output_dir / "dense_config.json").read_text(encoding="utf-8"))
        passage_ids = json.loads((output_dir / "passage_ids.json").read_text(encoding="utf-8"))
        if len(passage_ids) != int(metadata["count"]):
            raise ValueError("Dense passage ID count does not match metadata")
        return cls(
            output_dir=output_dir,
            passage_ids=passage_ids,
            count=int(metadata["count"]),
            dimension=int(metadata["dimension"]),
            model_name=str(metadata["model_name"]),
            encoder=encoder,
        )

    def embedding_rows(self, indices: Sequence[int]) -> np.ndarray:
        return np.asarray(self._embeddings[list(indices)], dtype=np.float32)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            try:
                import torch

                return "mps" if torch.backends.mps.is_available() else "cpu"
            except ImportError:
                return "cpu"
        if device == "mps":
            try:
                import torch

                return "mps" if torch.backends.mps.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return "cpu"

    def _score_block(self, queries: np.ndarray, corpus: np.ndarray, device: str) -> np.ndarray:
        if device == "mps":
            import torch

            query_tensor = torch.from_numpy(np.ascontiguousarray(queries)).to("mps")
            corpus_tensor = torch.from_numpy(np.ascontiguousarray(corpus)).to("mps")
            return (query_tensor @ corpus_tensor.T).float().cpu().numpy()
        return queries @ corpus.T

    def search_embeddings(
        self,
        query_embeddings: np.ndarray,
        *,
        k: int = 20,
        block_size: int = 65_536,
        exclude_ids: Sequence[str | None] | None = None,
        device: str = "auto",
    ) -> list[list[DenseSearchHit]]:
        if k <= 0:
            raise ValueError("dense k must be > 0")
        if block_size <= 0:
            raise ValueError("dense block_size must be > 0")
        queries = _normalize(np.asarray(query_embeddings, dtype=np.float32))
        if queries.shape[1] != self.dimension:
            raise ValueError(f"Dense query dimension mismatch: expected={self.dimension}, got={queries.shape[1]}")
        exclusions = list(exclude_ids or [None] * len(queries))
        if len(exclusions) != len(queries):
            raise ValueError("exclude_ids must contain one value per query")
        excluded_indices = [self._id_to_index.get(value) if value is not None else None for value in exclusions]
        active_k = min(int(k), self.count)
        best_scores = np.full((len(queries), active_k), -np.inf, dtype=np.float32)
        best_indices = np.full((len(queries), active_k), -1, dtype=np.int64)
        active_device = self._resolve_device(device)

        for start in range(0, self.count, block_size):
            end = min(start + block_size, self.count)
            corpus = np.asarray(self._embeddings[start:end], dtype=np.float32)
            scores = self._score_block(queries, corpus, active_device)
            for query_index, excluded in enumerate(excluded_indices):
                if excluded is not None and start <= excluded < end:
                    scores[query_index, excluded - start] = -np.inf

            local_k = min(active_k, end - start)
            local_positions = np.argpartition(scores, -local_k, axis=1)[:, -local_k:]
            local_scores = np.take_along_axis(scores, local_positions, axis=1)
            local_indices = local_positions.astype(np.int64) + start
            combined_scores = np.concatenate([best_scores, local_scores], axis=1)
            combined_indices = np.concatenate([best_indices, local_indices], axis=1)
            selected = np.argpartition(combined_scores, -active_k, axis=1)[:, -active_k:]
            best_scores = np.take_along_axis(combined_scores, selected, axis=1)
            best_indices = np.take_along_axis(combined_indices, selected, axis=1)

        output: list[list[DenseSearchHit]] = []
        for row_scores, row_indices in zip(best_scores, best_indices, strict=True):
            candidates = [
                (float(score), int(index))
                for score, index in zip(row_scores, row_indices, strict=True)
                if index >= 0 and np.isfinite(score)
            ]
            candidates.sort(key=lambda item: (-item[0], self.passage_ids[item[1]]))
            output.append(
                [
                    DenseSearchHit(
                        passage_id=self.passage_ids[index],
                        score=score,
                        rank=rank,
                    )
                    for rank, (score, index) in enumerate(candidates[:k], start=1)
                ]
            )
        return output

    def search_texts(
        self,
        queries: Sequence[str],
        *,
        k: int = 20,
        batch_size: int = 64,
        block_size: int = 65_536,
        device: str = "auto",
    ) -> list[list[DenseSearchHit]]:
        if self.encoder is None:
            raise RuntimeError("Dense text search requires an encoder")
        prefixed = [text if (text := str(query).strip()).casefold().startswith("query:") else f"query: {text}" for query in queries]
        embeddings = self.encoder.encode(prefixed, batch_size=batch_size)
        return self.search_embeddings(embeddings, k=k, block_size=block_size, device=device)


__all__ = ["DenseIndex", "DenseSearchHit", "E5TextEncoder", "TextEncoder"]
