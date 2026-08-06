from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from justatom.retrieval.contracts import EmbeddingProfile, apply_prefix, validate_embeddings
from justatom.retrieval.errors import ConfigurationError

_LOCAL_DEPENDENCY_ERROR = "Local embeddings require pip install 'justatom[torch]'"


def _available_devices() -> tuple[bool, bool]:
    try:
        import torch
    except ImportError as exc:
        raise ConfigurationError(_LOCAL_DEPENDENCY_ERROR) from exc

    mps = getattr(torch.backends, "mps", None)
    return torch.cuda.is_available(), bool(mps and mps.is_available())


def resolve_device(requested: str) -> str:
    device = requested.strip().lower()
    if device == "auto":
        cuda_available, mps_available = _available_devices()
        if cuda_available:
            return "cuda:0"
        if mps_available:
            return "mps"
        return "cpu"
    if device == "cpu":
        return "cpu"
    if device in {"cuda", "cuda:0"}:
        if not _available_devices()[0]:
            raise ConfigurationError(f"Requested device {requested!r} is unavailable")
        return "cuda:0"
    if device == "mps":
        if not _available_devices()[1]:
            raise ConfigurationError(f"Requested device {requested!r} is unavailable")
        return "mps"
    raise ConfigurationError(f"Unsupported embedding device {requested!r}")


class _LocalEncoder:
    def __init__(
        self,
        *,
        torch: Any,
        processor: Any,
        runner: Any,
        igniset: Any,
        data_loader: Any,
        device: str,
    ) -> None:
        self._torch = torch
        self._processor = processor
        self._runner = runner
        self._igniset = igniset
        self._data_loader = data_loader
        self._device = device

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        rows = [{"content": text} for text in texts]
        dataset, tensor_names = self._igniset(dicts=rows, processor=self._processor, batch_size=len(rows))
        loader = self._data_loader(dataset=dataset, tensor_names=tensor_names, batch_size=len(rows))

        vectors: list[list[float]] = []
        with self._torch.inference_mode():
            for batch in loader:
                batches = {name: value.to(self._device) for name, value in batch.items()}
                vectors.extend(self._runner(batch=batches)[0].cpu().numpy().tolist())
        return vectors

    def close(self) -> None:
        self._runner = None
        self._processor = None
        self._igniset = None
        self._data_loader = None
        self._torch = None


def _build_local_encoder(model: str, device: str, max_length: int) -> _LocalEncoder:
    try:
        import torch

        from justatom.modeling.mask import ILanguageModel
        from justatom.processing import ITokenizer, RuntimeProcessor, igniset
        from justatom.processing.loader import NamedDataLoader
        from justatom.running.encoders import EncoderRunner
    except ImportError as exc:
        raise ConfigurationError(_LOCAL_DEPENDENCY_ERROR) from exc

    processor = RuntimeProcessor(
        tokenizer=ITokenizer.from_pretrained(model),
        max_seq_len=max_length,
        prefix="",
    )
    runner = EncoderRunner(
        model=ILanguageModel.load(model),
        prediction_heads=[],
        device=device,
        processor=processor,
    ).eval()
    return _LocalEncoder(
        torch=torch,
        processor=processor,
        runner=runner,
        igniset=igniset,
        data_loader=NamedDataLoader,
        device=device,
    )


class HuggingFaceEmbedder:
    def __init__(self, model: str, device: str = "auto", profile: EmbeddingProfile | None = None) -> None:
        if not model.strip():
            raise ConfigurationError("model must be non-empty")
        self.model = model
        self.profile = profile or EmbeddingProfile()
        self.device = resolve_device(device)
        self._encoder: _LocalEncoder | None = _build_local_encoder(model, self.device, self.profile.max_length)
        self._closed = False

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, prefix=self.profile.query_prefix)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, prefix=self.profile.document_prefix)

    async def close(self) -> None:
        if not self._closed:
            if self._encoder is not None:
                self._encoder.close()
                self._encoder = None
            self._closed = True

    async def _embed(self, texts: Sequence[str], *, prefix: str) -> list[list[float]]:
        if not texts:
            return []
        normalized = [apply_prefix(text, prefix, skip_if_present=self.profile.skip_prefix_if_present) for text in texts]
        encoder = self._encoder
        if encoder is None:
            raise RuntimeError("HuggingFaceEmbedder is closed")

        vectors: list[list[float]] = []
        for start in range(0, len(normalized), self.profile.batch_size):
            batch = normalized[start : start + self.profile.batch_size]
            vectors.extend(await asyncio.to_thread(encoder.encode, batch))
        return validate_embeddings(vectors, expected_count=len(texts))
