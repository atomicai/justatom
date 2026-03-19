from __future__ import annotations

import asyncio as asio

import torch

from justatom.modeling.mask import ILanguageModel
from justatom.processing import RuntimeProcessor
from justatom.processing import ITokenizer
from justatom.processing import igniset
from justatom.processing.loader import NamedDataLoader
from justatom.running.embeddings.base import IEmbeddingClient
from justatom.running.encoders import EncoderRunner


class LocalEmbeddingClient(IEmbeddingClient):
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        prefix: str = "",
        max_seq_len: int = 512,
        batch_size: int = 64,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.prefix = prefix
        self.max_seq_len = int(max_seq_len)
        self.batch_size = int(batch_size)

        lm_model = ILanguageModel.load(model_name_or_path)
        self.processor = RuntimeProcessor(
            tokenizer=ITokenizer.from_pretrained(model_name_or_path),
            max_seq_len=self.max_seq_len,
            prefix=self.prefix,
        )
        self.runner = EncoderRunner(
            model=lm_model,
            prediction_heads=[],
            device=self.device,
            processor=self.processor,
        ).eval()

    @torch.no_grad()
    def _embed_sync(
        self,
        texts: list[str],
        *,
        streaming_preprocessing: bool = False,
    ) -> list[list[float]]:
        rows = [{"content": t} for t in texts]
        dataset, tensor_names = igniset(
            dicts=rows,
            processor=self.processor,
            batch_size=self.batch_size,
            streaming=streaming_preprocessing,
        )
        loader = NamedDataLoader(
            dataset=dataset,
            tensor_names=tensor_names,
            batch_size=self.batch_size,
        )

        vectors_out: list[list[float]] = []
        for batch in loader:
            batches = {k: v.to(self.device) for k, v in batch.items()}
            vectors = self.runner(batch=batches)[0].cpu().numpy().tolist()
            vectors_out.extend(vectors)
        return vectors_out

    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
        streaming_preprocessing: bool = False,
        **props,
    ) -> list[list[float]]:
        del model, props
        if len(texts) == 0:
            return []
        return await asio.to_thread(
            self._embed_sync,
            texts,
            streaming_preprocessing=streaming_preprocessing,
        )
