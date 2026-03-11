from justatom.running.embeddings.base import IEmbeddingClient
from justatom.running.embeddings.local import LocalEmbeddingClient
from justatom.running.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingClient,
)


class EmbeddingClientFactory:
    @staticmethod
    def from_backend(backend: str, **kwargs) -> IEmbeddingClient:
        normalized = backend.strip().lower()
        if normalized in {"openai", "openai-compatible", "openai_compatible"}:
            return OpenAICompatibleEmbeddingClient(
                base_url=kwargs["base_url"],
                api_key=kwargs["api_key"],
                model=kwargs["model"],
                timeout=kwargs.get("timeout", 30.0),
                query_prefix=kwargs.get("query_prefix", ""),
                passage_prefix=kwargs.get("passage_prefix", ""),
                default_input_type=kwargs.get("default_input_type", "raw"),
                prefix_enabled=kwargs.get("prefix_enabled", True),
                prefix_skip_if_present=kwargs.get("prefix_skip_if_present", True),
                default_pooling=kwargs.get("default_pooling"),
                default_encoding_format=kwargs.get("default_encoding_format"),
                default_max_seq_len=kwargs.get("default_max_seq_len"),
            )
        if normalized == "local":
            return LocalEmbeddingClient(
                model_name_or_path=kwargs["model_name_or_path"],
                device=kwargs.get("device", "cpu"),
                prefix=kwargs.get("prefix", ""),
                max_seq_len=kwargs.get("max_seq_len", 512),
                batch_size=kwargs.get("batch_size", 64),
            )
        raise ValueError(f"Unknown embedding backend: {backend}")


__all__ = [
    "IEmbeddingClient",
    "LocalEmbeddingClient",
    "OpenAICompatibleEmbeddingClient",
    "EmbeddingClientFactory",
]
