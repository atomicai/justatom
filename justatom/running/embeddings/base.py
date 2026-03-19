from __future__ import annotations

import abc


class IEmbeddingClient(abc.ABC):
    @abc.abstractmethod
    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
        **props,
    ) -> list[list[float]]:
        """Return one embedding vector per input text."""

    async def close(self) -> None:
        """Optional cleanup hook for clients with network/session resources."""
        return None
