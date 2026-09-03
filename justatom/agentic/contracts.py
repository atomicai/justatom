from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from justatom.agentic.schemas import PlannerReply, PlannerRequest, RunTrace

if TYPE_CHECKING:
    from justatom.etc.schema import Document


class TraceDeliveryPendingError(TimeoutError):
    """The trace confirmation timed out while accepted delivery is still pending."""


class TracePersistenceError(RuntimeError):
    """Required trace persistence could not be confirmed."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@runtime_checkable
class AgentRetriever(Protocol):
    async def retrieve(self, query: str, *, top_k: int = 5, **kwargs: Any) -> list[Document]: ...


@runtime_checkable
class ChatBackend(Protocol):
    @property
    def backend_name(self) -> str: ...

    @property
    def model_name(self) -> str | None: ...

    @property
    def prompt_fingerprint(self) -> str: ...

    @property
    def config_fingerprint(self) -> str: ...

    async def plan(self, request: PlannerRequest) -> PlannerReply: ...

    async def close(self) -> None: ...


@runtime_checkable
class TraceSink(Protocol):
    async def write(self, trace: RunTrace) -> None: ...

    async def close(self) -> None: ...


__all__ = [
    "AgentRetriever",
    "ChatBackend",
    "TraceDeliveryPendingError",
    "TracePersistenceError",
    "TraceSink",
]
