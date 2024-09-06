import asyncio as asio
from collections.abc import Coroutine, Sequence

# Utility below to limit the number of parallel coros.
# Using via:
# await asio.gather(
#        *_limit_concurrency(
#           [coro(...) for coro in coros], concurrency=2
#       )
#   )


def _limit_concurrency(coroutines: Sequence[Coroutine], concurrency: int) -> list[Coroutine]:
    """Decorate coroutines to limit concurrency.
    Enforces a limit on the number of coroutines that can run concurrently in higher
    level asyncio-compatible concurrency managers like asyncio.gather(coroutines) and
    asyncio.as_completed(coroutines).
    """

    semaphore = asio.Semaphore(concurrency)

    async def with_concurrency_limit(coroutine: Coroutine) -> Coroutine:
        async with semaphore:
            return await coroutine

    return [with_concurrency_limit(coroutine) for coroutine in coroutines]


__all__ = ["_limit_concurrency"]
