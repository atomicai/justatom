from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Generic

from justatom.mq.types.rabbitmq import Message


class IMQConsuming(Generic[Message], ABC):
    @abstractmethod
    async def get_consumer_generator(self, routing_key: str) -> AsyncGenerator[Message, None]:
        pass
