from abc import ABC, abstractmethod
from typing import Any, Generic

from justatom.mq.types.rabbitmq import Message


class IMQProducing(Generic[Message], ABC):
    @abstractmethod
    async def send_message(self, routing_key: str, data: Any) -> Message:
        pass
