import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Any

import aio_pika
from aio_pika.message import AbstractMessage

from justatom.mq.consuming.mask import IMQConsuming
from justatom.mq.mixins.rabbitmq import AsyncRabbitMQMixin

__all__ = ["RabbitMQConsuming"]

executor = ThreadPoolExecutor(max_workers=4)


def wrap_callback(callback: Callable[[aio_pika.IncomingMessage, dict[str, str]], None]) -> Callable:
    @wraps(callback)
    async def wrapper(message: aio_pika.IncomingMessage):
        str_message, metadata = RabbitMQConsuming.deserialize_message(message.body)
        message.body = str_message
        await callback(message, metadata)

    return wrapper


class RabbitMQConsuming(AsyncRabbitMQMixin, IMQConsuming):
    @staticmethod
    def deserialize_message(message: bytes) -> tuple[str, dict[str, Any]]:
        metadata = json.loads(message)
        message = metadata.pop("message")

        return message, metadata

    async def _get_queue(
        self,
        routing_key: str | None = None,
        exchange_name: str = "router_exchange",
        quueue_name: str = "router_queue",
    ) -> tuple[aio_pika.Channel, aio_pika.Exchange, aio_pika.Queue]:
        if routing_key is None:
            routing_key = self.client_name

        connection = await self._get_or_create_robust_connection()

        channel = await connection.channel()
        exchange = await channel.declare_exchange(exchange_name, aio_pika.ExchangeType.DIRECT)

        queue = await channel.declare_queue(quueue_name)

        await queue.bind(exchange, routing_key)

        return channel, exchange, queue

    async def get_consumer_generator(
        self,
        routing_key: str | None = None,
        exchange_name: str = "router_exchange",
        quueue_name: str = "router_queue",
    ) -> AsyncGenerator[AbstractMessage, None]:
        channel, _, queue = await self._get_queue(routing_key, exchange_name, quueue_name)

        async with queue.iterator() as queue:
            async for message in queue:
                async with message.process():
                    str_message, metadata = self.deserialize_message(message.body)
                    message.body = str_message
                    yield message, metadata

        await channel.close()

    async def consume_with_callback(
        self,
        callback: Callable[[str, dict[str, Any]], None],
        routing_key: str | None = None,
        exchange_name: str = "router_exchange",
        quueue_name: str = "router_queue",
    ) -> None:
        ### callback example
        #
        # async def callback(message: str, metadata: dict[str, Any]):
        #

        async def on_message(message: aio_pika.IncomingMessage) -> None:
            async with message.process():
                loop = asyncio.get_event_loop()
                str_message, metadata = self.deserialize_message(message.body)
                await loop.run_in_executor(executor, callback, str_message, metadata)

        _, _, queue = await self._get_queue(routing_key, exchange_name, quueue_name)

        await queue.consume(on_message)
        await asyncio.Future()
