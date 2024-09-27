import json
from typing import Any

import aio_pika

from justatom.mq.mixins.rabbitmq import AsyncRabbitMQMixin
from justatom.mq.producing.mask import IMQProducing

__all__ = ["RabbitMQProducing"]


class RabbitMQProducing(AsyncRabbitMQMixin, IMQProducing):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._connection = None

    @staticmethod
    def _serialize_message(message: Any, **metadata) -> bytes:
        message = json.dumps(
            {
                "message": message,
                **metadata,
            }
        )
        return bytes(message, encoding="utf-8")

    async def send_message(
        self,
        routing_key: str,
        message: Any,
        exchange_name: str = "router_exchange",
        producer_routing_key: str | None = None,
    ) -> None:
        if producer_routing_key is None:
            producer_routing_key = self.client_name

        connection = await self._get_or_create_robust_connection()

        channel = await connection.channel()
        exchange = await channel.declare_exchange(exchange_name, aio_pika.ExchangeType.DIRECT)
        await exchange.publish(
            aio_pika.Message(body=self._serialize_message(message=message, producer_routing_key=producer_routing_key)),
            routing_key=routing_key,
        )
        await channel.close()
