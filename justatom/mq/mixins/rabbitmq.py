from collections.abc import AsyncGenerator

from aio_pika.exchange import AbstractExchange
from aio_pika.robust_connection import RobustChannel, RobustConnection

from justatom.mq.settings.rabbitmq import SettingsRabbitMQ


class AsyncRabbitMQMixin:
    def __init__(self, settings: SettingsRabbitMQ, client_name: str):
        self.settings = settings
        self.client_name = client_name

    async def _get_or_create_robust_connection(self) -> RobustConnection:
        if self._connection is None:
            self._connection = RobustConnection(
                url=f"amqp://{self.settings.username}:{self.settings.password}@" f"{self.settings.host}:{self.settings.port}/",
                retry_delay=self.settings.reconnect_interval,
                retry_max=self.settings.reconnect_attempts,
                fail_fast=self.settings.fail_fast,
            )
            await self._connection.connect()

        if self._connection.is_closed:
            await self._connection.connect()

        return self._connection

    @staticmethod
    async def _create_robust_channel(
        connection: RobustConnection,
    ) -> AsyncGenerator[RobustChannel, None]:
        async with connection:
            channel = await connection.channel()
            yield channel

    @staticmethod
    async def _create_exchnage(channel: RobustChannel, name: str) -> AbstractExchange:
        exchange = await channel.declare_exchange(
            name,
        )
        return exchange
