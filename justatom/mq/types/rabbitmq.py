from typing import TypeVar

from aio_pika.abc import AbstractMessage

Message = TypeVar("Message", bound=AbstractMessage)
