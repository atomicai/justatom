import logging

import random_name

from justatom.rabbit import Producer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    producer = Producer.fire(
        config={
            "host": "localhost",
            "port": 5672,
            "username": "user",
            "password": "password",
            "exchange": "my-exchange",
            "exchange_type": "direct",
        }
    )

    post = "This is some post about selling the iphone 12 pro bitch! --- " + random_name.generate_name()

    message = {"post": post, "id": 2077}

    producer.publish("argument", message)
