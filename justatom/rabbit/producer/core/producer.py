import json

import pika

from justatom.rabbit.producer.base import Producer as Base


class Producer(Base):
    def __init__(self, config):
        self.config = config
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                self.config["host"],
                self.config["port"],
                "/",
                pika.PlainCredentials(self.config["username"], self.config["password"]),
            )
        )

    def __del__(self):
        self.connection.close()

    def publish(self, routing_key, message):
        channel = self.connection.channel()
        # Connect to the Exchange client. Using the "topic" or "direct" type for delivering message(s) to the consumer(s)
        # This method creates an exchange if it does not already exist, and if the exchange exists, verifies that it is of the correct and expected class.  # noqa: E501
        channel.exchange_declare(exchange=self.config["exchange"], exchange_type=self.config["exchange_type"], durable=True)

        channel.basic_publish(exchange=self.config["exchange"], routing_key=routing_key, body=json.dumps(message))
