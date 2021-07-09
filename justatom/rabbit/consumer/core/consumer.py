from justatom.rabbit.consumer.base import Consumer as Base
import pika
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class Consumer(Base):
    
    def __init__(
        self,
        queue_name: str,
        binding_key: str,
        config: dict,
        report_fn: Callable
    ):
        self.queue_name = queue_name
        self.binding_key = binding_key
        self.config = config
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                self.config['host'],
                self.config['port'],
                '/',
                pika.PlainCredentials(
                    self.config['username'],
                    self.config['password']
                )
            )
        )
        
        self.report_fn = report_fn
    
    
    def __del__(self):
        self.connection.close()
        
    def consume(self):
        channel = self.connection.channel()
        channel.exchange_declare(exchange=self.config['exchange'], exchange_type=self.config['exchange_type'], durable=True)
        channel.basic_consume(queue=self.queue_name, on_message_callback=self.report_fn, auto_ack=True)
        channel.start_consuming()
