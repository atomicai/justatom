import json
import logging

from justatom.rabbit import Consumer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    def callback(ch, method, properties, body):
        data = json.loads(body)
        logger.info(f'... POST {data["id"]} {data["post"]} is received...')
        # --- ATOM analyzing
        # ---
        # ---

    consumer = Consumer.fire(
        queue_name="my-app",
        binding_key="argument",
        config={
            "host": "localhost",
            "port": 5672,
            "username": "user",
            "password": "password",
            "exchange": "my-exchange",
            "exchange_type": "direct",
        },
        report_fn=callback,
    )

    consumer.consume()
