from justatom.mq.consuming.rabbitmq import RabbitMQConsuming
from justatom.mq.producing.rabbitmq import RabbitMQProducing


class RabbitMQClient(RabbitMQConsuming, RabbitMQProducing):
    pass
