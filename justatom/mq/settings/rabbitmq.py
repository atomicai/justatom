from pydantic import Field

from justatom.mq.settings.mask import YAMLSettings


class SettingsRabbitMQ(YAMLSettings):
    host: str = Field("localhost", alias="HOST")
    port: int = Field(5672, alias="PORT")
    username: str = Field("guest", alias="USERNAME")
    password: str = Field("guest", alias="PASSWORD")
    reconnect_interval: int = Field(5, alias="RECONNECT_INTERVAL")
    fail_fast: bool = Field(True, alias="FAIL_FAST")
    reconnect_attempts: int = Field(5, alias="RECONNECT_ATTEMPTS")
