from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic_settings.sources import (
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)


class YAMLSettings(BaseSettings):
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        config_name = settings_cls.Config.yaml_file_path
        return (
            YamlConfigSettingsSource(settings_cls, yaml_file=Path("global.yaml")),
            YamlConfigSettingsSource(settings_cls, yaml_file=Path(config_name)),
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    class Config:
        yaml_file_path = "settings.yaml"
