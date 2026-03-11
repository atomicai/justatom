from justatom.configuring.scenarios import load_scenario_config


def __getattr__(name):
    if name == "Config":
        from justatom.configuring.prime import Config

        return Config
    raise AttributeError(name)


__all__ = ["Config", "load_scenario_config"]
