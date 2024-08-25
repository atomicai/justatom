import logging
from inspect import isabstract, signature

from justatom.lodash.loader.base.loader import Loader as Base

logger = logging.getLogger(__name__)


class Loader(Base):
    classes = {}

    def __init_subclass__(cls, api: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        logger.info(f"Subclass {cls.__name__} is called")
        if not isabstract(cls):
            api = cls.__name__ if api is None else api
            cls.classes.setdefault(api, {})
            cls.classes[api]["klass"] = cls
            cls.classes[api]["instance"] = None

    @classmethod
    def fire(cls, name: str | None = None, **kwargs):
        name = cls.__name__ if name is None else name
        sig = signature(cls.classes[name]["klass"])
        unused_args = {k: v for k, v in kwargs.items() if k not in sig.parameters}
        logger.debug(f"Got more parameters than needed for loading {name}: {unused_args}. " f"Those won't be used!")
        if cls.classes[name]["instance"] is None:
            cls.classes[name]["instance"] = cls.classes[name]["klass"](**kwargs)
        return cls.classes[name]["instance"]
