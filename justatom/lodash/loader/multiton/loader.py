import hashlib
import logging
from inspect import isabstract, signature

import cloudpickle

from justatom.lodash.loader.base.loader import Loader as Base

logger = logging.getLogger(__name__)


class Loader(Base):
    classes = {}

    def __init_subclass__(cls, api: str | None = None, **kwargs):
        """
        This automatically keeps track of all available subclasses.
        Enables generic load() and load_from_dir() for all specific implementation(s).
        """
        super().__init_subclass__(**kwargs)
        if not isabstract(cls):
            api = cls.__name__ if api is None else api
            cls.classes.setdefault(api, {})
            cls.classes[api]["klass"] = cls
            cls.classes[api]["instance"] = {}

    @classmethod
    def fire(cls, name: str | None = None, *args, **kwargs):
        name = cls.__name__ if name is None else name
        sig = signature(cls.classes[name]["klass"])
        unused_args = {k: v for k, v in kwargs.items() if k not in sig.parameters}
        logger.debug(f"Got more parameters than needed for loading {name}: {unused_args}. " f"Those won't be used!")
        instance = cls.classes[name]["instance"]
        uhid = hashlib.md5(cloudpickle.dumps((args, kwargs))).hexdigest()
        if uhid not in instance.keys():  # noqa: SIM118
            instance[uhid] = {"args": (args, kwargs), "instance": cls(*args, **kwargs)}
        return instance[uhid]["instance"]
