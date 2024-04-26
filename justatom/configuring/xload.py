from typing import Union, Optional, Dict
from pathlib import Path
import torch
from loguru import logger
from justatom.configuring.inject import Injector
from justatom.etc.pattern import singleton
from justatom.modeling.mask import GRANTED_MODEL_NAMES
from justatom.processing.mask import GRANTED_PROCESSOR_NAMES


@singleton
class ILoader:

    def ignite_component(
        self, component: str, wtf: Union[torch.nn.Module, Union[str, Path]], props: Optional[Dict] = None
    ):
        if isinstance(wtf, torch.nn.Module):
            return wtf
        wrapper = Path(wtf)
        if component.lower() in GRANTED_MODEL_NAMES:
            return Injector.inject("model", wrapper, props=props)
        elif component.lower() in GRANTED_PROCESSOR_NAMES:
            return Injector.inject("processor", wrapper, props=props)
        else:
            logger.error(f"ARG error `component` name = {component}")
            raise ValueError(f"Unexpected argument {str(wtf)} propagated inside `Loader`")


Loader = ILoader()

__all__ = ["Loader"]
