from langchain_core.callbacks import AsyncCallbackHandler, AsyncCallbackManager
from langchain_openai import ChatOpenAI
from loguru import logger


def construct_llm_protocol(
    host: str = None,
    port: int = None,
    model_name: str = None,
    temperature: float = 0.22,
    max_tokens=4096,
    streaming: bool = True,
    verbose: bool = False,
    api_key: str = "<YOUR_API_KEY_HERE>",
    callbacks: list[AsyncCallbackHandler] = None,
):
    if host is not None:
        base_url = f"{host}" if port is None else f"{host}:{port}"
    else:
        assert model_name is not None, logger.error(
            """
        Using openai model parameter is mandatory. Please set VAR `model_name` to one of the available models. See 
        """
        )
        base_url = None
    callback_manager = AsyncCallbackManager(callbacks) if callbacks is not None else None
    return ChatOpenAI(
        base_url=base_url,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        verbose=verbose,
        api_key=api_key,
        callbacks=callback_manager,
    )


__all__ = ["construct_llm_protocol"]
