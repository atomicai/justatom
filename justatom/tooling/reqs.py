import os


async def openai_chat(content_or_messages: str | list[dict], model: str = "gpt-4", timeout: int = 2, props: dict | None = None):
    import openai_async as remoapi

    content_or_messages = (
        [{"role": "user", "content": content_or_messages}] if isinstance(content_or_messages, str) else content_or_messages
    )
    coro_response = await remoapi.chat_complete(
        os.environ.get("OPENAI_API_KEY"), timeout=timeout, payload=dict(model=model, messages=content_or_messages)
    )
    response = coro_response.json()
    props = {} if props is None else props
    return dict(response=response["choices"][0]["message"]["content"], **props)
