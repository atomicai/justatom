import os
import random

import openai

from justatom.configuring import Config
from justatom.modeling.mask import IRemoteLargeLanguageModel


class OpenAILargeLanguageModel(IRemoteLargeLanguageModel):
    def __init__(self, models: str = None):
        super().__init__()
        self.models = models or Config.external.models

    async def generate(self, prompt: str, history: list[str] = None, role: str = None, models: list[str] = None, **kwargs):
        models = models or self.models
        role = "You’re a kind helpful assistant" if role is None else role
        model = models[random.randint(0, len(models) - 1)]
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": role}, {"role": "user", "content": prompt}],
            api_key=os.environ.get("OPENAI_API_KEY"),
            **kwargs,
        )
        try:
            return response["choices"][0]["message"]["content"].strip()
        except IndexError:
            return ""


__all__ = ["OpenAILargeLanguageModel"]
