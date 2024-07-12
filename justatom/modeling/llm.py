from justatom.modeling.mask import IRemoteLargeLanguageModel
from typing import List
import os
import openai
import random
from justatom.configuring import Config


class OpenAILargeLanguageModel(IRemoteLargeLanguageModel):

    def __init__(self, models: str = None):
        super().__init__()
        self.models = models or Config.external.models

    async def generate(
        self, prompt: str, history: List[str] = None, role: str = None, models: List[str] = None, **kwargs
    ):
        models = models or self.models
        role = "You’re a kind helpful assistant" if role is None else role
        model = models[random.randint(0, len(models) - 1)]
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": role}, {"role": "user", "content": prompt}],
            api_key=os.environ.get("OPENAI_API_KEY"),
            **kwargs
        )
        try:
            return response["choices"][0]["message"]["content"].strip()
        except IndexError:
            return ""


__all__ = ["OpenAILargeLanguageModel"]
