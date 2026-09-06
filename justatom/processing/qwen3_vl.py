"""Text preprocessing shared by Qwen3-VL training and local retrieval.

Matches the official embedder's text-only chat formatting, default instruction,
right padding and prompt-level truncation. No image/video dependencies are needed.
"""

from pathlib import Path


class Qwen3VLTextTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        self.tokenizer.init_kwargs["justatom_text_format"] = "qwen3_vl"

    def __getattr__(self, name):
        # Also allow the proxy to be pickled by worker processes.
        if name == "tokenizer":
            raise AttributeError(name)
        return getattr(self.tokenizer, name)

    def __call__(self, texts, **kwargs):
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("Qwen3-VL text encoding accepts strings only, not multimodal inputs")
        conversations = [
            [
                {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
                {"role": "user", "content": [{"type": "text", "text": text or "NULL"}]},
            ]
            for text in texts
        ]
        prompts = self.tokenizer.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        kwargs["add_special_tokens"] = False
        return self.tokenizer(prompts[0] if single else prompts, **kwargs)

    def save_pretrained(self, destination: str | Path, **kwargs):
        return self.tokenizer.save_pretrained(destination, **kwargs)
