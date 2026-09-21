"""WeMM text-only chat formatting, including its native embedding postprocessor."""

from justatom.processing.qwen3_vl import Qwen3VLTextTokenizer


class WeMMTextTokenizer(Qwen3VLTextTokenizer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.tokenizer.init_kwargs["justatom_text_format"] = "wemm"

    def __call__(self, texts, **kwargs):
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("WeMM text encoding accepts strings only")
        conversations = [[{"role": "user", "content": [{"type": "text", "text": text}]}] for text in texts]
        prompts = self.tokenizer.apply_chat_template(conversations, add_generation_prompt=False, tokenize=False)
        # tokenizer.json appends <embedding> as a special token AFTER truncation.
        # Disabling this (as for Qwen3-VL) would pool the wrong token.
        kwargs["add_special_tokens"] = True
        return self.tokenizer(prompts[0] if single else prompts, **kwargs)
