import copy
import json

import json_repair

from justatom.running.mask import IPromptRunner


class KEYPromptRunner(IPromptRunner):
    def __init__(self, system_prompt: str, source_language: str = "русском", **props):
        super().__init__(system_prompt=system_prompt.strip())
        self.source_language = source_language

    def _prepare(self, content: str, title: str, source_language: str | None = None, **props):
        source_language = source_language or self.source_language
        prompt = f"""
        Обрати внимание, что ключевые слова или фразы должны быть подстрокой параграфа и состоять из НЕ более двух, максимум трех слов.\n
        Каждая фраза или ключевое слово должны иметь краткое, но емкое объяснение на {source_language} языке в зависимости от контекста, в котором они употреблены. \n
        Параграф из вселенной \"{title}\":\n{content}\n\n
        Выдай ответ в виде  json в формате: {{"keywords_or_phrases": [{{"keyword_or_phrase": <Выделенная тобою фраза>, "explanation": <Объяснение на {source_language} языке для ребенка в соответствии с контекстом, в котором употреблена ключевая фраза или слово>}}]}}.\n
        Выдай только json.
        """.strip()  # noqa
        return prompt

    def finalize(self, content: str, raw_response: str, as_json_string: bool = False, **props):
        response = []
        js_response = json_repair.loads(raw_response)
        for js_phrase in js_response["keywords_or_phrases"]:
            if content.find(js_phrase["keyword_or_phrase"]) < 0:
                pos = content.lower().find(js_phrase["keyword_or_phrase"])
                if pos < 0:
                    continue
                new_js_phrase = copy.deepcopy(
                    {
                        "keyword_or_phrase": content[pos : pos + len(js_phrase["keyword_or_phrase"])],
                        "explanation": js_phrase["explanation"],
                    }
                )
            else:
                new_js_phrase = js_phrase
            response.append(new_js_phrase)
        final_response = {"keywords_or_phrases": response, **props}
        if as_json_string:
            return json.dumps(final_response)
        return final_response


class TRLSPromptRunner(IPromptRunner):
    def __init__(self, system_prompt: str, source_language: str | None = None, **props):
        super().__init__(system_prompt=system_prompt.strip())
        self.source_language = source_language

    def _prepare(self, content: str, title: str, source_language: str | None = None, **props):
        source_language = source_language or self.source_language
        prompt = f"""
        Параграф из вселенной \"{title}\":\n{content}\n\n
        Выдай ответ в виде  json в формате: {{"translation": "<Твой перевод на {source_language} языке, учитывающий контекст параграфа из вселенной {title}>}}"
        Выдай только json.
        """.strip()  # noqa
        return prompt

    def finalize(self, content: str, raw_response: str, as_json_string: bool = False, **props):
        if as_json_string:
            return json_repair.loads(raw_response)
        return raw_response


class REPHRASEPromptRunner(IPromptRunner):
    def __init__(self, system_prompt: str, styles: list[str] | None = None, **props):
        super().__init__(system_prompt=system_prompt.strip())
        self.styles = styles

    def _prepare(self, content, title: str, **props):
        styles: str = self.styles if self.styles is not None else ["AI ассистент"]
        prompt = f"""
        Сделай несколько (две или более) парафраз из исходного параграфа или вопроса.\n
        Представь, что ты один из следующих профессоналов: \"{[' | '.join(styles)]}\". Создавай перефраз в соответствии с речевыми оборотами твоей професии (роли). Можешь использовать сленговые слова и добавлять приветствия, неформальную, ненормативную лексику и другие речевые обороты.\n
        Параграф (вопрос) из вселенной\"{title}\":\n{content}\n\n
        Выдай ответ в виде json в формате: {{"rephrase_phrases": "[{{"phrase": <Твой пересказ, учитывающий контекст параграфа или вопроса из вселенной {title}>}}, ...]"}}
        """.strip()  # noqa
        return prompt

    def finalize(self, content: str, raw_response: str, as_json_string: bool = False, **props):
        if as_json_string:
            return json_repair.loads(raw_response)
        return raw_response


class QUERIESPropmtRunner(IPromptRunner):
    def __init__(
        self,
        system_prompt: str,
        styles: list[str] | None = None,
        source_language: str | None = None,
        title: str | None = None,
        **props,
    ):
        super().__init__(system_prompt=system_prompt.strip())
        self.styles = styles
        self.source_language = source_language
        self.title = title

    def _prepare(self, content, title: str | None = None, source_language: str | None = None, **props):
        source_language = source_language or self.source_language
        styles: str = self.styles if self.styles is not None else ["AI ассистент"]
        title = title or self.title
        prompt = f"""
        Generate multiple (two or more) queries in {source_language} language using provided paragraph (context).\n
        Imagine, you are one of the professional: \"{[' | '.join(styles)]}\". Ask your question in line with the speech patterns of your profession (role). Feel free to use slang, greetings, informal or even profane language, and any other turns of phrase.\n
        In your question, refer to the universe (theme) \"{title}\" if the question is broad enough or might be misunderstood. Avoid doing so whenever possible, but if, for example, it's about a book or a unique section, be sure to specify it.\n
        Questions must be strictly in the {source_language} language!\n
        Paragraph belongs to the topic - \"{title}\".\n
        Paragraph:\n\n{content}\n\n
        Each synthetic dataset should be in JSON format containing:
        - "answer": [{{
            "question": [Generated short question],
            "response": [The answer can be found directly in the text of the document]
        }}, ...]
        Respond with JSON only.
        """.strip()  # noqa
        return prompt

    def finalize(self, content: str, raw_response: str, as_json_string: bool = False, **props):
        if as_json_string:
            return json_repair.loads(raw_response)
        return raw_response
