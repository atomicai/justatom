import copy
import json

import json_repair

from justatom.running.mask import IPromptRunner


class KEYPromptRunner(IPromptRunner):
    _system_prompt = f"""
    You are a highly capable language assistant with remarkable skillset on the following:
    - History and mechanics of computer games.
    - Well-versed in many films.
    - Skilled at providing user support and guidance for complex systems (e.g. user portals, 
      databases, or other technical domains).
    - Scientific facts and general historical facts
    """  # noqa

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
    _system_prompt = f"""
    You are a highly capable language assistant with remarkable skillset on the following:
    - History and mechanics of computer games.
    - Well-versed in many films.
    - Skilled at providing user support and guidance for complex systems (e.g. user portals, 
      databases, or other technical domains).
    - Scientific facts and general historical facts
    """  # noqa

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
    _system_prompt = f"""
    You are a highly capable language assistant with remarkable skillset on the following:
    - History and mechanics of computer games.
    - Well-versed in many films.
    - Skilled at providing user support and guidance for complex systems (e.g. user portals, 
      databases, or other technical domains).
    - Scientific facts and general historical facts
    """  # noqa

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


class QUERIESPromptRunner(IPromptRunner):
    _system_prompt = f"""
    You are a highly capable language assistant with remarkable skillset on the following:
    - History and mechanics of computer games.
    - Well-versed in many films.
    - Skilled at providing user support and guidance for complex systems (e.g. user portals, 
      databases, or other technical domains).
    - Scientific facts and general historical facts
    """  # noqa

    def __init__(self, source_language: str, title: str, system_prompt: str = None, **props):
        super().__init__(system_prompt=system_prompt.strip() if system_prompt is not None else self._system_prompt.strip())
        self.title = title
        self.source_language = source_language
        self.title = title

    def _prepare(self, content, title: str | None = None, source_language: str | None = None, **props):
        source_language = source_language or self.source_language
        styles: str = self.styles if self.styles is not None else ["AI assistant"]
        title = title or self.title
        prompt = f"""
        Generate multiple (two or more) queries in {source_language} using the paragraphs (context) provided below.\n
        Imagine you are one of the professionals: \"{[' | '.join(styles)]}\". Formulate your questions in the style and speech patterns typical of this profession. Feel free to use slang, greetings, informal or even profane language, or any other turns of phrase.\n
        When relevant, refer to the universe (theme) \"{title}\" if the question is broad enough or risks misunderstanding without it. Avoid referencing it otherwise. However, if your question concerns a specific book, game, movie, or unique section, please be explicit.\n
        All questions must be strictly in {source_language}.\n
        Below are the paragraphs, belonging to the topic \"{title}\".\n
        \n{content}\n
        Your output must be in JSON format and include the following structure:
        - "answer": [{{
            "question": \"...\", // A generated question in {source_language}
            "response": \"...\", // The answer found directly in the text
        }}, ...]\n
        where:
        - \"question\" is the generated question in {source_language}, adhering to the chosen style.
        - \"response\" is an answer that can be found directly in the document text.
        \n
        Please respond only with the JSON.
        """.strip()  # noqa
        return prompt

    def finalize(self, content: str, raw_response: str, as_json_string: bool = False, **props):
        if as_json_string:
            return json_repair.loads(raw_response)
        return raw_response


class QUERIESWithSourcesPromptRunner(IPromptRunner):
    _system_prompt = f"""
    You are a highly capable language assistant with remarkable skillset on the following:
    - History and mechanics of computer games.
    - Well-versed in many films.
    - Skilled at providing user support and guidance for complex systems (e.g. user portals, 
      databases, or other technical domains).
    - Scientific facts and general historical facts
    """  # noqa

    def __init__(
        self,
        system_prompt: str | None = None,
        styles: list[str] | None = None,
        source_language: str | None = None,
        title: str | None = None,
        **props,
    ):
        super().__init__(system_prompt=system_prompt.strip() if system_prompt is not None else self._system_prompt.strip())
        self.styles = styles
        self.source_language = source_language
        self.title = title

    def _prepare(self, content, title: str | None = None, source_language: str | None = None, **props):
        source_language = source_language or self.source_language
        styles: str = self.styles if self.styles is not None else ["AI assistant"]
        title = title or self.title
        prompt = f"""
        Generate multiple (two or more) queries in {source_language} language using provided paragraphs (context).\n
        Each paragraph is numbered in sequence, starting from 0. For every question you generate, specify how strongly it relates to each paragraph on a scale from 0 to 10:
            - 0 means it is not related at all.
            - 10 means the question cannot be answered without that paragraph.\n
        Please ask complex questions that require analyzing multiple paragraphs, rather than focusing on just one.\n
        Imagine you are a professional in the style of: "{[' | '.join(styles)]}".\n
        Formulate your questions using that professional’s typical speech patterns—feel free to use slang, greetings, informal or even profane language, or any other manner of expression.\n
        When formulating your questions, refer to the universe (theme) "{title}" only if it is broad enough or if not mentioning it might cause confusion. Otherwise, avoid referencing it. However, if the question involves, for example, a specific book or a unique section, be sure to specify it.\n
        All questions must be strictly in {source_language}.\n
        Below are the paragraphs, belonging to the topic \"{title}\".\n
        \n{content}\n
        Your output must be in JSON format and include the following structure:
        - "answer": [{{
            "question": \"...\", // A generated question in {source_language}
            "response": \"...\", // The answer found directly in the text
            "relevance": [ ... ] // Array of relevance scores (0–10) for each paragraph
        }}, ...]\n
        where:
        - \"question\" is the generated question in {source_language}, adhering to the chosen style.
        - \"response\" is an answer that can be found directly in the document text.
        - \"relevance\" is an array indicating how crucial each paragraph is to the question, on a scale from 0 to 10.
        \n
        Please respond only with the JSON.
        """.strip()  # noqa
        return prompt

    def finalize(self, content: str, raw_response: str, as_json_string: bool = False, **props):
        if as_json_string:
            return json_repair.loads(raw_response)
        return raw_response


class RESPONSEWithConfirmPromptRunner(IPromptRunner):
    _system_prompt = f"""
    You are a highly capable language assistant with remarkable skillset on the following:
    - History and mechanics of computer games.
    - Well-versed in many films.
    - Skilled at providing user support and guidance for complex systems (e.g. user portals, 
      databases, or other technical domains).
    - Scientific facts and general historical facts
    """  # noqa

    def __init__(self, source_language: str | None = None, title: str | None = None, system_prompt: str = None, **props):
        super().__init__(system_prompt=system_prompt.strip() if system_prompt is not None else self._system_prompt.strip())
        self.title = title
        self.source_language = source_language

    def _prepare(self, query: str, content: str, source_language: str | None = None, title: str | None = None, **props):
        source_language = source_language or self.source_language
        title = title or self.title
        prompt = f"""
        You are given a query and a paragraph containing relevant information. **However**, when generating your answer:\n
        1. **Do not** explicitly reference or cite the paragraph text (e.g., do not say "according to the provided paragraph" or "based on the text"). \n
        2. Write your response **as if you have known the facts all along**. \n
        3. Use a natural style of communication in {source_language}.\n
        4. If the paragraph does **not** contain any information to answer the query, respond with a brief comment explaining the lack of information.\n
        Generate an answer in {source_language} based on the information provided:\n
        -  \"query\" represents the user's question, belonging to the universe (theme) \"{title}\".\n
        -  \"paragraph\" refers to the text from which you must derive your answer.\n
        > Important: If the required information is not found in the paragraph, do not invent an answer. Instead:\n
        - Set "is_context_present" to false.\n
        - Replace the answer with a brief comment explaining why the information is missing.\n
        When relevant, refer to the universe (theme) \"{title}\" if the question is broad enough or might be misunderstood without it. Avoid referencing it otherwise. However, if the question concerns a specific book, game, movie, or unique section, be explicit.\n
        Your answer must be strictly in {source_language}.\n
        Below are the query and paragraphs related to the topic \"{title}\":\n
        \n{query}\n
        \n{content}\n
        Your output must be in JSON format and include the following structure:
        - "answer": [{{
            "response": \"...\", // Your generated answer (or comment, if info is missing) in {source_language}
            "is_context_present": \"...\" // Boolean indicating if the necessary info is in the paragraph
        }}, ...]\n
        Where:
        - \"question\" is the exact question from the input.
        - \"response\" is your answer in {source_language} language stated naturally without referencing \"the paragraph\".
        - \"is_context_present\" is either true or false, indicating whether the paragraph contains the required information.\n
        \n
        **Respond only with the JSON.** Avoid any meta-explanations or references to the prompt or paragraph.
        """  # noqa

        return prompt

    def finalize(self, content: str, raw_response: str, as_json_string: bool = False, **props):
        if as_json_string:
            return json_repair.loads(raw_response)
        return raw_response


class RESPONSEWithConfirmAndSourcesPromptRunner(IPromptRunner):
    _system_prompt = f"""
    You are a highly capable language assistant with remarkable skillset on the following:
    - History and mechanics of computer games.
    - Well-versed in many films.
    - Skilled at providing user support and guidance for complex systems (e.g. user portals, 
      databases, or other technical domains).
    - Scientific facts and general historical facts
    """  # noqa

    def __init__(self, source_language: str | None = None, title: str | None = None, system_prompt: str = None, **props):
        super().__init__(system_prompt=system_prompt.strip() if system_prompt is not None else self._system_prompt.strip())
        self.title = title
        self.source_language = source_language

    def _prepare(self, query: str, content: str, source_language: str | None = None, title: str | None = None, **props):
        source_language = source_language or self.source_language
        title = title or self.title
        prompt = f"""
        You are given a query and a paragraphs containing relevant information.\n
        Each paragraph is numbered in sequence, starting from 0. For response you generate, specify how strongly it relates to each paragraph on a scale from 0 to 10:
            - 0 means it is not related at all.
            - 10 means the question cannot be answered without that paragraph.\n**However**, when generating your answer:\n
        1. **Do not** explicitly reference or cite the paragraph text (e.g., do not say "according to the provided paragraph" or "based on the text"). \n
        2. Write your response **as if you have known the facts all along**. \n
        3. Use a natural style of communication in {source_language}.\n
        4. If the paragraphs does **not** contain any information to answer the query, respond with a brief comment explaining the lack of information.\n
        Generate an answer in {source_language} based on the information provided:\n
        -  \"query\" represents the user's question, belonging to the universe (theme) \"{title}\".\n
        -  \"Paragraph <num>\" refers to the text at position <num> from which you must derive your answer and set the relevance score. Remember, each paragraphs starts with \"Paragraph <num>\" where <num> indicates the index of the paragraph\n
        > Important: If the required information is not found in the paragraph, do not invent an answer. Instead:\n
        - Set "is_context_present" to false.\n
        - Replace the answer with a brief comment explaining why the information is missing.\n
        When relevant, refer to the universe (theme) \"{title}\" if the question is broad enough or might be misunderstood without it. Avoid referencing it otherwise. However, if the question concerns a specific book, game, movie, or unique section, be explicit.\n
        Your answer must be strictly in {source_language}.\n
        Below are the query and paragraphs related to the topic \"{title}\":\n
        \n{query}\n
        \n{content}\n
        Your output must be in JSON format and include the following structure:
        - "answer": [{{
            "response": \"...\", // Your generated answer (or comment, if info is missing) in {source_language}
            "relevance": [ ... ] // Array of relevance scores (0–10) for each paragraph
            "is_context_present": \"...\" // Boolean indicating if the necessary info is in the paragraph
        }}, ...]\n
        Where:
        - \"response\" is your answer in {source_language} language stated naturally without referencing \"the paragraph\"".
        - \"relevance\" is an array indicating how crucial each paragraph is to the question, on a scale from 0 to 10.
        - \"is_context_present\" is either true or false, indicating whether the paragraph contains the required information.\n
        \n
        **Respond only with the JSON.** Avoid any meta-explanations or references to the prompt or paragraph.
        """  # noqa

        return prompt

    def finalize(self, content: str, raw_response: str, as_json_string: bool = False, **props):
        if as_json_string:
            return json_repair.loads(raw_response)
        return raw_response
