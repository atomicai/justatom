import copy
import json

import json_repair

from justatom.running.mask import IPromptRunner


class KPHExtractorPromptRunner(IPromptRunner):
    _system_prompt = f"""
    You are a highly capable AI language assistant with remarkable skillset on the following:
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
        Extract key words or short phrases **that are exact substrings of the paragraph** (each no longer than three words).

        For every extracted item, write a **brief yet clear explanation** in {source_language}, phrased so simply that a child could understand it, and tailored to the context in which the phrase appears.

        Paragraph from the “{title}” universe:
        {content}

        Return **only** valid JSON in this form:
        {{
        "keywords_or_phrases": [
            {{
            "keyword_or_phrase": "<your extracted phrase>",
            "explanation": "<your {source_language} explanation>"
            }}
        ]
        }}
        """  # noqa
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


class QUESTIONGeneratorPromptRunner(IPromptRunner):
    _system_prompt = f"""
    You are a highly capable AI language assistant with remarkable skillset on the following:
    - History and mechanics of computer games.
    - Well-versed in many films.
    - Skilled at providing user support and guidance for complex systems (e.g. user portals, 
      databases, or other technical domains).
    - Scientific facts and general historical facts
    """  # noqa

    _styles = [
        "Russian gopnik thug (street slang, brash tone, casual profanity)",
        "Rap artist (rhymed bars, urban slang, strong beat cadence)",
        "Screenwriter (cinematic narration, stage directions, dramatic flair)",
        "IT geek (tech jargon, acronyms, dry humor, code references)",
        "High-level banker (formal business jargon, polished courtesy)",
        "Drug-addict stoner (slurred, laid-back speech, spaced-out remarks)",
    ]  # noqa

    def __init__(self, source_language: str, title: str, system_prompt: str = None, **props):
        super().__init__(system_prompt=system_prompt.strip() if system_prompt is not None else self._system_prompt.strip())
        self.title = title
        self.source_language = source_language
        self.title = title

    def _prepare(self, content, title: str | None = None, source_language: str | None = None, styles: list[str] = None, **props):
        source_language = source_language or self.source_language
        self.styles = styles if styles is not None else self._styles
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


class QUESTIONGeneratorSourcerPromptRunner(IPromptRunner):
    _system_prompt = f"""
    You are a highly capable language assistant with remarkable skillset on the following:
    - History and mechanics of computer games.
    - Well-versed in many films.
    - Skilled at providing user support and guidance for complex systems (e.g. user portals, 
      databases, or other technical domains).
    - Scientific facts and general historical facts
    """  # noqa

    _styles = [
        "Russian gopnik thug (street slang, brash tone, casual profanity)",
        "Rap artist (rhymed bars, urban slang, strong beat cadence)",
        "Screenwriter (cinematic narration, stage directions, dramatic flair)",
        "IT geek (tech jargon, acronyms, dry humor, code references)",
        "High-level banker (formal business jargon, polished courtesy)",
        "Drug-addict stoner (slurred, laid-back speech, spaced-out remarks)",
    ]  # noqa

    def __init__(
        self,
        system_prompt: str | None = None,
        styles: list[str] | None = None,
        source_language: str | None = None,
        title: str | None = None,
        **props,
    ):
        super().__init__(system_prompt=system_prompt.strip() if system_prompt is not None else self._system_prompt.strip())
        self.styles = styles if styles is not None else self._styles
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


class RANKERWithScorePromptRunner(IPromptRunner):
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

    def _prepare(
        self,
        query: str,
        content: str,
        golden_answer: str,
        responses: list[str],
        source_language: str | None = None,
        title: str | None = None,
        **props,
    ):
        # Format the candidate responses as a numbered list.
        responses_list = "\n".join([f"response_{i+1}: {resp}" for i, resp in enumerate(responses)])

        # Construct the prompt with explicit instructions.
        prompt = f"""
    Your task is to evaluate the quality of each candidate answer
    provided by different language models in response to a given question.
    Use the input data below, and compare each candidate answer with the golden answer
    in terms of accuracy, completeness, style, and length.

    Input Data:
    1. Question:
    {query}

    2. Paragraph (Content):
    {content}

    3. Golden Answer:
    {golden_answer}

    4. Candidate Answers:
    {responses_list}

    Instructions:
    For each candidate answer, provide a score from 1 to 10 based on the following aspects:
    - Accuracy: How well does the candidate answer correctly address the question?
    - Completeness: Does the candidate answer cover all essential aspects compared to the golden answer?
    - Style and Length: Evaluate if the candidate answer maintains
    a suitable tone, clarity, and appropriate length relative to the golden answer.

    Scoring Guidelines:
    - A score of 1 indicates that the candidate answer is entirely inadequate.
    - A score of 10 indicates that the candidate answer is exceptional, fully addressing the question and
    matching the quality, style, and depth of the golden answer.

    Output Format:
    Return your evaluation in a valid JSON object following this exact structure:
    {{
    "answer": {{
        "response_1": <score>,
        "response_2": <score>,
        ...
    }}
    }}

    Note:
    - Ensure that the number of key-value pairs exactly matches the number of candidate responses.
    - Do not include any additional commentary or text outside the specified JSON structure.
    """
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
        - {{"answer": [{{
            "response": \"...\", // Your generated answer (or comment, if info is missing) in {source_language}
            "is_context_present": \"...\" // Boolean indicating if the necessary info is in the paragraph
        }}, ...]}}\n
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


class RESPONSEWithConfirmSourcerPromptRunner(IPromptRunner):
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


class RESPONSEWithStreamingPromptRunner(IPromptRunner):
    _system_prompt = f"""
    You are a highly capable language assistant with remarkable skillset on the following:
    - History and mechanics of computer games.
    - Well-versed in many films.
    - Skilled at providing user support and guidance for complex systems (e.g. user portals, 
      databases, or other technical domains).
    - Scientific facts and general historical facts
    """  # noqa

    def __init__(self, system_prompt: str = None, source_language: str = "RUSSIAN", title: str = "Documentation", **props):
        super().__init__(system_prompt=system_prompt.strip() if system_prompt is not None else self._system_prompt.strip())
        self.source_language = source_language
        self.title = title

    def _prepare(self, query: str, content: str, source_language: str = None, title: str = None, **props):
        source_language = source_language or self.source_language
        title = title or self.title
        prompt = f"""
        **Task**: Answer the user's query below, following these rules:  
1. **Primary Source**: The provided paragraph is your *main reference*, but you can **supplement it with your own knowledge** if needed.  
2. **Theme Context**: The query relates to "{title}". Use this to guide your response, but avoid over-referencing it unless necessary.  
3. **Language**: Respond in {source_language}.  
4. **Accuracy**:  
   - If the paragraph contains enough information, prioritize it.  
   - If the paragraph is insufficient, fill gaps with your expertise **without hallucinating**.  
   - If the topic is outside your knowledge, say: "I don't have enough data about [specific detail]. Could you clarify?"  
   
**Avoid**:  
- Phrases like "According to the paragraph..." or "The text says...".  
- Unnecessary meta-commentary about sources.  


**Query**:
{query}  

**Relevant Paragraph (for context)**:  
{content}  

Answer concisely, blending the paragraph with your expertise when appropriate.  
        """  # noqa

        return prompt


class RESPONSEWithStreamingJsonPromptRunner(RESPONSEWithStreamingPromptRunner):
    _system_prompt = f"""
    You are a highly capable language assistant with remarkable skillset on the following:
    - History and mechanics of computer games.
    - Well-versed in many films.
    - Skilled at providing user support and guidance for complex systems (e.g. user portals, 
      databases, or other technical domains).
    - Scientific facts and general historical facts
    """  # noqa

    def __init__(self, system_prompt: str = None, source_language: str = "RUSSIAN", title: str = "Documentation", **props):
        super().__init__(
            system_prompt=system_prompt.strip() if system_prompt is not None else self._system_prompt.strip(),
            source_language=source_language,
            title=title,
        )

    def _prepare(self, query: str, content: str, source_language: str = None, title: str = None, **props):
        source_language = source_language or self.source_language
        title = title or self.title
        prompt = f"""
        **Task**: Answer the user's query in JSON format, following these rules:  
1. **Response Format**:  
   ```json
   {{
     "answer": "<YOUR_RESPONSE>",
     "sources": ["paragraph", "internal_knowledge"]  // Укажи источники ([\"paragraph\"] OR [\"internal_knowledge\"] OR [\"paragarph\", \"internal_knowledge\"]
   }}
   ```
2. **Content use**:
    - The provided paragraph is your primary source, but you can add your own knowledge if relevant.
    - If the paragraph is insufficient, fill gaps logically without hallucinations.
3. **Theme Context**: The query relates to "{title}". Use this to guide your response, but avoid over-referencing it unless necessary.  
4. **Language**: Respond in {source_language}.  

**Avoid**:  
- Phrases like "According to the paragraph..." or "The text says...".  
- Unnecessary meta-commentary about sources and format.


**Query**:  
{query}  

**Relevant Paragraph (for context)**:  
{content}  

Answer concisely, blending the paragraph with your expertise when appropriate.  

**Example Output**
```json
{{
  \"response\":  \"В Skyrim крафт доступен у кузнечного горна, алхимического стола и стола зачарователя. Например, для оружия нужны железные слитки и кожа.\",
  \"sources\": [\"paragraph\", \"internal_knowledge\"]
}}
```
        """  # noqa

        return prompt


class POLAROIDSPromptRunner(IPromptRunner):
    _system_prompt = f"""
    You are a highly capable language assistant with remarkable skillset on the following:
    - History and mechanics of computer games.
    - Well-versed in many films.
    - Skilled at providing user support and guidance for complex systems (e.g. user portals, 
      databases, or other technical domains).
    - Scientific facts and general historical facts
    """  # noqa

    def __init__(self, system_prompt: str = None, source_language: str = "English"):
        super().__init__(
            system_prompt=system_prompt.strip() if system_prompt is not None else self._system_prompt.strip(),
        )
        self.source_language = source_language

    def _prepare(
        self,
        title: str,
        source_language: str = None,
        content: str = None,
        **props,
    ):
        if content is None:
            prompt = f"""
            Your task is to:
            1. Generate an **ICONIC dialogue** from the specified universe (`title`)
            in the requested `source_language` (e.g., English, Russian).
            2. Format the output strictly as JSON with:
                - Dialogue lines with explicit speaker tags
                - List of speakers
                - Author/creator
                - Scene summary (in the same `source_language`)
                - The language used (`source_language`)

            **Rules**
            - **Language**: Generate dialogue and summary **strictly in the requested `source_language`**.
            - **Canon Compliance**: Use only **official/canon dialogues** (no fan fiction or adaptations).
            - **Speaker Format**: Tag speakers like `[Character Name]: Text`.
            - **Length**: 3-12 exchanges maximum.
            - **Precision**: The `content` field must allow exact scene identification (location, context).

            **JSON Template**
            ```json
            {{
            "response": "[Character 1]: Dialogue line\n[Character 2]: Response line",
            "speakers": ["Character 1", "Character 2"],
            "author": "Author/Creator",
            "content": "Scene description (location, purpose)",
            "source_language": "Language used (e.g., Russian)"
            }}

            **Examples**
            For title="Голодные игры", source_language="Russian":
            ```json
            {{
            "response": "[Китнисс]: Прим! Прим! (ведущим)
            Я доброволец! Я доброволец! Я хочу участвовать в Играх!\n[Эффи Тринкет]: Кажется, у нас есть доброволец! ...
            Первый в истории Дистрикта 12 доброволец, прошу на сцену! ... Как тебя зовут?\n
            [Китнисс]: Китнисс Эвердин.\n
            [Эффи Тринкет]: Готова поспорить, это твоя сестренка. Я угадала?\n
            [Китнисс]: Да.",
            "speakers": ["Китнисс", "Эффи Тринкет"],
            "author": "Сьюзен Коллинз",
            "content": "Сцена жеребьёвки, где Китнисс заменяет Прим (Площадь Дистрикта 12)",
            "source_language": "Russian"
            }}

            For title="The Lord of the Rings", source_language="English":
            ```json
            {{
            "response": "[Gandalf]: You shall not pass!\n[Balrog]: [roars in flames]",
            "speakers": ["Gandalf", "Balrog"],
            "author": "J.R.R. Tolkien",
            "content": "Bridge of Khazad-dûm confrontation during the Fellowship's escape",
            "source_language": "English"
            }}

            Now, generate a dialogue for:

            Universe (title): {title}

            Language: {source_language}

            Valid JSON only. No external commentary. Preserve character names' original language
            (e.g., "Гэндальф" for Russian, "Gandalf" for English).
            Match the source_language for dialogue and summary.
            """
        else:
            prompt = f"""
            You are given:
            - `title`: the universe or work (e.g., "Harry Potter", "Голодные игры").
            - `source_language`: the language in which to generate the output (e.g., English, Russian).
            - `context`: a **specific paragraph or excerpt** from the canon text, describing a scene or its fragment. Use this context to guide you in identifying and generating the relevant dialogue.
            
            **Task**
Generate an **ICONIC, strictly canonical dialogue** from the given `title` and the **precise scene** specified in the provided `context`, using only the official, original dialogue (no fanfiction or adaptation).

**Output requirements**
- Output must be **valid JSON** matching the structure below (no commentary).
- All dialogue lines must have explicit speaker tags in the format `[Character Name]:`.
- Use the **original character names in the requested language** (e.g., "Гэндальф" in Russian, "Gandalf" in English).
- The summary (`content`) must describe the scene, location, and purpose in the same `source_language`.
- **All fields must be filled.**

            **JSON Format**
```json
{{
  "response": "[Speaker 1]: Dialogue line\\n[Speaker 2]: Response line",
  "speakers": ["Speaker 1", "Speaker 2", "..."],
  "author": "Author/Creator",
  "content": "Scene description (location, purpose)",
  "source_language": "Language used (e.g., Russian)"
}}
            **Rules**

Language: All output (dialogue and summary) must be in the specified source_language.

Canon only: Use only official/canon dialogues.

Length: 3 to 12 exchanges (lines).

Context use: The context is a specific excerpt. Only use the material from this part of the canon.

No extra text: Output JSON only, no explanations.

**Examples**
            For title="Голодные игры", source_language="Russian", and context="Китнисс добровольно вызывает себя вместо Прим на сцене во время жеребьевки":
            
            ```json
            {{
            "response": "[Китнисс]: Прим! Прим! (ведущим) Я доброволец! Я доброволец! Я хочу участвовать в Играх!\n[Эффи Тринкет]: Кажется, у нас есть доброволец! ... Первый в истории Дистрикта 12 доброволец, прошу на сцену! ... Как тебя зовут?\n[Китнисс]: Китнисс Эвердин.\n[Эффи Тринкет]: Готова поспорить, это твоя сестренка. Я угадала?\n[Китнисс]: Да.",
            "speakers": ["Китнисс", "Эффи Тринкет"],
            "author": "Сьюзен Коллинз",
            "content": "Сцена жеребьёвки, где Китнисс заменяет Прим (Площадь Дистрикта 12)",
            "source_language": "Russian"
            }}

            For title="The Lord of the Rings", source_language="English", and context="Bridge of Khazad-dûm confrontation during the Fellowship's escape":
            
            ```json
            {{
            "response": "[Gandalf]: You shall not pass!\n[Balrog]: [roars in flames]",
            "speakers": ["Gandalf", "Balrog"],
            "author": "J.R.R. Tolkien",
            "content": "Bridge of Khazad-dûm confrontation during the Fellowship's escape",
            "source_language": "English"
            }}
            
            Now, generate a JSON output for:
            
            Universe (title): {title}
            Language (source_language): {source_language}
            Scene context (context): {content}

Only valid JSON output.

            """  # noqa

        return prompt
