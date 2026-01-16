from collections import Counter
import string
from justatom.tooling import stl
import math


class keywords_metrics:

    @staticmethod
    def _compute_recall(query: str, keywords_or_phrases: list[str], **props):
        k_words = Counter(
            stl.flatten_list([kwp.lower().split(" ") for kwp in keywords_or_phrases])
        )
        q_words = (
            "".join(w for w in query if w not in string.punctuation)
            .lower()
            .strip()
            .split()
        )
        recall = sum([1.0 / math.log(1 + k_words.get(w, 1)) for w in q_words])  # type: ignore

        return recall

    @staticmethod
    def _compute_inverse_recall(
        query: str,
        keywords_or_phrases_or_content: list[str],
        stopsyms: str | None = None,
        **props,
    ):
        stopsyms = "«»\":'" if stopsyms is None else stopsyms
        stopsyms = stopsyms + string.punctuation
        if isinstance(keywords_or_phrases_or_content, list):
            k_words = Counter(
                stl.flatten_list(
                    [
                        "".join(
                            [w for w in kwp.lower().strip() if w not in stopsyms]
                        ).split()
                        for kwp in keywords_or_phrases_or_content
                    ]
                )
            )
        else:
            k_words = Counter(
                [
                    "".join([ch for ch in w.lower().strip() if ch not in stopsyms])
                    for w in keywords_or_phrases_or_content.split()
                ]
            )
        q_words = "".join(w for w in query if w not in stopsyms).lower().strip().split()
        idf_recall = sum([1.0 / math.log(1 + k_words.get(w, 1)) for w in q_words if w in k_words]) / sum(  # type: ignore
            [1.0 / math.log(1 + k_words.get(w, 1)) for w in q_words]  # type: ignore
        )
        return idf_recall


__all__ = ["keywords_metrics"]
