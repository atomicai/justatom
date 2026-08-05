from collections.abc import Iterable
from typing import Any

from justatom.storing.datasets import DatasetLoader


def documents_from_input(source: str | Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    if isinstance(source, str):
        return DatasetLoader.read(source, lazy=True)
    return [
        {
            "content": document["content"],
            "meta": document.get("meta", {}),
            "keywords_or_phrases": document.get("keywords_or_phrases", []),
        }
        for document in source
    ]


__all__ = ["documents_from_input"]
