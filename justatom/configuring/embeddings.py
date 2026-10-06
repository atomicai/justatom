"""Native retrieval prompts shared by training and evaluation."""

import json
from pathlib import Path

QWEN_QUERY = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
NATIVE_PREFIXES = {
    "Qwen/Qwen3-Embedding-0.6B": (QWEN_QUERY, ""),
    "Qwen/Qwen3-Embedding-4B": (QWEN_QUERY, ""),
    "Qwen/Qwen3-VL-Embedding-2B": ("", ""),
    "nvidia/Nemotron-3-Embed-1B-BF16": ("query: ", "passage: "),
    "google/embeddinggemma-300m": (
        "task: search result | query: ",
        "title: none | text: ",
    ),
    "tencent/WeMM-Embedding-2B": ("", ""),
}


def native_embedding_prefixes(model):
    if not isinstance(model, (str, Path)):
        return None
    if str(model) in NATIVE_PREFIXES:
        return NATIVE_PREFIXES[str(model)]
    config_file = Path(model) / "config.json"
    if not config_file.is_file():
        return None
    config = json.loads(config_file.read_text())
    # Exports retain customized prompts, including an explicitly empty prefix.
    processor_file = Path(model) / "processor_config.json"
    if processor_file.is_file():
        processor = json.loads(processor_file.read_text())
        if "queries_prefix" in processor and "pos_queries_prefix" in processor:
            return tuple(p.rstrip() + " " if p else "" for p in (processor["queries_prefix"], processor["pos_queries_prefix"]))
    name = config.get("justatom_model_name", config.get("_name_or_path", ""))
    if name in NATIVE_PREFIXES:
        return NATIVE_PREFIXES[name]
    klass_names = {
        "Qwen3EmbeddingModel": "Qwen/Qwen3-Embedding-4B",
        "Qwen3VLEmbeddingModel": "Qwen/Qwen3-VL-Embedding-2B",
        "Nemotron3EmbeddingModel": "nvidia/Nemotron-3-Embed-1B-BF16",
        "EmbeddingGemmaModel": "google/embeddinggemma-300m",
        "WeMMEmbeddingModel": "tencent/WeMM-Embedding-2B",
    }
    klass = config.get("klass")
    if klass in klass_names:
        return NATIVE_PREFIXES[klass_names[klass]]
    model_type = config.get("model_type")
    if model_type == "qwen3":
        return NATIVE_PREFIXES["Qwen/Qwen3-Embedding-4B"]
    if model_type == "ministral3" and config.get("is_causal") is False:
        return NATIVE_PREFIXES["nvidia/Nemotron-3-Embed-1B-BF16"]
    if model_type == "gemma3_text" and config.get("use_bidirectional_attention"):
        return NATIVE_PREFIXES["google/embeddinggemma-300m"]
    if model_type == "qwen3_vl" or (model_type == "qwen3_5" and config.get("matryoshka_dimensions")):
        return ("", "")
    return None
