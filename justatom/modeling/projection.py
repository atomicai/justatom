"""Load EmbeddingGemma's published dense projection without executing remote code."""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from torch import nn


def model_file(source, filename, **kwargs):
    if Path(source).is_dir():
        return Path(source) / filename
    hub_kwargs = {k: kwargs[k] for k in ("revision", "token", "cache_dir", "local_files_only") if k in kwargs}
    return Path(hf_hub_download(str(source), filename, **hub_kwargs))


def dense_projection(specs):
    activations = {
        "Identity": nn.Identity,
        "Tanh": nn.Tanh,
        "ReLU": nn.ReLU,
        "GELU": nn.GELU,
    }
    layers = []
    for spec in specs:
        activation = spec["activation_function"].rsplit(".", 1)[-1]
        if activation not in activations:
            raise ValueError(f"Unsupported embedding projection activation: {activation}")
        layers.extend(
            [
                nn.Linear(spec["in_features"], spec["out_features"], bias=spec["bias"]),
                activations[activation](),
            ]
        )
    return nn.Sequential(*layers)


def load_gemma_projection(source, **kwargs):
    config = json.loads(model_file(source, "config.json", **kwargs).read_text())
    specs = config.get("justatom_projection")
    if specs is not None:
        projection = dense_projection(specs)
        projection.load_state_dict(load_file(str(model_file(source, "embedding_projection.safetensors", **kwargs))))
        return projection, specs

    modules = json.loads(model_file(source, "modules.json", **kwargs).read_text())
    kinds = [m["type"].rsplit(".", 1)[-1] for m in modules]
    if kinds not in (
        ["Transformer", "Pooling", "Dense", "Dense"],
        ["Transformer", "Pooling", "Dense", "Dense", "Normalize"],
    ):
        raise ValueError(f"Unexpected EmbeddingGemma pipeline: {kinds}")
    pooling = json.loads(model_file(source, modules[1]["path"] + "/config.json", **kwargs).read_text())
    modes = [k for k, v in pooling.items() if k.startswith("pooling_mode_") and v]
    if modes != ["pooling_mode_mean_tokens"] or not pooling.get("include_prompt", True):
        raise ValueError("EmbeddingGemma requires mean pooling including prompt tokens")
    specs = [json.loads(model_file(source, m["path"] + "/config.json", **kwargs).read_text()) for m in modules[2:4]]
    projection = dense_projection(specs)
    for index, module in enumerate(modules[2:4]):
        state = load_file(str(model_file(source, module["path"] + "/model.safetensors", **kwargs)))
        projection[2 * index].load_state_dict({key.removeprefix("linear."): value for key, value in state.items()})
    return projection, specs


def save_gemma_projection(projection, destination):
    state = {key: value.detach().cpu().contiguous() for key, value in projection.state_dict().items()}
    save_file(state, str(Path(destination) / "embedding_projection.safetensors"))
