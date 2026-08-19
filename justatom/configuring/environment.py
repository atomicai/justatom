from __future__ import annotations

import os

import dotenv

_HF_TOKEN_ALIASES = (
    "HUGGINGFACE_HUB_TOKEN",
    "HF_HUB_TOKEN",
    "HUGGINGFACE_API_KEY",
    "HF_API_KEY",
)


def load_runtime_environment() -> None:
    """Load `.env` and expose supported Hugging Face aliases to the Hub."""

    dotenv.load_dotenv()
    if os.environ.get("HF_TOKEN"):
        return
    for name in _HF_TOKEN_ALIASES:
        token = os.environ.get(name)
        if token and token.strip():
            os.environ["HF_TOKEN"] = token.strip()
            return


__all__ = ["load_runtime_environment"]
