from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _camelize(value: str | Path | None) -> str:
    text = "" if value is None else str(value)
    parts = re.split(r"[^A-Za-z0-9]+", text)
    return "".join(part[:1].upper() + part[1:] for part in parts if part)


def _model_token(model_name_or_path: str | Path | None) -> str:
    text = "" if model_name_or_path is None else str(model_name_or_path)
    lower = text.lower()
    if "multilingual-e5-small" in lower:
        return "E5Small"
    if "multilingual-e5-base" in lower:
        return "E5Base"
    if "multilingual-e5-large" in lower:
        return "E5Large"

    filename = Path(text).name or text
    return _camelize(filename) or "Model"


def _prebuilt_collection_name(model_name_or_path: str | Path | None) -> str | None:
    text = "" if model_name_or_path is None else str(model_name_or_path)
    match = re.search(r"(Model[A-Za-z0-9]+SEPCollection[A-Za-z0-9]+(?:SEPTag[A-Za-z0-9]+)?)", text)
    if not match:
        return None

    collection_name = match.group(1)
    for delimiter in ("SEPMode", "SEPLoss", "SEPTemp"):
        if delimiter in collection_name:
            collection_name = collection_name.split(delimiter, 1)[0]
    return collection_name


def _dataset_token(dataset_name_or_path: str | Path | None) -> str:
    text = "" if dataset_name_or_path is None else str(dataset_name_or_path)
    if text == "justatom":
        return "JustAtom"

    text = re.sub(r"^[A-Za-z]+://", "", text)
    text = text.split("?", 1)[0]
    filename = Path(text).stem or Path(text).name or text
    return _camelize(filename) or "Dataset"


def build_collection_tag_payload(**kwargs: Any) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}


def resolve_collection_tag(
    collection_tag: str | None,
    *,
    auto_generate: bool = False,
    **payload: Any,
) -> str | None:
    if collection_tag:
        return _camelize(collection_tag)
    if not auto_generate:
        return None

    digest_source = json.dumps(build_collection_tag_payload(**payload), sort_keys=True, default=str)
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest().upper()[:10]
    return f"Cfg{digest}"


def resolve_collection_name(
    collection_name: str | None,
    *,
    model_name_or_path: str | Path | None = None,
    dataset_name_or_path: str | Path | None = None,
    collection_tag: str | None = None,
) -> str:
    prebuilt = _prebuilt_collection_name(model_name_or_path)
    if prebuilt:
        return prebuilt

    if collection_name and collection_name != "Document":
        base = _camelize(collection_name)
    else:
        base = f"Model{_model_token(model_name_or_path)}SEPCollection{_dataset_token(dataset_name_or_path)}"

    if collection_tag:
        return f"{base}SEPTag{_camelize(collection_tag)}"
    return base or "Document"


def resolve_artifact_dirname(collection_name: str | None) -> str:
    return _camelize(collection_name) or "Default"


def build_collection_metadata(
    *,
    collection_name: str,
    collection_tag: str | None,
    model_name_or_path: str | Path | None,
    dataset_name_or_path: str | Path | None,
    save_dir: str | Path,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "collection_name": collection_name,
        "collection_tag": collection_tag,
        "model_name_or_path": None if model_name_or_path is None else str(model_name_or_path),
        "dataset_name_or_path": None if dataset_name_or_path is None else str(dataset_name_or_path),
        "save_dir": str(save_dir),
        "payload": payload or {},
    }


def write_collection_metadata(
    *,
    save_dir: str | Path,
    metadata: dict[str, Any],
) -> tuple[Path, Path | None]:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    local_meta_path = save_path / "collection_metadata.json"
    # Some dataset configs carry Path-like values; persist them as strings.
    local_meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")

    registry_path: Path | None = None
    collection_tag = metadata.get("collection_tag")
    if collection_tag:
        registry_path = save_path / "collection_tag_registry.json"
        registry_payload = {"collection_tag": collection_tag, "collection_name": metadata.get("collection_name")}
        registry_path.write_text(json.dumps(registry_payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

    return local_meta_path, registry_path


__all__ = [
    "build_collection_metadata",
    "build_collection_tag_payload",
    "resolve_artifact_dirname",
    "resolve_collection_name",
    "resolve_collection_tag",
    "write_collection_metadata",
]
