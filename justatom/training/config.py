from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, TypeVar


class TrainingMethod(str, Enum):
    VANILLA = "vanilla"
    ATOM_GATE = "atom_gate"
    ATOMIC = "atomic"


class ExperimentRole(str, Enum):
    CANONICAL = "canonical"
    ABLATION = "ablation"


class MarginMode(str, Enum):
    OFF = "off"
    CONSTANT = "constant"
    QUERY = "query"


@dataclass(frozen=True)
class ExperimentConfig:
    role: ExperimentRole = ExperimentRole.CANONICAL
    seed: int = 42


@dataclass(frozen=True)
class LoraAdapterConfig:
    enabled: bool = False
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: str | tuple[str, ...] = "all-linear"
    use_rslora: bool = True
    bias: str = "none"


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str = "intfloat/multilingual-e5-small"
    query_prefix: str = "query:"
    content_prefix: str = "passage:"
    max_query_seq_len: int | None = None
    max_seq_len: int = 512
    lora: LoraAdapterConfig = field(default_factory=LoraAdapterConfig)


@dataclass(frozen=True)
class DatasetConfig:
    id: str | None = None
    name_or_path: str | None = None
    lazy: bool = True
    config: str | None = None
    labels_field: str = "queries"
    content_field: str = "content"
    split: str | None = None
    limit: int | None = None
    drop_columns: tuple[str, ...] = ()
    chunk_id_col: str | None = None
    keywords_col: str | None = "keywords_or_phrases"
    keywords_nested_col: str | None = None
    explanation_nested_col: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FilterConfig:
    fields: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class OptimizationConfig:
    optimizer: str = "adamw"
    lr_encoder: float = 2e-5
    lr_heads: float = 1e-2
    weight_decay: float = 0.01
    batch_size: int = 32
    grad_acc_steps: int = 1
    epochs: int = 1
    num_samples: int = 100


@dataclass(frozen=True)
class ObjectiveConfig:
    temperature: float = 0.05
    learnable_temperature: bool = True
    decoupled: bool = True
    simcse_dropout_weight: float = 0.0
    soft_fn_attract_weight: float = 0.0
    soft_fn_topk: int = 1


@dataclass(frozen=True)
class AlphaHeadConfig:
    layers: int = 1
    hidden_dim: int | None = None
    dropout: float = 0.0
    activation: str = "gelu"


@dataclass(frozen=True)
class AlphaGateConfig:
    enabled: bool = False
    supervision_weight: float = 0.3
    head: AlphaHeadConfig = field(default_factory=AlphaHeadConfig)


@dataclass(frozen=True)
class AdaptiveBankConfig:
    enabled: bool = False
    collision_threshold: float = 0.0
    collision_beta: float = 0.05


@dataclass(frozen=True)
class MarginConfig:
    mode: MarginMode = MarginMode.OFF
    base: float = 0.05
    scale: float = 0.02
    minimum: float = 0.0
    maximum: float = 0.15
    admission_beta: float = 0.05
    regularization_weight: float = 0.0


@dataclass(frozen=True)
class MemoryBankConfig:
    enabled: bool = False
    size: int = 0
    warmup_steps: int = 0
    mining: str = "all"
    hard_negatives: int = 0
    random_negatives: int = 0
    hard_warmup_steps: int = 0
    hard_ramp_steps: int = 1
    adaptive: AdaptiveBankConfig = field(default_factory=AdaptiveBankConfig)
    margin: MarginConfig = field(default_factory=MarginConfig)


@dataclass(frozen=True)
class GradientProjectionConfig:
    """One-sided gradient surgery for the ATOMIC memory objective."""

    enabled: bool = False
    memory_weight: float = 1.0
    eps: float = 1e-12


@dataclass(frozen=True)
class TelemetryConfig:
    backend: str = "csv"
    metrics_path: str | None = None
    wandb_project: str = "justatom"
    run_name: str | None = None


@dataclass(frozen=True)
class ArtifactConfig:
    save_dir: str | None = None
    collection_name: str | None = None
    collection_tag: str | None = None
    save_research_checkpoint: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    accelerator: str = "auto"
    devices: str | int = "auto"
    precision: str = "auto"
    gradient_checkpointing: bool = False


@dataclass(frozen=True)
class TrainConfig:
    method: TrainingMethod
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    alpha_gate: AlphaGateConfig = field(default_factory=AlphaGateConfig)
    memory_bank: MemoryBankConfig = field(default_factory=MemoryBankConfig)
    gradient_projection: GradientProjectionConfig = field(default_factory=GradientProjectionConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


_DATASET_METADATA_FIELDS = {
    "display_name",
    "source",
    "upstream_source",
    "manifest_path",
    "selection",
    "train",
    "eval",
    "corpus",
}


def _path(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


EnumType = TypeVar("EnumType", bound=Enum)


def _enum_value(enum_type: type[EnumType], value: Any, path: str) -> EnumType:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{path} must be one of: {allowed}") from exc


def _overlay_dataclass(current: Any, raw: Mapping[str, Any], path: str = "") -> Any:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path or 'config'} must be a mapping")

    known = {item.name for item in fields(current)}
    unknown = set(raw) - known
    if unknown:
        name = sorted(unknown)[0]
        raise ValueError(f"unknown configuration field: {_path(path, name)}")

    updates: dict[str, Any] = {}
    for name, value in raw.items():
        existing = getattr(current, name)
        field_path = _path(path, name)
        if is_dataclass(existing):
            updates[name] = _overlay_dataclass(existing, value, field_path)
        elif isinstance(existing, Enum):
            updates[name] = _enum_value(type(existing), value, field_path)
        elif isinstance(value, Path):
            updates[name] = str(value)
        elif isinstance(value, Mapping):
            updates[name] = dict(value)
        else:
            updates[name] = value
    return replace(current, **updates)


def _normalize_dataset(raw: Any) -> Any:
    if not isinstance(raw, Mapping):
        return raw
    normalized = dict(raw)
    metadata = normalized.pop("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("dataset.metadata must be a mapping")
    metadata = dict(metadata)
    if "drop_columns" in normalized:
        raw_drop_columns = normalized["drop_columns"]
        if raw_drop_columns is None:
            normalized["drop_columns"] = ()
        elif isinstance(raw_drop_columns, str):
            normalized["drop_columns"] = (raw_drop_columns,)
        else:
            normalized["drop_columns"] = tuple(raw_drop_columns)
    for name in _DATASET_METADATA_FIELDS:
        if name in normalized:
            if name in metadata:
                raise ValueError(f"dataset.{name} duplicates dataset.metadata.{name}")
            metadata[name] = normalized.pop(name)
    if metadata:
        normalized["metadata"] = metadata
    return normalized


def _normalize_model(raw: Any) -> Any:
    if not isinstance(raw, Mapping):
        return raw
    normalized = dict(raw)
    lora = normalized.get("lora")
    if isinstance(lora, Mapping):
        lora = dict(lora)
        target_modules = lora.get("target_modules")
        if isinstance(target_modules, list):
            lora["target_modules"] = tuple(target_modules)
        normalized["lora"] = lora
    return normalized


def _require_bool(value: Any, path: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")


def _require_int(value: Any, path: str, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be >= {minimum}")


def _require_number(value: Any, path: str, minimum: float | None = None, maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{path} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{path} must be <= {maximum}")


def validate_train_config(config: TrainConfig) -> None:
    _require_int(config.experiment.seed, "experiment.seed", 0)
    if not isinstance(config.model.name_or_path, str) or not config.model.name_or_path:
        raise ValueError("model.name_or_path must be a non-empty string")
    _require_int(config.model.max_seq_len, "model.max_seq_len", 1)
    if config.model.max_query_seq_len is not None:
        _require_int(config.model.max_query_seq_len, "model.max_query_seq_len", 1)
    lora = config.model.lora
    _require_bool(lora.enabled, "model.lora.enabled")
    _require_int(lora.rank, "model.lora.rank", 1)
    _require_int(lora.alpha, "model.lora.alpha", 1)
    _require_number(lora.dropout, "model.lora.dropout", 0.0, 1.0)
    _require_bool(lora.use_rslora, "model.lora.use_rslora")
    if lora.bias not in {"none", "all", "lora_only"}:
        raise ValueError("model.lora.bias must be one of: none, all, lora_only")
    if isinstance(lora.target_modules, str):
        if not lora.target_modules:
            raise ValueError("model.lora.target_modules must be a non-empty string or list of strings")
    elif isinstance(lora.target_modules, tuple):
        if not lora.target_modules or not all(isinstance(name, str) and name for name in lora.target_modules):
            raise ValueError("model.lora.target_modules must be a non-empty string or list of strings")
    else:
        raise ValueError("model.lora.target_modules must be a non-empty string or list of strings")
    if lora.enabled and config.model.name_or_path == "justatom/pfbert":
        raise ValueError("model.lora is supported only for Hugging Face encoders; justatom/pfbert is not supported")

    _require_bool(config.dataset.lazy, "dataset.lazy")
    if config.dataset.limit is not None:
        _require_int(config.dataset.limit, "dataset.limit", 0)
    if config.filters.fields is not None and not isinstance(config.filters.fields, Mapping):
        raise ValueError("filters.fields must be a mapping or null")

    if config.optimization.optimizer != "adamw":
        raise ValueError("optimization.optimizer must be adamw")
    _require_number(config.optimization.lr_encoder, "optimization.lr_encoder", 0.0)
    _require_number(config.optimization.lr_heads, "optimization.lr_heads", 0.0)
    _require_number(config.optimization.weight_decay, "optimization.weight_decay", 0.0)
    _require_int(config.optimization.batch_size, "optimization.batch_size", 1)
    _require_int(config.optimization.grad_acc_steps, "optimization.grad_acc_steps", 1)
    _require_int(config.optimization.epochs, "optimization.epochs", 1)
    if config.optimization.num_samples != -1:
        _require_int(config.optimization.num_samples, "optimization.num_samples", 1)

    _require_number(config.objective.temperature, "objective.temperature", 1e-12)
    _require_bool(config.objective.learnable_temperature, "objective.learnable_temperature")
    _require_bool(config.objective.decoupled, "objective.decoupled")
    _require_number(config.objective.simcse_dropout_weight, "objective.simcse_dropout_weight", 0.0)
    _require_number(config.objective.soft_fn_attract_weight, "objective.soft_fn_attract_weight", 0.0)
    _require_int(config.objective.soft_fn_topk, "objective.soft_fn_topk", 1)

    _require_bool(config.alpha_gate.enabled, "alpha_gate.enabled")
    _require_number(config.alpha_gate.supervision_weight, "alpha_gate.supervision_weight", 0.0)
    _require_int(config.alpha_gate.head.layers, "alpha_gate.head.layers", 1)
    if config.alpha_gate.head.hidden_dim is not None:
        _require_int(config.alpha_gate.head.hidden_dim, "alpha_gate.head.hidden_dim", 1)
    _require_number(config.alpha_gate.head.dropout, "alpha_gate.head.dropout", 0.0, 1.0)
    if config.alpha_gate.head.activation not in {"gelu", "relu", "silu", "tanh"}:
        raise ValueError("alpha_gate.head.activation must be one of: gelu, relu, silu, tanh")

    bank = config.memory_bank
    _require_bool(bank.enabled, "memory_bank.enabled")
    _require_int(bank.size, "memory_bank.size", 0)
    _require_int(bank.warmup_steps, "memory_bank.warmup_steps", 0)
    if bank.mining not in {"all", "random", "hard", "mixed"}:
        raise ValueError("memory_bank.mining must be one of: all, random, hard, mixed")
    _require_int(bank.hard_negatives, "memory_bank.hard_negatives", 0)
    _require_int(bank.random_negatives, "memory_bank.random_negatives", 0)
    _require_int(bank.hard_warmup_steps, "memory_bank.hard_warmup_steps", 0)
    _require_int(bank.hard_ramp_steps, "memory_bank.hard_ramp_steps", 1)
    _require_bool(bank.adaptive.enabled, "memory_bank.adaptive.enabled")
    _require_number(bank.adaptive.collision_beta, "memory_bank.adaptive.collision_beta", 1e-12)

    margin = bank.margin
    _require_number(margin.scale, "memory_bank.margin.scale", 0.0)
    _require_number(margin.minimum, "memory_bank.margin.minimum", 0.0)
    _require_number(margin.maximum, "memory_bank.margin.maximum", margin.minimum)
    _require_number(margin.admission_beta, "memory_bank.margin.admission_beta", 1e-12)
    _require_number(margin.regularization_weight, "memory_bank.margin.regularization_weight", 0.0)

    projection = config.gradient_projection
    _require_bool(projection.enabled, "gradient_projection.enabled")
    _require_number(projection.memory_weight, "gradient_projection.memory_weight", 0.0)
    _require_number(projection.eps, "gradient_projection.eps", 1e-30)

    if config.telemetry.backend not in {"csv", "wandb"}:
        raise ValueError("telemetry.backend must be one of: csv, wandb")
    _require_bool(config.artifacts.save_research_checkpoint, "artifacts.save_research_checkpoint")
    if not isinstance(config.runtime.devices, (str, int)) or isinstance(config.runtime.devices, bool):
        raise ValueError("runtime.devices must be a string or integer")
    if not isinstance(config.runtime.precision, str) or not config.runtime.precision:
        raise ValueError("runtime.precision must be a non-empty string")
    _require_bool(config.runtime.gradient_checkpointing, "runtime.gradient_checkpointing")


def parse_train_config(raw: Mapping[str, Any]) -> TrainConfig:
    if not isinstance(raw, Mapping):
        raise ValueError("training config must be a mapping")
    if "method" not in raw:
        raise ValueError("method is required")

    method = _enum_value(TrainingMethod, raw["method"], "method")
    from justatom.training.methods import canonical_method_config, resolve_method

    base = canonical_method_config(method)
    payload = dict(raw)
    payload.pop("method")
    if "model" in payload:
        payload["model"] = _normalize_model(payload["model"])
    if "dataset" in payload:
        payload["dataset"] = _normalize_dataset(payload["dataset"])
    config = _overlay_dataclass(base, payload)
    validate_train_config(config)
    return resolve_method(config)


def train_config_to_dict(config: TrainConfig) -> dict[str, Any]:
    def plain(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Mapping):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    return plain(asdict(config))
