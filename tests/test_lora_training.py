from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from justatom.training.config import LoraAdapterConfig, RuntimeConfig, TrainingMethod
from justatom.training.job import apply_lora_adapter, resolve_training_precision
from justatom.training.methods import canonical_method_config
from justatom.training.module import ContrastiveTrainingModule


def test_apply_lora_adapter_uses_feature_extraction_and_generic_all_linear(monkeypatch):
    captured = {}
    backbone = object()
    language_model = SimpleNamespace(model=backbone)

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("peft.LoraConfig", FakeLoraConfig)
    monkeypatch.setattr("peft.TaskType", SimpleNamespace(FEATURE_EXTRACTION="feature-extraction"))
    monkeypatch.setattr("peft.get_peft_model", lambda model, config: (model, config))

    result = apply_lora_adapter(language_model, LoraAdapterConfig(enabled=True))

    assert result is language_model
    assert result.model[0] is backbone
    assert captured == {
        "task_type": "feature-extraction",
        "inference_mode": False,
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "target_modules": "all-linear",
        "use_rslora": True,
        "bias": "none",
    }


def test_apply_lora_adapter_preserves_disabled_encoder():
    language_model = SimpleNamespace(model=object())

    assert apply_lora_adapter(language_model, LoraAdapterConfig()) is language_model


@pytest.mark.parametrize("accelerator", ["cpu", "mps"])
def test_auto_precision_keeps_cpu_and_mps_in_float32(accelerator):
    assert resolve_training_precision(RuntimeConfig(accelerator=accelerator)) == "32-true"


def test_auto_precision_prefers_bfloat16_on_supported_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_bf16_supported", lambda: True)

    assert resolve_training_precision(RuntimeConfig(accelerator="cuda")) == "bf16-mixed"


def test_explicit_precision_is_preserved():
    runtime = RuntimeConfig(accelerator="mps", precision="16-mixed")

    assert resolve_training_precision(runtime) == "16-mixed"


def test_lora_can_be_combined_with_atomic_method():
    config = canonical_method_config(TrainingMethod.ATOMIC)
    config = replace(config, model=replace(config.model, lora=LoraAdapterConfig(enabled=True)))

    assert not config.alpha_gate.enabled
    assert config.memory_bank.enabled
    assert config.gradient_projection.enabled
    assert config.model.lora.enabled


def test_qwen3_lora_all_linear_has_trainable_adapter_gradients():
    import torch
    from peft import PeftModel
    from transformers import Qwen3Config, Qwen3Model

    from justatom.modeling.prime import Qwen3EmbeddingModel

    backbone = Qwen3Model(
        Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
        )
    )
    language_model = Qwen3EmbeddingModel(backbone)
    apply_lora_adapter(language_model, LoraAdapterConfig(enabled=True, rank=4, alpha=8))

    assert isinstance(language_model.model, PeftModel)
    trainable = [parameter for parameter in language_model.parameters() if parameter.requires_grad]
    assert trainable

    input_ids = torch.randint(0, 64, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    query, positive = language_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pos_input_ids=input_ids,
        pos_attention_mask=attention_mask,
    )
    (query - positive.roll(1, 0)).square().mean().backward()

    assert any(parameter.grad is not None for parameter in trainable)


def test_adapter_is_saved_before_encoder_is_merged(monkeypatch, tmp_path):
    events = []

    class FakePeftModel:
        def save_pretrained(self, destination, *, safe_serialization):
            events.append(("adapter", safe_serialization))

        def merge_and_unload(self, *, safe_merge):
            events.append(("merge", safe_merge))
            return "merged-backbone"

    class FakeEncoder:
        def __init__(self):
            self.model = SimpleNamespace(model=FakePeftModel())

        def to(self, device):
            events.append(("device", device))

        def save(self, destination):
            events.append(("encoder", self.model.model))

    class Harness:
        _lora_model = ContrastiveTrainingModule._lora_model
        save_lora_adapter = ContrastiveTrainingModule.save_lora_adapter
        save_deployable_encoder = ContrastiveTrainingModule.save_deployable_encoder

        def __init__(self):
            config = canonical_method_config(TrainingMethod.VANILLA)
            self.config = replace(config, model=replace(config.model, lora=LoraAdapterConfig(enabled=True)))
            self.encoder = FakeEncoder()

    monkeypatch.setattr("peft.PeftModel", FakePeftModel)
    module = Harness()

    assert module.save_lora_adapter(tmp_path / "adapter") == tmp_path / "adapter"
    assert module.save_deployable_encoder(tmp_path / "encoder") == tmp_path / "encoder"
    assert events == [
        ("adapter", True),
        ("device", "cpu"),
        ("merge", True),
        ("encoder", "merged-backbone"),
    ]
