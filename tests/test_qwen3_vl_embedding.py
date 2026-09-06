from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3VLConfig, Qwen3VLModel

from justatom.api.train import resolve_train_config
from justatom.modeling.mask import ILanguageModel
from justatom.modeling.prime import Qwen3VLEmbeddingModel
from justatom.processing.mask import IProcessor
from justatom.processing.prime import RuntimeProcessor, TrainWithContrastiveProcessor
from justatom.processing.qwen3_vl import Qwen3VLTextTokenizer
from justatom.processing.tokenizer import ITokenizer
from justatom.running.encoders import EncoderRunner
from justatom.training.config import LoraAdapterConfig
from justatom.training.job import apply_lora_adapter, build_training_loader, load_encoder
from justatom.training.module import ContrastiveTrainingModule


@pytest.fixture
def backbone():
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "rope_scaling": {"rope_type": "default", "mrope_section": [4, 2, 2]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 64,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
    )
    return Qwen3VLModel(config)


@pytest.fixture
def tokenizer():
    vocab = {"[UNK]": 0, "[PAD]": 1, "q": 2, "doc": 3, "system": 4, "user": 5, "assistant": 6}
    core = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="[UNK]",
        pad_token="[PAD]",
        chat_template=(
            "{% for message in messages %}{{ message['role'] }}: "
            "{{ message['content'][0]['text'] }}\n{% endfor %}"
            "{% if add_generation_prompt %}assistant: {% endif %}"
        ),
    )


def vl_config(geometry=False):
    name = "geometry-anchor-bank" if geometry else "vanilla"
    return resolve_train_config(config_path=f"configs/experiments/qwen3-vl-2b-lora-{name}.yaml")


def pair_batch():
    return {
        "input_ids": torch.tensor([[3, 7, 9, 2], [5, 3, 2, 1]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        "pos_input_ids": torch.tensor([[7, 5, 3, 2], [9, 4, 2, 1]]),
        "pos_attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        "doc_key_id": torch.tensor([1, 2]),
        "content_key_id": torch.tensor([11, 12]),
        "query_key_id": torch.tensor([21, 22]),
    }


def test_hub_mapping_forwards_revision(backbone):
    with patch("transformers.Qwen3VLModel.from_pretrained", return_value=backbone) as loader:
        model = ILanguageModel.load("Qwen/Qwen3-VL-Embedding-2B", revision="pinned")
    loader.assert_called_once_with("Qwen/Qwen3-VL-Embedding-2B", revision="pinned")
    assert isinstance(model, Qwen3VLEmbeddingModel)
    assert model.output_dims == 64


def test_raw_snapshot_and_saved_encoder_load_offline(backbone, tmp_path):
    backbone.save_pretrained(tmp_path / "raw")
    model = ILanguageModel.load(tmp_path / "raw", local_files_only=True).eval()
    batch = pair_batch()
    expected = model(batch["input_ids"], batch["attention_mask"])[0]
    model.save(tmp_path / "encoder")
    restored = ILanguageModel.load(tmp_path / "encoder", local_files_only=True).eval()
    actual = restored(batch["input_ids"], batch["attention_mask"])[0]
    torch.testing.assert_close(actual, expected)


def test_last_token_pool_handles_mixed_padding(backbone):
    model = Qwen3VLEmbeddingModel(backbone)
    hidden = torch.arange(3 * 5 * 64).reshape(3, 5, 64).float()
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 0, 0, 0], [0, 1, 1, 0, 0]])
    torch.testing.assert_close(model.last_token_pool(hidden, mask), hidden[torch.arange(3), torch.tensor([4, 1, 2])])


def test_forward_pair_hidden_layer_and_dimensions(backbone):
    model = Qwen3VLEmbeddingModel(backbone).eval()
    batch = pair_batch()
    query, doc = model(**{k: v for k, v in batch.items() if "key_id" not in k}, layer_idx=-2, target_dim=64)
    assert query.shape == doc.shape == (2, 64)
    torch.testing.assert_close(query.norm(dim=1), torch.ones(2))
    with pytest.raises(ValueError, match="target_dim"):
        model(batch["input_ids"], batch["attention_mask"], target_dim=32)


def test_lora_freezes_visual_and_base_with_live_text_gradients(backbone):
    model = Qwen3VLEmbeddingModel(backbone)
    apply_lora_adapter(model, LoraAdapterConfig(enabled=True, rank=4, alpha=8))
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert names and all("language_model." in name and "lora_" in name for name in names)
    assert not any("visual" in name and "lora_" in name for name, _ in model.named_parameters())
    assert not any(p.requires_grad for p in model.model.visual.parameters())
    encoder = EncoderRunner(model=model, prediction_heads=[], device="cpu")
    q, d = encoder.encode_pair(pair_batch())
    torch.nn.functional.cross_entropy(q @ d.T / 0.05, torch.arange(2)).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("targets", [("q_proj", "v_proj"), r"language_model\..*\.q_proj"])
def test_explicit_lora_targets_stay_in_language_tower(backbone, targets):
    model = Qwen3VLEmbeddingModel(backbone)
    apply_lora_adapter(model, LoraAdapterConfig(enabled=True, target_modules=targets))
    assert all("language_model." in n for n, p in model.named_parameters() if p.requires_grad)


def test_visual_only_targets_and_bias_tuning_are_rejected(backbone):
    model = Qwen3VLEmbeddingModel(backbone)
    with pytest.raises(ValueError, match="language_model"):
        apply_lora_adapter(model, LoraAdapterConfig(enabled=True, target_modules=r"visual\..*"))
    with pytest.raises(ValueError, match="bias=none"):
        apply_lora_adapter(model, LoraAdapterConfig(enabled=True, bias="all"))


def test_tokenizer_matches_chat_template_and_survives_export(tokenizer, tmp_path):
    wrapped = Qwen3VLTextTokenizer(tokenizer)
    expected = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )
    assert wrapped("q")["input_ids"] == expected
    wrapped.save_pretrained(tmp_path)
    loaded = ITokenizer.from_pretrained(tmp_path, local_files_only=True)
    assert isinstance(loaded, Qwen3VLTextTokenizer)
    assert loaded("q")["input_ids"] == expected
    assert loaded.padding_side == "right"


def test_train_and_runtime_preprocessing_match_including_truncation(tokenizer, tmp_path):
    wrapped = Qwen3VLTextTokenizer(tokenizer)
    train = TrainWithContrastiveProcessor(
        tokenizer=wrapped, max_query_seq_len=16, max_seq_len=24, queries_prefix="", pos_queries_prefix=""
    )
    rows = [{"query": "q " * 30, "content": "doc " * 30}]
    train_ds, train_names, _ = train.dataset_from_dicts(rows)
    for field, length, key in [("query", 16, "input_ids"), ("content", 24, "pos_input_ids")]:
        runtime = RuntimeProcessor(tokenizer=wrapped, max_seq_len=length)
        ds, names, _ = runtime.dataset_from_dicts([{"content": rows[0][field]}])
        torch.testing.assert_close(ds.tensors[names.index("input_ids")], train_ds.tensors[train_names.index(key)])
    train.save(tmp_path)
    restored = IProcessor.load(tmp_path)
    restored_ds, _, _ = restored.dataset_from_dicts(rows)
    for actual, expected in zip(restored_ds.tensors, train_ds.tensors, strict=True):
        torch.testing.assert_close(actual, expected)


def test_vl_configs_are_matched_except_geometry_and_artifacts():
    vanilla, geometry = vl_config(), vl_config(True)
    for field in ("model", "dataset", "experiment", "optimization", "objective", "runtime", "memory_bank"):
        assert getattr(vanilla, field) == getattr(geometry, field)
    assert not vanilla.objective.decoupled
    assert geometry.anchor_bank.enabled and not geometry.memory_bank.enabled
    assert geometry.gradient_projection.memory_weight == 0


def test_geometry_checkpointing_and_merge_roundtrip(backbone, tokenizer, tmp_path):
    from peft import PeftModel

    initial_state = {name: value.clone() for name, value in backbone.state_dict().items()}
    config = vl_config(True)
    config = replace(
        config,
        runtime=replace(config.runtime, accelerator="cpu"),
        anchor_bank=replace(config.anchor_bank, warmup_steps=0, size=8),
    )
    processor = TrainWithContrastiveProcessor(tokenizer=Qwen3VLTextTokenizer(tokenizer), queries_prefix="", pos_queries_prefix="")
    with patch("justatom.training.job.ILanguageModel.load", return_value=Qwen3VLEmbeddingModel(backbone)):
        encoder = load_encoder(config, processor)
    assert encoder.model.model.is_gradient_checkpointing
    assert next(encoder.model.parameters()).dtype == torch.float32
    module = ContrastiveTrainingModule.build(encoder, config)
    batch = pair_batch()
    first = module.compute_training_step(batch, step=0)
    assert first.anchor_loss is None
    first.loss.backward()
    _, _, teacher_q, teacher_d = module._anchor_views(batch, include_student=False)
    # Move the adapter away from the initially identical frozen teacher.
    with torch.no_grad():
        for name, p in encoder.model.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.01)
    encoder.zero_grad(set_to_none=True)
    _, _, unchanged_q, unchanged_d = module._anchor_views(batch, include_student=False)
    torch.testing.assert_close(unchanged_q, teacher_q, rtol=0, atol=0)
    torch.testing.assert_close(unchanged_d, teacher_d, rtol=0, atol=0)
    for key in ("doc_key_id", "query_key_id", "content_key_id"):
        batch[key] += 100
    output = module.compute_training_step(batch, step=1)
    assert output.anchor_loss is not None and output.anchor_loss.detach() > 0
    output.anchor_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters() if p.requires_grad)
    assert not module.anchor_bank.queries.requires_grad
    assert not module.anchor_bank.documents.requires_grad
    encoder.eval()
    expected = encoder.encode_queries(batch).detach()
    module.save_lora_adapter(tmp_path / "adapter")
    adapter_base = Qwen3VLModel(backbone.config)
    adapter_base.load_state_dict(initial_state)
    adapter_model = Qwen3VLEmbeddingModel(PeftModel.from_pretrained(adapter_base, tmp_path / "adapter")).eval()
    torch.testing.assert_close(adapter_model(batch["input_ids"], batch["attention_mask"])[0], expected)
    module.save_deployable_encoder(tmp_path / "encoder")
    restored = EncoderRunner.load(tmp_path / "encoder").eval()
    torch.testing.assert_close(restored.encode_queries(batch), expected, atol=1e-5, rtol=1e-4)


def test_exported_model_works_with_local_retrieval(backbone, tokenizer, tmp_path):
    from justatom.retrieval.embedders.huggingface import _build_local_encoder

    model = Qwen3VLEmbeddingModel(backbone).eval()
    wrapped = Qwen3VLTextTokenizer(tokenizer)
    model.save(tmp_path)
    wrapped.save_pretrained(tmp_path)
    inputs = wrapped(["q", "doc"], max_length=24, truncation=True, padding="max_length", return_tensors="pt")
    expected = model(inputs["input_ids"], inputs["attention_mask"])[0]
    local = _build_local_encoder(str(tmp_path), "cpu", max_length=24)
    try:
        actual = torch.tensor(local.encode(["q", "doc"]))
        torch.testing.assert_close(actual, expected)
    finally:
        local.close()


def test_training_loader_pins_the_tokenizer(tokenizer):
    config = vl_config()
    with (
        patch("justatom.training.job.prepare_training_data_from_config", return_value=[{"queries": "q", "content": "doc"}]),
        patch("justatom.training.job.ITokenizer.from_pretrained", return_value=Qwen3VLTextTokenizer(tokenizer)) as load,
    ):
        _, processor = build_training_loader(config)
    load.assert_called_once_with(config.model.name_or_path, revision=config.model.revision)
    assert processor.queries_prefix == processor.pos_queries_prefix == ""


@pytest.mark.parametrize("field,value", [("revision", ""), ("revision", 123), ("dtype", "bf16"), ("dtype", True)])
def test_invalid_model_loading_options_are_rejected(field, value):
    with pytest.raises(ValueError, match=f"model.{field}"):
        resolve_train_config(config={"model": {field: value}})


@pytest.mark.parametrize("device,expected", [("cpu", torch.float32), ("mps", torch.float32), ("cuda", torch.bfloat16)])
def test_training_storage_dtype_is_device_aware(backbone, tokenizer, device, expected):
    config = vl_config()
    config = replace(config, runtime=replace(config.runtime, accelerator=device, gradient_checkpointing=False))
    model = Qwen3VLEmbeddingModel(backbone)
    with (
        patch("justatom.training.job.ILanguageModel.load", return_value=model) as load,
        patch("justatom.training.job.EncoderRunner") as runner,
    ):
        load_encoder(config, TrainWithContrastiveProcessor(tokenizer=tokenizer))
    load.assert_called_once_with(model_name_or_path=config.model.name_or_path, revision=config.model.revision)
    assert model.model.get_input_embeddings().weight.dtype == expected
    assert runner.call_args.kwargs["device"] == ("cuda:0" if device == "cuda" else device)


def test_mps_defaults_to_float32_without_requiring_mps_hardware(backbone):
    model = Qwen3VLEmbeddingModel(backbone)
    with patch.object(torch.nn.Module, "to", autospec=True, return_value=model) as move:
        model.to("mps")
    move.assert_called_once_with(model, "mps", dtype=torch.float32)
