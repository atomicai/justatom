"""Architecture-level checks with small real Transformers, no downloaded weights."""

import json

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from justatom.api.train import resolve_train_config
from justatom.modeling.mask import ILanguageModel
from justatom.modeling.prime import EmbeddingGemmaModel, Nemotron3EmbeddingModel, Qwen3EmbeddingModel, WeMMEmbeddingModel
from justatom.modeling.projection import dense_projection
from justatom.processing.prime import RuntimeProcessor, TrainWithContrastiveProcessor
from justatom.processing.tokenizer import ITokenizer
from justatom.processing.wemm import WeMMTextTokenizer
from justatom.running.encoders import EncoderRunner
from justatom.training.config import LoraAdapterConfig
from justatom.training.job import apply_lora_adapter
from justatom.training.module import ContrastiveTrainingModule


@pytest.fixture(params=["qwen", "nemotron", "gemma", "wemm"])
def model(request):
    torch.manual_seed(31)
    options = {
        "vocab_size": 64,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
    }
    if request.param == "qwen":
        from transformers import Qwen3Config, Qwen3Model

        return Qwen3EmbeddingModel(Qwen3Model(Qwen3Config(**options)))
    if request.param == "nemotron":
        from transformers import Ministral3Config, Ministral3Model

        return Nemotron3EmbeddingModel(Ministral3Model(Ministral3Config(**options, is_causal=False)))
    if request.param == "gemma":
        from transformers import Gemma3TextConfig, Gemma3TextModel

        specs = [
            {
                "in_features": 64,
                "out_features": 96,
                "bias": False,
                "activation_function": "torch.nn.Identity",
            },
            {
                "in_features": 96,
                "out_features": 32,
                "bias": False,
                "activation_function": "torch.nn.Identity",
            },
        ]
        return EmbeddingGemmaModel(
            Gemma3TextModel(
                Gemma3TextConfig(
                    **options,
                    use_bidirectional_attention=True,
                    layer_types=["full_attention", "sliding_attention"],
                    sliding_window=16,
                )
            ),
            projection=dense_projection(specs),
            projection_specs=specs,
        )
    from transformers import Qwen3_5Config, Qwen3_5Model

    return WeMMEmbeddingModel(
        Qwen3_5Model(
            Qwen3_5Config(
                text_config={
                    **options,
                    "layer_types": ["linear_attention", "full_attention"],
                    "linear_num_key_heads": 4,
                    "linear_num_value_heads": 4,
                    "linear_key_head_dim": 16,
                    "linear_value_head_dim": 16,
                    "rope_parameters": {
                        "rope_type": "default",
                        "rope_theta": 10000,
                        "partial_rotary_factor": 0.5,
                        "mrope_section": [1, 1, 2],
                    },
                },
                vision_config={
                    "depth": 1,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_heads": 4,
                    "out_hidden_size": 64,
                    "num_position_embeddings": 16,
                },
            )
        )
    )


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


def test_embedding_is_padding_invariant_and_normalized(model):
    model.eval()
    ids = torch.tensor([[3, 5, 9]])
    plain = model(ids, torch.ones_like(ids))[0]
    padded = model(torch.tensor([[3, 5, 9, 1, 1]]), torch.tensor([[1, 1, 1, 0, 0]]))[0]
    torch.testing.assert_close(plain, padded, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(plain.norm(dim=-1), torch.ones(1))
    assert plain.shape == (1, model.output_dims)


def test_bidirectional_backbones_see_future_tokens(model):
    if not isinstance(model, (Nemotron3EmbeddingModel, EmbeddingGemmaModel)):
        pytest.skip("Only bidirectional embeddings")
    model.eval()
    first = model.model(input_ids=torch.tensor([[3, 5, 9]]), attention_mask=torch.ones(1, 3)).last_hidden_state[:, 0]
    changed = model.model(input_ids=torch.tensor([[3, 5, 4]]), attention_mask=torch.ones(1, 3)).last_hidden_state[:, 0]
    assert not torch.allclose(first, changed, atol=1e-5)


def test_lora_anchor_gradients_frozen_teacher_and_merged_export(model, tmp_path):
    model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.model.enable_input_require_grads()
    apply_lora_adapter(model, LoraAdapterConfig(enabled=True, rank=4, alpha=8))
    assert all("lora_" in name for name, p in model.named_parameters() if p.requires_grad)
    assert not any("visual" in name and "lora_" in name for name, _ in model.named_parameters())
    encoder = EncoderRunner(model=model, prediction_heads=[], device="cpu")
    config = resolve_train_config(
        config={
            "method": "atomic",
            "experiment": {"role": "ablation"},
            "model": {"lora": {"enabled": True, "rank": 4, "alpha": 8}},
            "memory_bank": {"enabled": False, "size": 0},
            "anchor_bank": {"enabled": True, "size": 8, "warmup_steps": 0},
            "gradient_projection": {"enabled": True, "memory_weight": 0.0},
        }
    )
    module = ContrastiveTrainingModule.build(encoder, config)
    batch = pair_batch()
    first = module.compute_training_step(batch, step=0)
    first.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
    _, _, base_q, base_d = module._anchor_views(batch, include_student=False)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.03)
    _, _, still_q, still_d = module._anchor_views(batch, include_student=False)
    torch.testing.assert_close(still_q, base_q, rtol=0, atol=0)
    torch.testing.assert_close(still_d, base_d, rtol=0, atol=0)
    for key in ("doc_key_id", "content_key_id", "query_key_id"):
        batch[key] += 100
    model.zero_grad(set_to_none=True)
    second = module.compute_training_step(batch, step=1)
    assert second.anchor_loss is not None and second.anchor_loss.item() > 0
    second.anchor_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
    expected = encoder.eval().encode_queries(batch).detach()
    module.save_deployable_encoder(tmp_path)
    restored = EncoderRunner.load(tmp_path).eval()
    torch.testing.assert_close(restored.encode_queries(batch), expected, atol=2e-5, rtol=1e-4)


def test_gemma_raw_sentence_transformer_projection_load(model, tmp_path):
    if not isinstance(model, EmbeddingGemmaModel):
        pytest.skip("Gemma projection only")
    model.model.config.justatom_projection = None
    model.model.save_pretrained(tmp_path)
    specs = [
        {
            "in_features": 64,
            "out_features": 96,
            "bias": False,
            "activation_function": "torch.nn.Identity",
        },
        {
            "in_features": 96,
            "out_features": 32,
            "bias": False,
            "activation_function": "torch.nn.Identity",
        },
    ]
    modules = [
        {"type": "sentence_transformers.models.Transformer", "path": ""},
        {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"},
    ]
    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "1_Pooling/config.json").write_text(json.dumps({"pooling_mode_mean_tokens": True}))
    for index, spec in enumerate(specs):
        path = tmp_path / f"{index + 2}_Dense"
        path.mkdir()
        (path / "config.json").write_text(json.dumps(spec))
        save_file(
            {"linear.weight": model.projection[index * 2].weight},
            str(path / "model.safetensors"),
        )
        modules.append({"type": "sentence_transformers.models.Dense", "path": path.name})
    (tmp_path / "modules.json").write_text(json.dumps(modules))
    model.model.config.justatom_projection = specs
    restored = ILanguageModel.load(tmp_path, local_files_only=True).eval()
    batch = pair_batch()
    torch.testing.assert_close(
        restored(batch["input_ids"], batch["attention_mask"])[0],
        model.eval()(batch["input_ids"], batch["attention_mask"])[0],
    )


def test_wemm_format_preserves_embedding_token_after_truncation_and_export(tmp_path):
    core = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "q": 2, "doc": 3, "<embedding>": 4},
            unk_token="[UNK]",
        )
    )
    core.pre_tokenizer = Whitespace()
    core.post_processor = TemplateProcessing(single="$A <embedding>", special_tokens=[("<embedding>", 4)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="[UNK]",
        pad_token="[PAD]",
        chat_template="{% for m in messages %}user: {{ m['content'][0]['text'] }}\n{% endfor %}",
    )
    wrapped = WeMMTextTokenizer(tokenizer)
    train = TrainWithContrastiveProcessor(
        tokenizer=wrapped,
        max_query_seq_len=8,
        max_seq_len=12,
        queries_prefix="",
        pos_queries_prefix="",
    )
    rows = [{"query": "q " * 30, "content": "doc " * 30}]
    ds, names, _ = train.dataset_from_dicts(rows)
    for field, length, key in [
        ("query", 8, "input_ids"),
        ("content", 12, "pos_input_ids"),
    ]:
        runtime = RuntimeProcessor(tokenizer=wrapped, max_seq_len=length)
        runtime_ds, runtime_names, _ = runtime.dataset_from_dicts([{"content": rows[0][field]}])
        ids = ds.tensors[names.index(key)]
        assert ids[0, -1].item() == 4
        torch.testing.assert_close(ids, runtime_ds.tensors[runtime_names.index("input_ids")])
    wrapped.save_pretrained(tmp_path)
    restored = ITokenizer.from_pretrained(tmp_path)
    assert isinstance(restored, WeMMTextTokenizer)
    assert restored("q") == wrapped("q")


@pytest.mark.parametrize(
    "name,query,document",
    [
        ("nvidia/Nemotron-3-Embed-1B-BF16", "query: ", "passage: "),
        (
            "google/embeddinggemma-300m",
            "task: search result | query: ",
            "title: none | text: ",
        ),
        ("tencent/WeMM-Embedding-2B", "", ""),
        (
            "Qwen/Qwen3-Embedding-4B",
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
            "",
        ),
    ],
)
def test_native_prompts_agree_between_training_and_retrieval(name, query, document, monkeypatch):
    from justatom.retrieval import runtime

    captured = {}
    monkeypatch.setattr(runtime, "HuggingFaceEmbedder", lambda **kwargs: captured.update(kwargs))
    config = resolve_train_config(config={"model": {"name_or_path": name}})
    runtime._build_embedder({"backend": "local", "model": name})
    assert config.model.query_prefix == captured["profile"].query_prefix == query
    assert config.model.content_prefix == captured["profile"].document_prefix == document
    config = resolve_train_config(
        config={
            "model": {
                "name_or_path": name,
                "query_prefix": "",
                "content_prefix": "custom: ",
            }
        }
    )
    runtime._build_embedder(
        {
            "backend": "local",
            "model": name,
            "query_prefix": "",
            "document_prefix": "custom: ",
        }
    )
    assert config.model.query_prefix == captured["profile"].query_prefix == ""
    assert config.model.content_prefix == captured["profile"].document_prefix == "custom: "


def test_exported_processor_prompts_override_family_defaults(tmp_path):
    from justatom.configuring.embeddings import native_embedding_prefixes

    (tmp_path / "config.json").write_text(json.dumps({"klass": "EmbeddingGemmaModel"}))
    (tmp_path / "processor_config.json").write_text(json.dumps({"queries_prefix": "custom:", "pos_queries_prefix": ""}))
    assert native_embedding_prefixes(tmp_path) == ("custom: ", "")


def test_gemma_refuses_missing_projection_and_half_precision(model):
    if not isinstance(model, EmbeddingGemmaModel):
        pytest.skip("Gemma-specific precision/projection requirements")
    with pytest.raises(ValueError, match="projection"):
        EmbeddingGemmaModel(model.model)
    with pytest.raises(ValueError, match="float16"):
        model.to(dtype=torch.float16)
