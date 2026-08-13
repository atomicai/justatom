import os
import re
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import ModuleType

_training_job = ModuleType("justatom.training.job")
_training_job.TrainingJob = object
_training_job.TrainingResult = object
# Isolate config resolution from training runtime imports.
_isolated_modules = {
    "justatom.training.job": _training_job,
}
_previous_modules = {name: sys.modules.get(name) for name in _isolated_modules}
sys.modules.update(_isolated_modules)
try:
    from justatom.api.eval import _parse_args as parse_eval_args
    from justatom.api.eval import resolve_eval_kwargs
    from justatom.api.train import resolve_train_config
finally:
    for _name, _previous in _previous_modules.items():
        if _previous is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _previous

from justatom.configuring.scenarios import load_scenario_config
from justatom.etc.errors import DocumentStoreError
from justatom.storing.weaviate import WeaviateDocumentStore


class ScenarioConfigTest(unittest.TestCase):
    def test_eval_uses_packaged_defaults_without_repo_config(self):
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="justatom_no_repo_config_") as td:
            os.chdir(td)
            try:
                kwargs = resolve_eval_kwargs(config={"dataset": {"name_or_path": "demo.jsonl"}})
            finally:
                os.chdir(previous_cwd)

        retrieval = kwargs["retrieval_config"]
        self.assertEqual(retrieval["mode"], "vector")
        self.assertEqual(retrieval["embedding"]["backend"], "local")
        self.assertEqual(retrieval["store"]["collection"], "Document")
        self.assertEqual(kwargs["top_k"], 20)
        self.assertEqual(kwargs["dataset_name_or_path"], "demo.jsonl")
        self.assertIsNone(kwargs["split"])
        self.assertIsNone(kwargs["limit"])

    def test_eval_remote_config_reaches_runtime_builder(self):
        kwargs = resolve_eval_kwargs(
            config={
                "retrieval": {
                    "mode": "vector",
                    "embedding": {
                        "backend": "openai-compatible",
                        "base_url": "http://encoder:8000/v1",
                        "model": "remote-model",
                    },
                    "store": {
                        "collection": "RemoteDocs",
                        "url": "http://weaviate:8080",
                    },
                }
            }
        )

        retrieval = kwargs["retrieval_config"]
        self.assertEqual(retrieval["embedding"]["base_url"], "http://encoder:8000/v1")
        self.assertEqual(retrieval["store"]["url"], "http://weaviate:8080")

    def test_dataset_id_loads_repo_preset_and_keeps_explicit_overrides(self):
        prev_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="justatom_cfg_") as td:
            root = Path(td)
            (root / "configs").mkdir()
            (root / "configs" / "dataset").mkdir()
            (root / "configs" / "dataset" / "demo.yaml").write_text(
                "name_or_path: preset.jsonl\nlabels_field: labels\ncontent_field: preset_content\n",
                encoding="utf-8",
            )
            (root / "configs" / "evaluate.yaml").write_text(
                "dataset:\n  id: demo\n  content_field: final_content\n",
                encoding="utf-8",
            )

            os.chdir(root)
            try:
                cfg = load_scenario_config("evaluate")
            finally:
                os.chdir(prev_cwd)

        self.assertEqual(cfg["dataset"]["name_or_path"], "preset.jsonl")
        self.assertEqual(cfg["dataset"]["labels_field"], "labels")
        self.assertEqual(cfg["dataset"]["content_field"], "final_content")

    def test_eval_supports_direct_dict_config_and_overrides(self):
        config = {
            "dataset": {
                "name_or_path": "base.jsonl",
                "labels_field": "labels",
            },
            "retrieval": {"store": {"collection": "BaseCollection"}},
            "search": {"top_k": 7},
        }
        original_config = deepcopy(config)
        kwargs = resolve_eval_kwargs(
            config=config,
            overrides={
                "search": {"top_k": 11},
                "retrieval": {"store": {"collection": "EvalCollection"}},
            },
        )

        self.assertEqual(config, original_config)
        self.assertEqual(kwargs["dataset_name_or_path"], "base.jsonl")
        self.assertEqual(kwargs["labels_field"], "labels")
        self.assertEqual(kwargs["top_k"], 11)
        self.assertEqual(kwargs["retrieval_config"]["store"]["collection"], "EvalCollection")

    def test_eval_auto_collection_name_reflects_model_and_dataset(self):
        kwargs = resolve_eval_kwargs(
            config={
                "dataset": {"id": "justatom"},
                "retrieval": {
                    "embedding": {"model": "intfloat/multilingual-e5-small"},
                },
            }
        )

        self.assertEqual(
            kwargs["retrieval_config"]["store"]["collection"],
            "ModelE5SmallSEPCollectionJustAtom",
        )
        self.assertNotIn("collection_tag", kwargs)

    def test_eval_consumes_collection_tag_before_runtime_builder(self):
        kwargs = resolve_eval_kwargs(
            config={
                "dataset": {"id": "justatom"},
                "retrieval": {
                    "mode": "vector",
                    "embedding": {
                        "backend": "local",
                        "model": "intfloat/multilingual-e5-small",
                    },
                    "store": {
                        "collection": "Document",
                        "tag": "ablation-lr-1e5",
                    },
                },
            }
        )

        store = kwargs["retrieval_config"]["store"]
        self.assertEqual(store["collection"], "ModelE5SmallSEPCollectionJustAtomSEPTagAblationLr1e5")
        self.assertNotIn("collection_tag", kwargs)
        self.assertNotIn("tag", store)

    def test_eval_reuses_prebuilt_training_run_name_from_local_checkpoint_path(self):
        kwargs = resolve_eval_kwargs(
            config={
                "dataset": {"id": "justatom"},
                "retrieval": {
                    "embedding": {
                        "model": "weights/ModelE5SmallSEPCollectionJustAtomSEPModeGammaOnlySEPLossContrastive/BiGamma/epoch1"
                    },
                },
            }
        )

        self.assertEqual(
            kwargs["retrieval_config"]["store"]["collection"],
            "ModelE5SmallSEPCollectionJustAtom",
        )
        self.assertNotIn("collection_tag", kwargs)

    def test_eval_explicit_collection_name_beats_auto_name(self):
        kwargs = resolve_eval_kwargs(
            config={
                "dataset": {"id": "justatom"},
                "retrieval": {
                    "embedding": {"model": "intfloat/multilingual-e5-small"},
                    "store": {"collection": "ManualCollection"},
                },
            }
        )

        self.assertEqual(kwargs["retrieval_config"]["store"]["collection"], "ManualCollection")

    def test_eval_keyword_config_does_not_require_embedding_model(self):
        kwargs = resolve_eval_kwargs(
            config={
                "retrieval": {
                    "mode": "keyword",
                    "embedding": {"model": None},
                }
            }
        )

        self.assertNotIn("embedding", kwargs["retrieval_config"])

    def test_eval_cli_overlays_explicit_retrieval_options(self):
        kwargs = parse_eval_args(
            [
                "--search-mode",
                "hybrid",
                "--embedding-backend",
                "openai-compatible",
                "--embedding-base-url",
                "http://encoder:8000/v1",
                "--embedding-api-key",
                "secret",
                "--embedding-model",
                "remote-model",
                "--query-prefix",
                "query: ",
                "--document-prefix",
                "passage: ",
                "--collection-name",
                "CliDocs",
                "--weaviate-url",
                "http://weaviate:8080",
                "--weaviate-grpc-port",
                "50052",
                "--alpha",
                "0.7",
            ]
        )

        retrieval = kwargs["retrieval_config"]
        self.assertEqual(retrieval["mode"], "hybrid")
        self.assertEqual(retrieval["alpha"], 0.7)
        self.assertEqual(
            retrieval["embedding"],
            {
                "backend": "openai-compatible",
                "base_url": "http://encoder:8000/v1",
                "api_key": "secret",
                "model": "remote-model",
                "query_prefix": "query: ",
                "document_prefix": "passage: ",
                "batch_size": 64,
                "max_length": 512,
            },
        )
        self.assertEqual(
            retrieval["store"],
            {
                "collection": "CliDocs",
                "url": "http://weaviate:8080",
                "grpc_port": 50052,
            },
        )

    def test_eval_cli_search_mode_choices_are_keyword_vector_and_hybrid(self):
        for mode in ("keyword", "vector", "hybrid"):
            with self.subTest(mode=mode):
                kwargs = parse_eval_args(["--search-mode", mode])
                self.assertEqual(kwargs["retrieval_config"]["mode"], mode)

        with self.assertRaises(SystemExit):
            parse_eval_args(["--search-mode", "embedding"])

    def test_eval_cli_rejects_retired_retrieval_flags(self):
        for option, value in (
            ("--search-pipeline", "embedding"),
            ("--model-name-or-path", "model"),
            ("--content-prefix", "passage: "),
            ("--weaviate-host", "localhost"),
            ("--weaviate-port", "2211"),
        ):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                parse_eval_args([option, value])

    def test_eval_rejects_retired_config_sections(self):
        retired_configs = (
            {"model": {"name": "legacy-model"}},
            {"collection": {"name": "LegacyDocs"}},
            {"weaviate": {"host": "localhost"}},
            {"props": {"alpha": 0.5}},
            {"search": {"pipeline": "embedding"}},
        )

        for config in retired_configs:
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "retired"):
                resolve_eval_kwargs(config=config)

    def test_builtin_eval_dataset_preset_resolves_from_packaged_defaults(self):
        kwargs = resolve_eval_kwargs(config={"dataset": {"id": "demo-eval"}})

        self.assertEqual(kwargs["dataset_name_or_path"], "demo")
        self.assertTrue(kwargs["dataset_lazy"])
        self.assertEqual(kwargs["labels_field"], "labels")
        self.assertEqual(kwargs["chunk_id_col"], "chunk_id")

    def test_dotted_dataset_id_override_loads_dataset_preset(self):
        kwargs = resolve_eval_kwargs(
            config_path="configs/evaluate.yaml",
            overrides={
                "dataset": {
                    "id": "boolq-ru",
                    "split": "validation|train",
                    "limit": 500,
                }
            },
        )

        self.assertEqual(kwargs["dataset_name_or_path"], "d0rj/boolq-ru")
        self.assertEqual(kwargs["labels_field"], "question")
        self.assertEqual(kwargs["content_field"], "passage")
        self.assertEqual(kwargs["split"], "validation|train")
        self.assertEqual(kwargs["limit"], 500)

    def test_repo_justatom_dataset_preset_resolves_for_eval(self):
        kwargs = resolve_eval_kwargs(config={"dataset": {"id": "justatom"}})

        self.assertEqual(kwargs["dataset_name_or_path"], ".data/polaroids.ai.data.json")
        self.assertFalse(kwargs["dataset_lazy"])
        self.assertEqual(kwargs["labels_field"], "queries")
        self.assertEqual(kwargs["content_field"], "content")
        self.assertEqual(kwargs["chunk_id_col"], "chunk_id")
        self.assertEqual(kwargs["keywords_or_phrases_field"], "keywords_or_phrases")
        self.assertIsNone(kwargs["split"])
        self.assertIsNone(kwargs["limit"])

    def test_repo_miracl_ru_dataset_preset_resolves_for_eval(self):
        kwargs = resolve_eval_kwargs(config={"dataset": {"id": "miracl-ru"}})

        self.assertEqual(kwargs["dataset_name_or_path"], ".data/retrieval/miracl-ru-train.jsonl")
        self.assertEqual(kwargs["labels_field"], "queries")
        self.assertEqual(kwargs["content_field"], "content")
        self.assertEqual(kwargs["chunk_id_col"], "chunk_id")
        self.assertIsNone(kwargs["split"])
        self.assertIsNone(kwargs["limit"])

    def test_repo_electrical_engineering_ru_dataset_preset_resolves_for_eval(self):
        kwargs = resolve_eval_kwargs(config={"dataset": {"id": "electrical-engineering-ru"}})

        self.assertEqual(
            kwargs["dataset_name_or_path"],
            "d0rj/Electrical-engineering-ru",
        )
        self.assertEqual(kwargs["labels_field"], "input")
        self.assertEqual(kwargs["content_field"], "output")
        self.assertEqual(kwargs["split"], "train")
        self.assertIsNone(kwargs["limit"])

    def test_repo_boolq_ru_dataset_preset_resolves_for_eval(self):
        kwargs = resolve_eval_kwargs(config={"dataset": {"id": "boolq-ru"}})

        self.assertEqual(
            kwargs["dataset_name_or_path"],
            "d0rj/boolq-ru",
        )
        self.assertEqual(kwargs["labels_field"], "question")
        self.assertEqual(kwargs["content_field"], "passage")
        self.assertEqual(kwargs["split"], "train")
        self.assertIsNone(kwargs["limit"])

    def test_repo_meme_russian_ir_dataset_preset_resolves_for_eval(self):
        kwargs = resolve_eval_kwargs(config={"dataset": {"id": "meme-russian-ir"}})

        self.assertEqual(kwargs["dataset_name_or_path"], "justatom/meme-russian-ir")
        self.assertEqual(kwargs["labels_field"], "generated")
        self.assertEqual(kwargs["content_field"], "description")
        self.assertEqual(kwargs["split"], "train")
        self.assertIsNone(kwargs["limit"])

    def test_mmarco_config_is_separate_from_hugging_face_dataset_id(self):
        kwargs = resolve_eval_kwargs(config={"dataset": {"id": "mmarco-russian"}})

        self.assertEqual(kwargs["dataset_name_or_path"], "unicamp-dl/mmarco")
        self.assertEqual(kwargs["dataset_config"], "russian")
        self.assertTrue(kwargs["dataset_lazy"])

    def test_repo_meme_russian_ir_dataset_preset_resolves_for_train(self):
        config = resolve_train_config(config={"dataset": {"id": "meme-russian-ir"}})

        self.assertEqual(config.dataset.name_or_path, "justatom/meme-russian-ir")
        self.assertEqual(config.dataset.labels_field, "generated")
        self.assertEqual(config.dataset.content_field, "description")
        self.assertEqual(config.dataset.split, "train")
        self.assertIsNone(config.dataset.limit)

    def test_repo_mmarco_ru_selected_dataset_preset_resolves_for_train(self):
        config = resolve_train_config(config={"dataset": {"id": "mmarco-ru-selected"}})

        self.assertEqual(
            config.dataset.name_or_path,
            "justatom/mmarco-ru-selected",
        )
        self.assertEqual(config.dataset.labels_field, "query")
        self.assertEqual(config.dataset.content_field, "positive")
        self.assertEqual(config.dataset.chunk_id_col, "pair_id")
        self.assertEqual(config.dataset.split, "train")
        self.assertIsNone(config.dataset.limit)

    def test_repo_mmarco_ru_selected_train_and_dev_presets_resolve(self):
        train_config = resolve_train_config(config={"dataset": {"id": "mmarco-ru-selected-train"}})
        eval_kwargs = resolve_eval_kwargs(config={"dataset": {"id": "mmarco-ru-selected-dev"}})

        self.assertEqual(
            train_config.dataset.name_or_path,
            "justatom/mmarco-ru-selected",
        )
        self.assertEqual(train_config.dataset.labels_field, "query")
        self.assertEqual(train_config.dataset.content_field, "positive")
        self.assertEqual(train_config.dataset.chunk_id_col, "pair_id")
        self.assertEqual(train_config.dataset.split, "train")

        self.assertEqual(
            eval_kwargs["dataset_name_or_path"],
            "justatom/mmarco-ru-selected",
        )
        self.assertEqual(eval_kwargs["labels_field"], "query")
        self.assertEqual(eval_kwargs["content_field"], "positive")
        self.assertEqual(eval_kwargs["chunk_id_col"], "pair_id")
        self.assertEqual(eval_kwargs["split"], "dev")

    def test_repo_mmarco_ru_selected_preset_records_selection_contract(self):
        cfg = load_scenario_config("train", config={"dataset": {"id": "mmarco-ru-selected"}})
        dataset = cfg["dataset"]

        self.assertEqual(dataset["display_name"], "mMARCO-ru-selected")
        self.assertEqual(dataset["source"], "justatom/mmarco-ru-selected")
        self.assertEqual(dataset["upstream_source"], "ir_datasets:mmarco/v2/ru")
        self.assertEqual(dataset["split"], "train")
        self.assertEqual(dataset["selection"]["seed"], 42)
        self.assertEqual(dataset["selection"]["train_rows"], 50_000)
        self.assertEqual(dataset["selection"]["dev_queries"], 5_000)
        self.assertEqual(dataset["selection"]["eval_corpus_docs"], 50_000)
        self.assertEqual(dataset["train"]["name_or_path"], "justatom/mmarco-ru-selected")
        self.assertEqual(dataset["train"]["split"], "train")
        self.assertEqual(dataset["eval"]["name_or_path"], "justatom/mmarco-ru-selected")
        self.assertEqual(dataset["eval"]["split"], "dev")
        self.assertEqual(dataset["corpus"]["name_or_path"], "justatom/mmarco-ru-selected")
        self.assertEqual(dataset["corpus"]["split"], "corpus")
        self.assertEqual(
            dataset["manifest_path"],
            "https://huggingface.co/datasets/justatom/mmarco-ru-selected/resolve/main/manifest.json",
        )

    def test_train_supports_direct_dict_config(self):
        config = resolve_train_config(
            config={
                "method": "atomic",
                "dataset": {
                    "name_or_path": "train.jsonl",
                    "labels_field": "queries",
                },
                "model": {"name_or_path": "intfloat/multilingual-e5-base"},
                "optimization": {
                    "batch_size": 16,
                    "grad_acc_steps": 3,
                    "epochs": 3,
                },
                "objective": {"temperature": 0.07},
                "gradient_projection": {"memory_weight": 0.4},
                "telemetry": {"backend": "wandb"},
            }
        )

        self.assertEqual(config.dataset.name_or_path, "train.jsonl")
        self.assertEqual(config.model.name_or_path, "intfloat/multilingual-e5-base")
        self.assertEqual(config.optimization.batch_size, 16)
        self.assertEqual(config.optimization.grad_acc_steps, 3)
        self.assertEqual(config.optimization.epochs, 3)
        self.assertEqual(config.objective.temperature, 0.07)
        self.assertEqual(config.gradient_projection.memory_weight, 0.4)
        self.assertIsNone(config.dataset.split)
        self.assertIsNone(config.dataset.limit)
        self.assertEqual(config.telemetry.backend, "wandb")

    def test_builtin_train_dataset_preset_resolves_from_packaged_defaults(self):
        config = resolve_train_config(config={"dataset": {"id": "demo-train"}})

        self.assertEqual(config.dataset.name_or_path, "demo")
        self.assertTrue(config.dataset.lazy)
        self.assertEqual(config.dataset.labels_field, "queries")
        self.assertEqual(config.dataset.content_field, "content")

    def test_repo_justatom_dataset_preset_resolves_for_train(self):
        config = resolve_train_config(config={"dataset": {"id": "justatom"}})

        self.assertEqual(config.dataset.name_or_path, ".data/polaroids.ai.data.json")
        self.assertFalse(config.dataset.lazy)
        self.assertEqual(config.dataset.labels_field, "queries")
        self.assertEqual(config.dataset.content_field, "content")
        self.assertIsNone(config.dataset.split)
        self.assertIsNone(config.dataset.limit)
        self.assertEqual(config.dataset.chunk_id_col, "chunk_id")
        self.assertEqual(config.dataset.keywords_col, "keywords_or_phrases")

    def test_train_collection_tag_is_appended_to_auto_name(self):
        config = resolve_train_config(
            config={
                "dataset": {"id": "justatom"},
                "artifacts": {
                    "collection_name": "JustAtomCorpus",
                    "collection_tag": "adamw-lr2e5",
                },
            }
        )

        self.assertEqual(config.artifacts.collection_name, "JustAtomCorpus")
        self.assertEqual(config.artifacts.collection_tag, "adamw-lr2e5")

    def test_train_defaults_include_temperature_and_grad_acc_steps(self):
        config = resolve_train_config(config={"dataset": {"id": "justatom"}})

        self.assertEqual(config.objective.temperature, 0.05)
        self.assertEqual(config.optimization.grad_acc_steps, 1)

    def test_explicit_missing_config_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_scenario_config("evaluate", config_path="missing-evaluate.yaml")

    def test_dataset_split_and_limit_can_be_overridden_explicitly(self):
        kwargs = resolve_eval_kwargs(
            config={"dataset": {"id": "boolq-ru"}},
            overrides={"dataset": {"split": "validation|train", "limit": 25}},
        )

        self.assertEqual(kwargs["dataset_name_or_path"], "d0rj/boolq-ru")
        self.assertEqual(kwargs["split"], "validation|train")
        self.assertEqual(kwargs["limit"], 25)

    def test_weaviate_normalize_host_falls_back_for_empty_like_values(self):
        self.assertEqual(WeaviateDocumentStore._normalize_host(None), "localhost")
        self.assertEqual(WeaviateDocumentStore._normalize_host(""), "localhost")
        self.assertEqual(WeaviateDocumentStore._normalize_host("None"), "localhost")
        self.assertEqual(WeaviateDocumentStore._normalize_host("${WEAVIATE_HOST}"), "localhost")
        self.assertEqual(WeaviateDocumentStore._normalize_host("weaviate"), "weaviate")

    def test_weaviate_normalize_port_uses_defaults_for_empty_like_values(self):
        self.assertEqual(
            WeaviateDocumentStore._normalize_port(
                None,
                default=2211,
                setting_name="WEAVIATE_PORT",
            ),
            2211,
        )
        self.assertEqual(
            WeaviateDocumentStore._normalize_port(
                "",
                default=2211,
                setting_name="WEAVIATE_PORT",
            ),
            2211,
        )
        self.assertEqual(
            WeaviateDocumentStore._normalize_port(
                "None",
                default=2211,
                setting_name="WEAVIATE_PORT",
            ),
            2211,
        )
        self.assertEqual(
            WeaviateDocumentStore._normalize_port(
                "${WEAVIATE_PORT}",
                default=2211,
                setting_name="WEAVIATE_PORT",
            ),
            2211,
        )
        self.assertEqual(
            WeaviateDocumentStore._normalize_port(
                "2211",
                default=2211,
                setting_name="WEAVIATE_PORT",
            ),
            2211,
        )

    def test_weaviate_normalize_port_rejects_invalid_values(self):
        with self.assertRaises(DocumentStoreError):
            WeaviateDocumentStore._normalize_port(
                "abc",
                default=2211,
                setting_name="WEAVIATE_PORT",
            )

        with self.assertRaises(DocumentStoreError):
            WeaviateDocumentStore._normalize_port(
                0,
                default=2211,
                setting_name="WEAVIATE_PORT",
            )
