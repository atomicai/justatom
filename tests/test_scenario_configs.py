import os
import tempfile
import unittest
from pathlib import Path

from justatom.configuring.builtins import resolve_builtin_path
from justatom.api.eval import resolve_eval_kwargs
from justatom.api.train import resolve_train_kwargs
from justatom.configuring.scenarios import load_scenario_config
from justatom.etc.errors import DocumentStoreError
from justatom.storing.weaviate import WeaviateDocStore


class ScenarioConfigTest(unittest.TestCase):
    def test_eval_uses_packaged_defaults_without_repo_config(self):
        kwargs = resolve_eval_kwargs(config={"dataset": {"name_or_path": "demo.jsonl"}})

        self.assertEqual(kwargs["collection_name"], "Document")
        self.assertEqual(kwargs["search_pipeline"], "embedding")
        self.assertEqual(kwargs["top_k"], 20)
        self.assertEqual(kwargs["dataset_name_or_path"], "demo.jsonl")
        self.assertIsNone(kwargs["split"])
        self.assertIsNone(kwargs["limit"])

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
        kwargs = resolve_eval_kwargs(
            config={
                "dataset": {
                    "name_or_path": "base.jsonl",
                    "labels_field": "labels",
                },
                "search": {"top_k": 7},
            },
            overrides={
                "search": {"top_k": 11},
                "collection": {"name": "EvalCollection"},
            },
        )

        self.assertEqual(kwargs["dataset_name_or_path"], "base.jsonl")
        self.assertEqual(kwargs["labels_field"], "labels")
        self.assertEqual(kwargs["top_k"], 11)
        self.assertEqual(kwargs["collection_name"], "EvalCollection")

    def test_builtin_eval_dataset_preset_resolves_from_packaged_defaults(self):
        kwargs = resolve_eval_kwargs(config={"dataset": {"id": "demo-eval"}})

        self.assertEqual(kwargs["dataset_name_or_path"], "builtin://datasets/demo_retrieval.jsonl")
        self.assertEqual(kwargs["labels_field"], "labels")
        self.assertEqual(kwargs["chunk_id_col"], "chunk_id")

    def test_builtin_uri_resolves_to_packaged_dataset_path(self):
        path = resolve_builtin_path("builtin://datasets/demo_retrieval.jsonl")

        self.assertTrue(path.exists())
        self.assertEqual(path.name, "demo_retrieval.jsonl")

    def test_builtin_hf_dataset_preset_resolves_from_packaged_defaults(self):
        kwargs = resolve_eval_kwargs(config={"dataset": {"id": "mlnavigator-russian-retrieval"}})

        self.assertEqual(kwargs["dataset_name_or_path"], "hf://MLNavigator/russian-retrieval")
        self.assertEqual(kwargs["labels_field"], "q")
        self.assertEqual(kwargs["content_field"], "text")
        self.assertEqual(kwargs["split"], "train")
        self.assertIsNone(kwargs["limit"])

    def test_dotted_dataset_id_override_loads_dataset_preset(self):
        kwargs = resolve_eval_kwargs(
            config_path="configs/evaluate.yaml",
            overrides={
                "dataset": {
                    "id": "mlnavigator-russian-retrieval",
                    "split": "test",
                    "limit": 500,
                }
            },
        )

        self.assertEqual(kwargs["dataset_name_or_path"], "hf://MLNavigator/russian-retrieval")
        self.assertEqual(kwargs["labels_field"], "q")
        self.assertEqual(kwargs["content_field"], "text")
        self.assertEqual(kwargs["split"], "test")
        self.assertEqual(kwargs["limit"], 500)

    def test_repo_justatom_dataset_preset_resolves_for_eval(self):
        kwargs = resolve_eval_kwargs(config={"dataset": {"id": "justatom"}})

        self.assertEqual(kwargs["dataset_name_or_path"], "justatom")
        self.assertEqual(kwargs["labels_field"], "queries")
        self.assertEqual(kwargs["content_field"], "content")
        self.assertEqual(kwargs["chunk_id_col"], "chunk_id")
        self.assertEqual(kwargs["keywords_or_phrases_field"], "keywords_or_phrases")
        self.assertIsNone(kwargs["split"])
        self.assertIsNone(kwargs["limit"])

    def test_repo_electrical_engineering_ru_dataset_preset_resolves_for_eval(self):
        kwargs = resolve_eval_kwargs(config={"dataset": {"id": "electrical-engineering-ru"}})

        self.assertEqual(
            kwargs["dataset_name_or_path"],
            "hf://d0rj/Electrical-engineering-ru",
        )
        self.assertEqual(kwargs["labels_field"], "input")
        self.assertEqual(kwargs["content_field"], "output")
        self.assertEqual(kwargs["split"], "train")
        self.assertIsNone(kwargs["limit"])

    def test_repo_boolq_ru_dataset_preset_resolves_for_eval(self):
        kwargs = resolve_eval_kwargs(config={"dataset": {"id": "boolq-ru"}})

        self.assertEqual(
            kwargs["dataset_name_or_path"],
            "hf://d0rj/boolq-ru",
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

    def test_repo_meme_russian_ir_dataset_preset_resolves_for_train(self):
        kwargs = resolve_train_kwargs(config={"dataset": {"id": "meme-russian-ir"}})

        self.assertEqual(kwargs["dataset_name_or_path"], "justatom/meme-russian-ir")
        self.assertEqual(kwargs["labels_field"], "generated")
        self.assertEqual(kwargs["content_field"], "description")
        self.assertEqual(kwargs["split"], "train")
        self.assertIsNone(kwargs["limit"])

    def test_train_supports_direct_dict_config(self):
        kwargs = resolve_train_kwargs(
            config={
                "dataset": {
                    "name_or_path": "train.jsonl",
                    "labels_field": "queries",
                },
                "model": {"name": "intfloat/multilingual-e5-base"},
                "training": {
                    "batch_size": 16,
                    "n_epochs": 3,
                    "gamma_joint": True,
                    "alpha_train_only": True,
                    "alpha_mix_weight": 0.4,
                    "margin": 0.7,
                    "include_keywords_gamma": False,
                },
                "logging": {"backend": "wandb"},
            }
        )

        self.assertEqual(kwargs["dataset_name_or_path"], "train.jsonl")
        self.assertEqual(kwargs["model_name_or_path"], "intfloat/multilingual-e5-base")
        self.assertEqual(kwargs["batch_size"], 16)
        self.assertEqual(kwargs["n_epochs"], 3)
        self.assertTrue(kwargs["gamma_joint"])
        self.assertTrue(kwargs["alpha_train_only"])
        self.assertEqual(kwargs["alpha_mix_weight"], 0.4)
        self.assertEqual(kwargs["margin"], 0.7)
        self.assertFalse(kwargs["include_keywords_gamma"])
        self.assertIsNone(kwargs["split"])
        self.assertIsNone(kwargs["limit"])
        self.assertEqual(kwargs["log_backend"], "wandb")

    def test_builtin_train_dataset_preset_resolves_from_packaged_defaults(self):
        kwargs = resolve_train_kwargs(config={"dataset": {"id": "demo-train"}})

        self.assertEqual(kwargs["dataset_name_or_path"], "builtin://datasets/demo_retrieval.jsonl")
        self.assertEqual(kwargs["labels_field"], "queries")
        self.assertEqual(kwargs["content_field"], "content")

    def test_repo_justatom_dataset_preset_resolves_for_train(self):
        kwargs = resolve_train_kwargs(config={"dataset": {"id": "justatom"}})

        self.assertEqual(kwargs["dataset_name_or_path"], "justatom")
        self.assertEqual(kwargs["labels_field"], "queries")
        self.assertEqual(kwargs["content_field"], "content")
        self.assertIsNone(kwargs["split"])
        self.assertIsNone(kwargs["limit"])
        self.assertEqual(kwargs["chunk_id_col"], "chunk_id")
        self.assertEqual(kwargs["keywords_or_phrases_field"], "keywords_or_phrases")

    def test_explicit_missing_config_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_scenario_config("evaluate", config_path="missing-evaluate.yaml")

    def test_dataset_split_and_limit_can_be_overridden_explicitly(self):
        kwargs = resolve_eval_kwargs(
            config={"dataset": {"id": "mlnavigator-russian-retrieval"}},
            overrides={"dataset": {"split": "dev|test", "limit": 25}},
        )

        self.assertEqual(kwargs["dataset_name_or_path"], "hf://MLNavigator/russian-retrieval")
        self.assertEqual(kwargs["split"], "dev|test")
        self.assertEqual(kwargs["limit"], 25)

    def test_weaviate_normalize_host_falls_back_for_empty_like_values(self):
        self.assertEqual(WeaviateDocStore._normalize_host(None), "localhost")
        self.assertEqual(WeaviateDocStore._normalize_host(""), "localhost")
        self.assertEqual(WeaviateDocStore._normalize_host("None"), "localhost")
        self.assertEqual(WeaviateDocStore._normalize_host("${WEAVIATE_HOST}"), "localhost")
        self.assertEqual(WeaviateDocStore._normalize_host("weaviate"), "weaviate")

    def test_weaviate_normalize_port_uses_defaults_for_empty_like_values(self):
        self.assertEqual(
            WeaviateDocStore._normalize_port(
                None,
                default=2211,
                setting_name="WEAVIATE_PORT",
            ),
            2211,
        )
        self.assertEqual(
            WeaviateDocStore._normalize_port(
                "",
                default=2211,
                setting_name="WEAVIATE_PORT",
            ),
            2211,
        )
        self.assertEqual(
            WeaviateDocStore._normalize_port(
                "None",
                default=2211,
                setting_name="WEAVIATE_PORT",
            ),
            2211,
        )
        self.assertEqual(
            WeaviateDocStore._normalize_port(
                "${WEAVIATE_PORT}",
                default=2211,
                setting_name="WEAVIATE_PORT",
            ),
            2211,
        )
        self.assertEqual(
            WeaviateDocStore._normalize_port(
                "2211",
                default=2211,
                setting_name="WEAVIATE_PORT",
            ),
            2211,
        )

    def test_weaviate_normalize_port_rejects_invalid_values(self):
        with self.assertRaises(DocumentStoreError):
            WeaviateDocStore._normalize_port(
                "abc",
                default=2211,
                setting_name="WEAVIATE_PORT",
            )

        with self.assertRaises(DocumentStoreError):
            WeaviateDocStore._normalize_port(
                0,
                default=2211,
                setting_name="WEAVIATE_PORT",
            )
