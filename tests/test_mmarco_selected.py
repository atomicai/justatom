import json
import tempfile
import unittest
from pathlib import Path

from justatom.tooling import mmarco_selected


class MMarcoSelectedTests(unittest.TestCase):
    def test_pair_filter_rejects_empty_and_out_of_range_text(self):
        cfg = mmarco_selected.SelectionConfig(
            query_min_chars=3,
            query_max_chars=10,
            positive_min_chars=5,
            positive_max_chars=20,
        )

        self.assertTrue(mmarco_selected.valid_pair(" запрос ", " хороший текст ", cfg))
        self.assertFalse(mmarco_selected.valid_pair("  ", " хороший текст ", cfg))
        self.assertFalse(mmarco_selected.valid_pair("ab", " хороший текст ", cfg))
        self.assertFalse(mmarco_selected.valid_pair("слишком длинный запрос", " хороший текст ", cfg))
        self.assertFalse(mmarco_selected.valid_pair("запрос", "abc", cfg))
        self.assertFalse(mmarco_selected.valid_pair("запрос", "x" * 25, cfg))

    def test_dataset_card_declares_expected_hf_splits(self):
        card = mmarco_selected.render_dataset_card(
            repo_id="justatom/mmarco-ru-selected",
            cfg=mmarco_selected.SelectionConfig(),
        )

        self.assertIn("split: train", card)
        self.assertIn("path: data/train-*.parquet", card)
        self.assertIn("split: dev", card)
        self.assertIn("path: data/dev-*.parquet", card)
        self.assertIn("split: corpus", card)
        self.assertIn("path: data/corpus-*.parquet", card)

    def test_write_manifest_records_selection_contract(self):
        cfg = mmarco_selected.SelectionConfig(train_rows=10, dev_queries=3, eval_corpus_docs=7, seed=123)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = mmarco_selected.write_manifest(
                output_dir=Path(tmpdir),
                repo_id="justatom/mmarco-ru-selected",
                cfg=cfg,
                counts={"train": 10, "dev": 3, "corpus": 7},
            )

            manifest = json.loads(path.read_text())

        self.assertEqual(manifest["repo_id"], "justatom/mmarco-ru-selected")
        self.assertEqual(manifest["source"], "ir_datasets:mmarco/v2/ru")
        self.assertEqual(manifest["selection"]["seed"], 123)
        self.assertEqual(manifest["selection"]["train_rows"], 10)
        self.assertEqual(manifest["selection"]["dev_queries"], 3)
        self.assertEqual(manifest["selection"]["eval_corpus_docs"], 7)
        self.assertEqual(manifest["counts"], {"train": 10, "dev": 3, "corpus": 7})

    def test_resolve_hf_token_supports_hf_api_key(self):
        token = mmarco_selected.resolve_hf_token({"HF_API_KEY": " hf_dummy_token "})

        self.assertEqual(token, "hf_dummy_token")

    def test_hf_rows_are_aligned_to_one_schema_for_all_splits(self):
        train = mmarco_selected.align_rows_for_hf(
            [
                {
                    "pair_id": "q1:d1",
                    "query_id": "q1",
                    "positive_doc_id": "d1",
                    "negative_doc_id": "d2",
                    "query": "query",
                    "positive": "positive",
                    "source": "train",
                    "bucket": "qrels",
                }
            ]
        )
        dev = mmarco_selected.align_rows_for_hf(
            [
                {
                    "pair_id": "q2:d3",
                    "query_id": "q2",
                    "positive_doc_id": "d3",
                    "query": "query",
                    "positive": "positive",
                    "source": "dev",
                    "bucket": "qrels",
                }
            ]
        )
        corpus = mmarco_selected.align_rows_for_hf(
            [
                {
                    "doc_id": "d4",
                    "content": "content",
                    "source": "random",
                }
            ]
        )

        self.assertEqual(list(train[0]), mmarco_selected.HF_COLUMNS)
        self.assertEqual(list(dev[0]), mmarco_selected.HF_COLUMNS)
        self.assertEqual(list(corpus[0]), mmarco_selected.HF_COLUMNS)
        self.assertEqual(train[0]["pair_id"], "q1:d1")
        self.assertEqual(dev[0]["pair_id"], "q2:d3")
        self.assertEqual(corpus[0]["pair_id"], "")
        self.assertEqual(dev[0]["negative_doc_id"], "")
        self.assertEqual(corpus[0]["query"], "")
        self.assertEqual(corpus[0]["content"], "content")


if __name__ == "__main__":
    unittest.main()
