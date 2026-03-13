import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from justatom.etc.schema import Document
from justatom.tooling.dataset import DatasetRecordAdapter
from justatom.tooling.profiler import MemoryProfiler


class EvalDataNormalizationTest(unittest.TestCase):
    def test_from_source_supports_named_justatom_dataset(self):
        adapter = DatasetRecordAdapter.from_source(
            dataset_name_or_path="justatom",
            content_col="content",
            queries_col="queries",
            chunk_id_col="chunk_id",
            keywords_col="keywords_or_phrases",
            keywords_nested_col="keyword_or_phrase",
            explanation_nested_col="explanation",
            lazy=True,
        )

        first = next(adapter.iterator())
        self.assertTrue(first["id"])
        self.assertTrue(first["content"])
        self.assertTrue(first["meta"]["labels"])
        self.assertIn("title", first["meta"])
        self.assertIn("keywords_or_phrases", first["meta"])

    def test_from_source_supports_hf_uri(self):
        rows = [
            {
                "text": "Органические остатки представлены известковыми выделениями.",
                "q": "чем представлены органические остатки?",
                "a": "известковыми выделениями",
                "context": "long supporting context",
            }
        ]

        with patch("justatom.storing.dataset.load_dataset", return_value=rows) as mocked:
            adapter = DatasetRecordAdapter.from_source(
                dataset_name_or_path="hf://MLNavigator/russian-retrieval?split=train",
                content_col="text",
                queries_col="q",
                lazy=True,
            )

        first = next(adapter.iterator())
        mocked.assert_called_once_with(
            "MLNavigator/russian-retrieval",
            name=None,
            split="train",
            streaming=True,
        )
        self.assertEqual(
            first["content"],
            "Органические остатки представлены известковыми выделениями.",
        )
        self.assertEqual(first["meta"]["labels"], ["чем представлены органические остатки?"])
        self.assertEqual(first["meta"]["a"], "известковыми выделениями")
        self.assertEqual(first["meta"]["context"], "long supporting context")

    def test_from_source_supports_hf_split_fallback_candidates(self):
        rows = [{"text": "text-a", "q": "query-a"}]

        def _fake_load_dataset(dataset_name, name=None, split=None, streaming=None, **kwargs):
            del name, streaming, kwargs
            if dataset_name != "MLNavigator/russian-retrieval":
                raise AssertionError(dataset_name)
            if split == "dev":
                raise ValueError("Unknown split: dev")
            if split == "test":
                return rows
            raise AssertionError(split)

        with patch("justatom.storing.dataset.load_dataset", side_effect=_fake_load_dataset) as mocked:
            adapter = DatasetRecordAdapter.from_source(
                dataset_name_or_path="hf://MLNavigator/russian-retrieval",
                content_col="text",
                queries_col="q",
                split="dev|test",
                lazy=True,
            )

        first = next(adapter.iterator())
        self.assertEqual(first["content"], "text-a")
        self.assertEqual(first["meta"]["labels"], ["query-a"])
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(mocked.call_args_list[0].kwargs["split"], "dev")
        self.assertEqual(mocked.call_args_list[1].kwargs["split"], "test")

    def test_from_source_reports_helpful_error_for_single_missing_hf_split(self):
        with patch(
            "justatom.storing.dataset.load_dataset",
            side_effect=ValueError('Unknown split "test". Should be one of ["train"].'),
        ):
            with self.assertRaisesRegex(ValueError, r"test\|train|train\|test"):
                DatasetRecordAdapter.from_source(
                    dataset_name_or_path="hf://MLNavigator/russian-retrieval",
                    content_col="text",
                    queries_col="q",
                    split="test",
                    lazy=True,
                )

    def test_from_source_supports_builtin_jsonl_uri(self):
        adapter = DatasetRecordAdapter.from_source(
            dataset_name_or_path="builtin://datasets/demo_retrieval.jsonl",
            content_col="content",
            queries_col="labels",
            chunk_id_col="chunk_id",
            lazy=True,
        )

        first = next(adapter.iterator())
        self.assertEqual(first["id"], "doc-1")
        self.assertEqual(first["content"], "Cats sleep up to sixteen hours a day.")
        self.assertEqual(first["meta"]["labels"], ["how long do cats sleep", "cat sleeping hours"])
        self.assertEqual(first["meta"]["instruction"], "builtin-demo")

    def test_from_source_respects_limit_for_builtin_jsonl(self):
        adapter = DatasetRecordAdapter.from_source(
            dataset_name_or_path="builtin://datasets/demo_retrieval.jsonl",
            content_col="content",
            queries_col="labels",
            chunk_id_col="chunk_id",
            lazy=True,
            limit=1,
        )

        docs = list(adapter.iterator())
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["id"], "doc-1")

    def test_iterator_with_profiler_tracks_processed_items(self):
        rows = [
            {"chunk_id": "a", "content": "p1", "queries": ["q1"]},
            {"chunk_id": "b", "content": "p2", "queries": []},
            {"chunk_id": "c", "content": "p3", "queries": None},
        ]
        profiler = MemoryProfiler(enabled=True)

        adapter = DatasetRecordAdapter(
            records=rows,
            content_col="content",
            queries_col="queries",
            chunk_id_col="chunk_id",
        )
        docs = list(adapter.iterator(profiler=profiler))

        self.assertEqual(len(docs), 3)
        self.assertTrue(all(isinstance(doc, dict) for doc in docs))
        report = profiler.report()
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report.name, "DatasetRecordAdapter.iterator")
        self.assertEqual(report.items_processed, 3)

    def test_dataset_with_profiler_tracks_items_and_supports_json_or_documents(self):
        rows = [
            {"chunk_id": "a", "content": "p1", "queries": ["q1"]},
            {"chunk_id": "b", "content": "p2", "queries": []},
        ]
        profiler = MemoryProfiler(enabled=True)

        adapter = DatasetRecordAdapter(
            records=rows,
            content_col="content",
            queries_col="queries",
            chunk_id_col="chunk_id",
        )
        docs_json = adapter.dataset(profiler=profiler)

        self.assertEqual(len(docs_json), 2)
        self.assertTrue(all(isinstance(doc, dict) for doc in docs_json))

        docs_objects = adapter.dataset(as_json=False)
        self.assertEqual(len(docs_objects), 2)
        self.assertTrue(all(isinstance(doc, Document) for doc in docs_objects))

        report = profiler.report()
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report.name, "DatasetRecordAdapter.dataset")
        self.assertEqual(report.items_processed, 2)

    def test_from_source_respects_lazy_materialization_contract(self):
        fd, path = tempfile.mkstemp(suffix=".json", prefix="adapter_source_")
        Path(path).unlink(missing_ok=True)
        data_path = Path(path)
        data_path.write_text(
            json.dumps(
                [
                    {"chunk_id": "a", "content": "p1", "queries": ["q1"]},
                    {"chunk_id": "b", "content": "p2", "queries": ["q2"]},
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        try:
            eager_adapter = DatasetRecordAdapter.from_source(
                dataset_name_or_path=data_path,
                content_col="content",
                queries_col="queries",
                chunk_id_col="chunk_id",
                lazy=False,
            )
            self.assertIsInstance(eager_adapter.records, list)
            self.assertEqual(eager_adapter.records[0]["chunk_id"], "a")

            lazy_adapter = DatasetRecordAdapter.from_source(
                dataset_name_or_path=data_path,
                content_col="content",
                queries_col="queries",
                chunk_id_col="chunk_id",
                lazy=True,
            )
            self.assertFalse(isinstance(lazy_adapter.records, list))
            first = next(iter(lazy_adapter.records))
            self.assertEqual(first["chunk_id"], "a")
        finally:
            data_path.unlink(missing_ok=True)

    def test_from_source_supports_parquet_with_custom_columns(self):
        fd, path = tempfile.mkstemp(suffix=".parquet", prefix="adapter_source_")
        Path(path).unlink(missing_ok=True)
        data_path = Path(path)

        pl.DataFrame(
            [
                {
                    "instruction": "instr-a",
                    "input": "question-a",
                    "output": "answer-a",
                },
                {
                    "instruction": "instr-b",
                    "input": "question-b",
                    "output": "answer-b",
                },
            ]
        ).write_parquet(data_path)

        try:
            adapter = DatasetRecordAdapter.from_source(
                dataset_name_or_path=data_path,
                content_col="output",
                queries_col="input",
                lazy=True,
            )

            first = next(adapter.iterator())
            self.assertEqual(first["content"], "answer-a")
            self.assertEqual(first["meta"]["labels"], ["question-a"])
            self.assertNotIn("instruction", first)
            self.assertEqual(first["meta"]["instruction"], "instr-a")
        finally:
            data_path.unlink(missing_ok=True)

    def test_from_source_lazy_json_warns_and_falls_back_to_eager_load(self):
        fd, path = tempfile.mkstemp(suffix=".json", prefix="adapter_streaming_")
        Path(path).unlink(missing_ok=True)
        data_path = Path(path)
        rows = [
            {"chunk_id": "a", "content": "p1", "queries": ["q1"]},
            {"chunk_id": "b", "content": "p2", "queries": ["q2"]},
        ]
        data_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

        try:
            with patch("justatom.storing.dataset.logger.warning") as warning_mock:
                adapter = DatasetRecordAdapter.from_source(
                    dataset_name_or_path=data_path,
                    content_col="content",
                    queries_col="queries",
                    chunk_id_col="chunk_id",
                    lazy=True,
                )

                first = next(iter(adapter.records))
                self.assertEqual(first["chunk_id"], "a")
                warning_mock.assert_called_once()
                self.assertIn("lazy=True for .json is unsupported", warning_mock.call_args[0][0])
        finally:
            data_path.unlink(missing_ok=True)

    def test_from_source_lazy_json_wrapped_payload_warns_and_still_loads(self):
        fd, path = tempfile.mkstemp(suffix=".json", prefix="adapter_wrapped_")
        Path(path).unlink(missing_ok=True)
        data_path = Path(path)
        data_path.write_text(
            json.dumps(
                {
                    "data": [
                        {"chunk_id": "a", "content": "p1", "queries": ["q1"]},
                        {"chunk_id": "b", "content": "p2", "queries": ["q2"]},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        try:
            with patch("justatom.storing.dataset.logger.warning") as warning_mock:
                adapter = DatasetRecordAdapter.from_source(
                    dataset_name_or_path=data_path,
                    content_col="content",
                    queries_col="queries",
                    chunk_id_col="chunk_id",
                    lazy=True,
                )

                first = next(iter(adapter.records))
                self.assertEqual(first["chunk_id"], "a")
                warning_mock.assert_called_once()
        finally:
            data_path.unlink(missing_ok=True)

    def test_normalize_queries_handles_str_list_and_none(self):
        self.assertEqual(DatasetRecordAdapter.normalize_queries("one question"), ["one question"])
        self.assertEqual(DatasetRecordAdapter.normalize_queries(["q1", "q2", None, ""]), ["q1", "q2"])
        self.assertEqual(DatasetRecordAdapter.normalize_queries(None), [])
        self.assertEqual(DatasetRecordAdapter.normalize_queries([]), [])

    def test_normalize_queries_handles_json_string(self):
        self.assertEqual(DatasetRecordAdapter.normalize_queries('["q1", "q2"]'), ["q1", "q2"])

    def test_normalize_keywords_handles_all_supported_shapes(self):
        self.assertEqual(
            DatasetRecordAdapter.normalize_keywords(
                "single keyword",
                keywords_nested_col="keyword_or_phrase",
                explanation_nested_col="explanation",
            ),
            [{"keyword_or_phrase": "single keyword"}],
        )

        self.assertEqual(
            DatasetRecordAdapter.normalize_keywords(
                [
                    {
                        "keywords_or_phrase": "kw-a",
                        "explanation": "exp-a",
                    },
                    {"keyword_or_phrase": "kw-b"},
                ],
                keywords_nested_col="keyword_or_phrase",
                explanation_nested_col="explanation",
            ),
            [
                {"keyword_or_phrase": "kw-a", "explanation": "exp-a"},
                {"keyword_or_phrase": "kw-b"},
            ],
        )

        self.assertEqual(
            DatasetRecordAdapter.normalize_keywords(
                None,
                keywords_nested_col="keyword_or_phrase",
                explanation_nested_col="explanation",
            ),
            [],
        )
        self.assertEqual(
            DatasetRecordAdapter.normalize_keywords(
                [],
                keywords_nested_col="keyword_or_phrase",
                explanation_nested_col="explanation",
            ),
            [],
        )

    def test_wrapper_for_docs_keeps_rows_without_queries(self):
        rows = [
            {"chunk_id": "a", "content": "p1", "queries": None},
            {"chunk_id": "b", "content": "p2", "queries": []},
            {"chunk_id": "c", "content": "p3", "queries": "q3"},
        ]

        docs = list(
            DatasetRecordAdapter(
                records=rows,
                content_col="content",
                queries_col="queries",
                chunk_id_col="chunk_id",
            ).iterator()
        )

        self.assertEqual(len(docs), 3)
        self.assertEqual(docs[0]["meta"]["labels"], [])
        self.assertEqual(docs[1]["meta"]["labels"], [])
        self.assertEqual(docs[2]["meta"]["labels"], ["q3"])

    def test_wrapper_for_docs_raises_on_duplicate_chunk_id(self):
        rows = [
            {"chunk_id": "dup", "content": "p1", "queries": ["q1"]},
            {"chunk_id": "dup", "content": "p2", "queries": ["q2"]},
        ]

        with self.assertRaises(ValueError):
            list(
                DatasetRecordAdapter(
                    records=rows,
                    content_col="content",
                    queries_col="queries",
                    chunk_id_col="chunk_id",
                ).iterator()
            )

    def test_preserve_all_fields_default_true(self):
        rows = [
            {
                "chunk_id": "a",
                "content": "p1",
                "queries": ["q1"],
                "title": "book-a",
                "keywords_or_phrases": [{"keyword_or_phrase": "alpha"}],
            }
        ]

        doc = next(
            DatasetRecordAdapter(
                records=rows,
                content_col="content",
                queries_col="queries",
                chunk_id_col="chunk_id",
            ).iterator()
        )

        self.assertNotIn("title", doc)
        self.assertNotIn("chunk_id", doc)
        self.assertEqual(doc["meta"]["labels"], ["q1"])
        self.assertEqual(doc["meta"]["title"], "book-a")
        self.assertNotIn("chunk_id", doc["meta"])
        self.assertEqual(doc["meta"]["keywords_or_phrases"], [{"keyword_or_phrase": "alpha"}])

    def test_preserve_all_fields_can_be_disabled(self):
        rows = [
            {
                "chunk_id": "a",
                "content": "p1",
                "queries": ["q1"],
                "title": "book-a",
            }
        ]

        doc = next(
            DatasetRecordAdapter(
                records=rows,
                content_col="content",
                queries_col="queries",
                chunk_id_col="chunk_id",
                preserve_all_fields=False,
            ).iterator()
        )

        self.assertNotIn("title", doc)
        self.assertNotIn("title", doc["meta"])
        self.assertIn("id", doc)
        self.assertEqual(doc["meta"]["labels"], ["q1"])


if __name__ == "__main__":
    unittest.main()
