import unittest
import tempfile
from pathlib import Path
import json
import polars as pl

from justatom.etc.schema import Document
from justatom.tooling.dataset import DatasetRecordAdapter
from justatom.tooling.profiler import MemoryProfiler


class EvalDataNormalizationTest(unittest.TestCase):
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

    def test_normalize_queries_handles_str_list_and_none(self):
        self.assertEqual(
            DatasetRecordAdapter.normalize_queries("one question"), ["one question"]
        )
        self.assertEqual(
            DatasetRecordAdapter.normalize_queries(["q1", "q2", None, ""]), ["q1", "q2"]
        )
        self.assertEqual(DatasetRecordAdapter.normalize_queries(None), [])
        self.assertEqual(DatasetRecordAdapter.normalize_queries([]), [])

    def test_normalize_queries_handles_json_string(self):
        self.assertEqual(
            DatasetRecordAdapter.normalize_queries('["q1", "q2"]'), ["q1", "q2"]
        )

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
