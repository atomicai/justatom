import unittest

from justatom.tooling.dataset import DatasetRecordAdapter


class EvalDataNormalizationTest(unittest.TestCase):
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
            ).iter_documents()
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
                ).iter_documents()
            )


if __name__ == "__main__":
    unittest.main()
