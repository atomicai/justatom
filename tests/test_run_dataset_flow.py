from unittest.mock import patch

from justatom.api import dataset_input


def test_dataset_source_remains_lazy_until_indexer_consumes_it():
    rows = iter([{"content": "one"}, {"content": "two"}])

    with patch.object(dataset_input.DatasetLoader, "read", return_value=rows) as mocked:
        documents = dataset_input.documents_from_input("owner/data")

    assert documents is rows
    mocked.assert_called_once_with("owner/data", lazy=True)
