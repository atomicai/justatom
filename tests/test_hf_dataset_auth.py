import os
import unittest
from unittest.mock import patch

import polars as pl

from justatom.storing import dataset as dataset_module


class HFDatasetAuthTests(unittest.TestCase):
    def test_hf_api_key_is_passed_to_load_dataset(self):
        calls = []

        def fake_load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return pl.DataFrame([{"id": 1}])

        with patch.dict(os.environ, {"HF_API_KEY": "hf_dummy"}, clear=True):
            with patch.object(dataset_module, "load_dataset", fake_load_dataset):
                source = dataset_module.HFDataset("hf://justatom/private-dataset")
                source.iterator(split="train")

        self.assertEqual(calls[0][1]["token"], "hf_dummy")

    def test_hf_api_key_is_passed_to_parquet_fallback(self, tmp_path=None):
        parquet_path = None
        if tmp_path is None:
            import tempfile
            from pathlib import Path

            tmpdir = tempfile.TemporaryDirectory()
            self.addCleanup(tmpdir.cleanup)
            parquet_path = Path(tmpdir.name) / "train-00000-of-00001.parquet"
        else:
            parquet_path = tmp_path / "train-00000-of-00001.parquet"
        pl.DataFrame([{"id": 1}]).write_parquet(parquet_path)

        list_calls = []
        download_calls = []

        def fake_load_dataset(*args, **kwargs):
            raise TypeError("builder failed")

        def fake_list_repo_files(*args, **kwargs):
            list_calls.append((args, kwargs))
            return ["data/train-00000-of-00001.parquet"]

        def fake_hf_hub_download(*args, **kwargs):
            download_calls.append((args, kwargs))
            return str(parquet_path)

        with patch.dict(os.environ, {"HF_API_KEY": "hf_dummy"}, clear=True):
            with patch.object(dataset_module, "load_dataset", fake_load_dataset):
                with patch.object(dataset_module, "list_repo_files", fake_list_repo_files):
                    with patch.object(dataset_module, "hf_hub_download", fake_hf_hub_download):
                        dataset_module.HFDataset._repo_files.cache_clear()
                        source = dataset_module.HFDataset("hf://justatom/private-dataset")
                        data = source.iterator(split="train")

        self.assertIsInstance(data, pl.DataFrame)
        self.assertEqual(list_calls[0][1]["token"], "hf_dummy")
        self.assertEqual(download_calls[0][1]["token"], "hf_dummy")


if __name__ == "__main__":
    unittest.main()
