import torch
from torch.utils.data import TensorDataset

from justatom.processing import igniset
from justatom.processing.loader import NamedDataLoader


class DummyStreamingProcessor:
    def dataset_from_dicts(self, dicts, indices=None, return_baskets=False):
        size = len(dicts)
        input_ids = torch.tensor([[len(str(row.get("content", "")))] for row in dicts])
        attention_mask = torch.ones((size, 1), dtype=torch.long)
        labels = torch.ones((size, 1), dtype=torch.long)
        dataset = TensorDataset(input_ids, attention_mask, labels)
        tensor_names = ["input_ids", "attention_mask", "labels"]
        problematic = set()
        if return_baskets:
            return dataset, tensor_names, problematic, []
        return dataset, tensor_names, problematic


def test_named_dataloader_supports_streaming_dataset_batches():
    rows = [
        {"content": "alpha"},
        {"content": "beta"},
        {"content": "gamma"},
        {"content": "delta"},
        {"content": "epsilon"},
    ]

    dataset, tensor_names = igniset(
        rows,
        processor=DummyStreamingProcessor(),
        batch_size=2,
        streaming=True,
    )
    assert tensor_names is None

    loader = NamedDataLoader(
        dataset=dataset,
        tensor_names=tensor_names,
        batch_size=2,
    )

    batches = list(loader)

    assert len(loader) == 3
    assert len(batches) == 3
    assert batches[0]["input_ids"].shape == (2, 1)
    assert batches[1]["input_ids"].shape == (2, 1)
    assert batches[2]["input_ids"].shape == (1, 1)
    assert dataset.tensor_names == ["input_ids", "attention_mask", "labels"]
