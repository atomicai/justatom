from math import ceil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from justatom.etc.errors import ModelingError


class NamedDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        sampler: Sampler | None = None,
        tensor_names: list[str] | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """
        A modified version of the PyTorch DataLoader that returns a dictionary where the key is
        the name of the tensor and the value is the tensor itself.

        :param dataset: The dataset that will be wrapped by this NamedDataLoader
        :param sampler: The sampler used by the NamedDataLoader to choose which samples to include in the batch
        :param batch_size: The size of the batch to be returned by the NamedDataLoader
        :param tensor_names: The names of the tensor, in the order that the dataset returns them in.
        :param num_workers: number of workers to use for the DataLoader
        :param pin_memory: argument for Data Loader to use page-locked memory for faster transfer of data to GPU
        """

        def collate_fn(batch):
            """
            A custom collate function that formats the batch as a dictionary where the key is
            the name of the tensor and the value is the tensor itself
            """
            if type(dataset).__name__ == "_StreamingDataSet":  # noqa: SIM108
                _tensor_names = dataset.tensor_names
            else:
                _tensor_names = tensor_names

            if isinstance(batch[0], list):
                batch = batch[0]

            if len(batch[0]) != len(_tensor_names):
                raise ModelingError(
                    f"Dataset contains {len(batch[0])} tensors while there are {len(_tensor_names)} tensor names"
                    f" supplied: {_tensor_names}"
                )

            max_num_labels = self._compute_max_number_of_labels(batch=batch, tensor_names=_tensor_names)

            ret = {name: [] for name in tensor_names}
            for example in batch:
                for name, tensor in zip(_tensor_names, example, strict=False):
                    # each example may have a different number of answers/labels,
                    # so we need to pad the corresponding tensors to the max number of labels
                    if name == "labels" and tensor.ndim > 0:
                        num_labels = tensor.size(0)
                        if num_labels < max_num_labels:
                            padding = (0, 0, 0, max_num_labels - num_labels)
                            tensor = F.pad(tensor, padding, value=-1)
                    ret[name].append(tensor)

            for key in ret:
                ret[key] = torch.stack(ret[key])

            return ret

        super().__init__(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

    def __len__(self):
        if type(self.dataset).__name__ == "_StreamingDataSet":
            num_samples = len(self.dataset)
            num_batches = ceil(num_samples / self.dataset.batch_size)
            return num_batches
        else:
            return super().__len__()

    def _compute_max_number_of_labels(self, batch, tensor_names) -> int:
        """
        Compute the maximum number of labels in a batch.
        Each example may have a different number of labels, depending on the number of answers.
        """
        max_num_labels = 0
        for example in batch:
            for name, tensor in zip(tensor_names, example, strict=False):
                if name == "labels" and tensor.ndim > 0:
                    max_num_labels = max(max_num_labels, tensor.size(0))
        return max_num_labels


__all__ = ["NamedDataLoader"]
