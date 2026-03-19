import hashlib
import random
from collections.abc import Iterable

import simplejson as json
from loguru import logger
from torch.utils.data import ConcatDataset, IterableDataset

from justatom.processing.mask import IProcessor


def get_dict_checksum(payload_dict):
    """
    Get MD5 checksum for a dict.
    """
    checksum = hashlib.md5(
        json.dumps(payload_dict, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return checksum


class _StreamingDataSet(IterableDataset):
    """Stream processor outputs by chunking raw dicts and tensorizing on the fly."""

    def __init__(
        self,
        dicts: Iterable[dict],
        processor: IProcessor,
        batch_size: int = 1,
        shuffle: bool = False,
    ):
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.dicts = dicts
        self.processor = processor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tensor_names: list[str] | None = None
        self.problematic_sample_ids: set = set()
        self._num_samples = len(dicts) if hasattr(dicts, "__len__") else None

    def __len__(self):
        if self._num_samples is None:
            raise TypeError(
                "Streaming dataset length is unknown for this iterable source"
            )
        return self._num_samples

    def _iter_source(self):
        if self.shuffle:
            if isinstance(self.dicts, list):
                return iter(random.sample(self.dicts, len(self.dicts)))
            logger.warning(
                "shuffle=True is ignored for streaming datasets without random access."
            )
        return iter(self.dicts)

    def _iter_worker_shard(self):
        from torch.utils.data import get_worker_info

        worker = get_worker_info()
        source_iter = self._iter_source()
        if worker is None:
            yield from source_iter
            return

        for idx, row in enumerate(source_iter):
            if idx % worker.num_workers == worker.id:
                yield row

    def _emit_processed_batch(self, processing_batch: list[dict]):
        dataset, tensor_names, problematic_sample_ids = (
            self.processor.dataset_from_dicts(
                dicts=processing_batch,
                indices=list(range(len(processing_batch))),
            )
        )
        if tensor_names is not None:
            self.tensor_names = tensor_names
        self.problematic_sample_ids.update(problematic_sample_ids)
        for sample in dataset:
            yield sample

    def __iter__(self):
        processing_batch: list[dict] = []
        for row in self._iter_worker_shard():
            processing_batch.append(row)
            if len(processing_batch) < self.batch_size:
                continue
            yield from self._emit_processed_batch(processing_batch)
            processing_batch = []

        if processing_batch:
            yield from self._emit_processed_batch(processing_batch)


def igniset(
    dicts: list[dict] | Iterable[dict],
    processor: IProcessor,
    batch_size: int = 1,
    shuffle: bool = False,
    streaming: bool = False,
):
    from tqdm.autonotebook import tqdm

    if streaming:
        dataset = _StreamingDataSet(
            dicts=dicts,
            processor=processor,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        return dataset, None

    if not isinstance(dicts, list):
        dicts = list(dicts)

    datasets = []
    num_dicts = len(dicts)
    problems = set()
    dicts = random.sample(dicts, len(dicts)) if shuffle else dicts
    for i in tqdm(
        range(0, num_dicts, batch_size), desc="Preprocessing dataset", unit=" Dicts"
    ):
        processing_batch = dicts[i : i + batch_size]
        dataset, tensor_names, problematic_sample_ids = processor.dataset_from_dicts(
            dicts=processing_batch,
            indices=list(range(len(processing_batch))),  # TODO remove indices
        )
        datasets.append(dataset)
        problems.update(problematic_sample_ids)
    # TODO: Log problematic
    dataset = ConcatDataset(datasets=datasets)
    return dataset, tensor_names


__all__ = ["igniset"]
