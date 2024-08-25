import hashlib
import random

import simplejson as json

from justatom.processing.mask import IProcessor


def get_dict_checksum(payload_dict):
    """
    Get MD5 checksum for a dict.
    """
    checksum = hashlib.md5(json.dumps(payload_dict, sort_keys=True).encode("utf-8")).hexdigest()
    return checksum


def igniset(dicts: list[dict], processor: IProcessor, batch_size: int = 1, shuffle: bool = False):
    from torch.utils.data import ConcatDataset
    from tqdm.autonotebook import tqdm

    datasets = []
    num_dicts = len(dicts)
    problems = set()
    dicts = random.sample(dicts, len(dicts)) if shuffle else dicts
    for i in tqdm(range(0, num_dicts, batch_size), desc="Preprocessing dataset", unit=" Dicts"):
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
