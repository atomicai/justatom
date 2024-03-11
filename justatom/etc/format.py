from datetime import datetime
from pathlib import Path
from itertools import islice
import simplejson as json


def convert_date_to_rfc3339(date: str) -> str:
    """
    Converts a date to RFC3339 format, as Weaviate requires dates to be in RFC3339 format including the time and
    timezone.

    If the provided date string does not contain a time and/or timezone, we use 00:00 as default time
    and UTC as default time zone.

    This method cannot be part of WeaviateDocumentStore, as this would result in a circular import between weaviate.py
    and filter_utils.py.
    """
    parsed_datetime = datetime.fromisoformat(date)
    if parsed_datetime.utcoffset() is None:
        converted_date = parsed_datetime.isoformat() + "Z"
    else:
        converted_date = parsed_datetime.isoformat()

    return converted_date


def get_batches_from_generator(iterable, n):
    """
    Batch elements of an iterable into fixed-length chunks or blocks.
    """
    it = iter(iterable)
    x = tuple(islice(it, n))
    while x:
        yield x
        x = tuple(islice(it, n))


def grouper(iterable, n, worker_id=0, total_workers=1):
    """
    Split an iterable into a list of n-sized chunks. Each element in the chunk is a tuple of (index_num, element).

    Example:

    >>> list(grouper('ABCDEFG', 3))
    [[(0, 'A'), (1, 'B'), (2, 'C')], [(3, 'D'), (4, 'E'), (5, 'F')], [(6, 'G')]]



    Use with the StreamingDataSilo

    When StreamingDataSilo is used with multiple PyTorch DataLoader workers, the generator
    yielding dicts(that gets converted to datasets) is replicated across the workers.

    To avoid duplicates, we split the dicts across workers by creating a new generator for
    each worker using this method.

    Input --> [dictA, dictB, dictC, dictD, dictE, ...] with total worker=3 and n=2

    Output for worker 1: [(dictA, dictB), (dictG, dictH), ...]
    Output for worker 2: [(dictC, dictD), (dictI, dictJ), ...]
    Output for worker 3: [(dictE, dictF), (dictK, dictL), ...]

    This method also adds an index number to every dict yielded.

    :param iterable: a generator object that yields dicts
    :type iterable: generator
    :param n: the dicts are grouped in n-sized chunks that gets converted to datasets
    :type n: int
    :param worker_id: the worker_id for the PyTorch DataLoader
    :type worker_id: int
    :param total_workers: total number of workers for the PyTorch DataLoader
    :type total_workers: int
    """

    # TODO make me comprehensible :)
    def get_iter_start_pos(gen):
        start_pos = worker_id * n
        for i in gen:
            if start_pos:
                start_pos -= 1
                continue
            yield i

    def filter_elements_per_worker(gen):
        x = n
        y = (total_workers - 1) * n
        for i in gen:
            if x:
                yield i
                x -= 1
            else:
                if y != 1:
                    y -= 1
                    continue
                else:
                    x = n
                    y = (total_workers - 1) * n

    iterable = iter(enumerate(iterable))
    iterable = get_iter_start_pos(iterable)
    if total_workers > 1:
        iterable = filter_elements_per_worker(iterable)

    return iter(lambda: list(islice(iterable, n)), [])


def is_json(x):
    if issubclass(type(x), Path):
        return True
    try:
        json.dumps(x)
        return True
    except:
        return False


__all__ = ["is_json", "convert_date_to_rfc3339", "get_batches_from_generator", "grouper"]
