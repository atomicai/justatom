import copy
import itertools
import os
import random
import uuid
from collections.abc import MutableMapping
from copy import deepcopy

import numpy as np
import torch


class alist(list):
    async def __aiter__(self):
        for _ in self:
            yield _


class AsyncConstructor(object):
    async def __new__(cls, *a, **kw):
        instance = super().__new__(cls)
        await instance.__init__(*a, **kw)
        return instance


class NIterator:
    __slots__ = ("_is_next", "_the_next", "it")

    def __init__(self, it):
        self.it = iter(it)
        self._is_next = None
        self._the_next = None

    def has_next(self) -> bool:
        if self._is_next is None:
            try:
                self._the_next = next(self.it)
            except:  # noqa: E722
                self._is_next = False
            else:
                self._is_next = True
        return self._is_next

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._is_next:  # noqa: SIM108
            response = self._the_next
        else:
            response = next(self.it)
        self._is_next = None
        return response


def chunkify(f, chunksize=10_000_000, sep="\n"):
    """
    Read a file separating its content lazily.

    Usage:

    >>> with open('INPUT.TXT') as f:
    >>>     for item in chunkify(f):
    >>>         process(item)
    """
    chunk = None
    remainder = None  # data from the previous chunk.
    while chunk != "":
        chunk = f.read(chunksize)
        if remainder:  # noqa: SIM108
            piece = remainder + chunk
        else:
            piece = chunk
        pos = None
        while pos is None or pos >= 0:
            pos = piece.find(sep)
            if pos >= 0:
                if pos > 0:
                    yield piece[:pos]
                piece = piece[pos + 1 :]
                remainder = None
            else:
                remainder = piece
    if remainder:  # This statement will be executed iff @remainder != ''
        yield remainder


def merge_iterators(*iterators):
    return itertools.chain(*iterators)


def flatten_dict(dictionary, parent_key="", separator="_"):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def snapshot(data: dict | list[dict], sep: str = "_"):
    if isinstance(data, dict):
        data = [data]
    response = []
    for cur in data:
        cur_sample = flatten_dict(cur, separator=sep)
        for k, v in zip(cur_sample.keys(), cur_sample.values(), strict=False):
            response.append(str(k) + "=" + str(v))
    return sep.join(response)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def training_params(x: torch.nn.Module):
    return sum(p.numel() for p in x.parameters() if p.requires_grad)


def reuuid(text, namespace_uuid="91461c99-f89d-49d2-af96-d8e2e14e9b58"):
    """
    Генерирует детерминированный UUID в стиле RethinkDB на основе входного текста и заданного namespace.

    Args:
    text (str): Текст, на основе которого нужно сгенерировать UUID.
    namespace_uuid (str): Строка UUID для namespace, используемого RethinkDB для генерации детерминированных UUID.

    Returns:
    str: Строковое представление детерминированного UUID.
    """
    # Преобразование строки namespace в объект UUID
    namespace = uuid.UUID(namespace_uuid)

    # Генерация детерминированного UUID версии 5
    deterministic_uuid = uuid.uuid5(namespace, text)

    return str(deterministic_uuid)


def merge_in_order(a: dict | None = None, dv: dict | None = None, do_copy: bool = False):
    """
    This function perfrorm `merge` in a Asymmetric way.
    Standard `a.update(dv)` doesn't change with argument swapping.
    Hence we need to wrap them around.

    Args:
        a (Optional[Dict], optional): _description_. Defaults to None.
        dv (Optional[Dict], optional): _description_. Defaults to None.
        do_copy (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: Dict
        _description_: Positional merge of two different dict(s) as described.
    """
    # {**{ key: value }, **default_values} On average it is faster!
    a = a or dict()
    dv = dv or dict()
    if do_copy:  # noqa: SIM108
        response = copy.deepcopy(a)
    else:
        response = a
    response = {**dv, **response}
    return response


def flatten_list(nested_list):
    """Flatten an arbitrarily nested list, without recursion (to avoid
    stack overflows). Returns a new list, the original list is unchanged.
    >> list(flatten_list([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]]))
    [1, 2, 3, 4, 5]
    >> list(flatten_list([[1, 2], 3]))
    [1, 2, 3]
    """
    nested_list = deepcopy(nested_list)

    while nested_list:
        sublist = nested_list.pop(0)

        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            yield sublist


__all__ = [
    "NIterator",
    "flatten_dict",
    "snapshot",
    "flatten_list",
    "merge_in_order",
    "training_params",
    "seed_everything",
    "reuuid",
]
