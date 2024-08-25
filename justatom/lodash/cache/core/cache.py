from collections import OrderedDict


class Cache(OrderedDict):
    def __init__(self, *args, size: int = 2077, mem_size: int | None = 10_000, **kwargs):
        assert size > 0
        self.size = size
        self.mem_size = mem_size
        assert mem_size >= 10_000, f"Memory size {mem_size} is not possible. Has to be >= {10_000}"
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().move_to_end(key)

        while len(self) > self.size:
            previous = next(iter(self))
            super().__delitem__(previous)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        super().move_to_end(key)

        return value
