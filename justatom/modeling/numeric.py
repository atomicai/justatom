from justatom.etc.pattern import singleton
from typing import Union, List, Generator
import scipy as sc


@singleton
class NUMGenerator:

    SHAPES = "sphere"

    def __init__(self):
        pass

    def gamma(self, x: Union[float, List[float]]) -> Generator:
        if isinstance(x, (float, int)):
            yield sc.special.gamma(x)
        else:
            for xi in x:
                yield sc.special.gamma(xi)

    def beta(self, x: Union[float, List[float]], y: Union[float, List[float]]) -> Generator:
        assert type(x) is type(y), f"Data types do not match between {type(x)} != {type(y)}."
        if isinstance(x, (int, float)):
            yield sc.special.beta(x, y)
        elif isinstance(x, list):
            for xi, yi in zip(x, y):
                yield sc.special.beta(xi, yi)
        else:
            raise ValueError(f"Data types match {type(x)} == {type(y)} but not allowed. Use <float> or <List[float]>")

    def volume(self, eps: float, r: float, shape: str = "sphere") -> Generator:
        assert shape in self.SHAPES, f"Provided shape={shape} is not one of {','.join(self.SHAPES)}"
        # TODO:
        return 1


INUMGenerator = NUMGenerator()


__all__ = ["INUMGenerator"]
