from collections.abc import Generator

from scipy.special import gamma as gamma_fn

from justatom.etc.pattern import singleton


@singleton
class NUMGenerator:
    SHAPES = ["sphere"]

    def __init__(self):
        pass

    def gamma(self, x: float | list[float]) -> Generator:
        if isinstance(x, (float, int)):  # noqa: UP038
            yield gamma_fn(x)
        else:
            for xi in x:
                yield gamma_fn(xi)

    def beta(self, x: float | list[float], y: float | list[float]) -> Generator:
        assert type(x) is type(
            y
        ), f"Data types do not match between {type(x)} != {type(y)}."
        if isinstance(x, (int, float)):  # noqa: UP038
            yield gamma_fn(x) * gamma_fn(y) / gamma_fn(x + y)
        elif isinstance(x, list):
            for xi, yi in zip(x, y, strict=False):
                yield gamma_fn(xi) * gamma_fn(yi) / gamma_fn(xi + yi)
        else:
            raise ValueError(
                f"Data types match {type(x)} == {type(y)} but not allowed. Use <float> or <List[float]>"
            )

    def volume(self, eps: float, r: float, shape: str = "sphere") -> Generator:
        assert (
            shape in self.SHAPES
        ), f"Provided shape={shape} is not one of {','.join(self.SHAPES)}"
        # TODO:
        return 1


INUMGenerator = NUMGenerator()


__all__ = ["INUMGenerator"]
