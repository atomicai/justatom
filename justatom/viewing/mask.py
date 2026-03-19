import abc
from typing import Any

import polars as pl


class IChart(abc.ABC):
    chart: Any

    @abc.abstractmethod
    def view(self, data: pl.DataFrame, **props):
        pass

    def save(self, filename, ppi=200):
        self.chart.save(filename, ppi=ppi)
