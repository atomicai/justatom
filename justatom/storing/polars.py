from justatom.storing.mask import IDFDocStore
import polars as pl
from typing import Union


class POLARStore(IDFDocStore):

    def __init__(self, df: Union[pl.DataFrame, pl.Series]):
        self.df = df

    def count_per_col(self, col: str, view_as_dict: bool = False):
        df = self.df.with_row_count()
        pl_view = (
            df.with_columns(
                [pl.count("row_nr").over(col).alias(f"counts_per_{col}"), pl.first("row_nr").over(col).alias("mask")]
            )
            .filter(pl.col("row_nr") == pl.col("mask"))
            .sort(by=f"counts_per_{col}", descending=True)
        )

        if view_as_dict:
            pl_res = {
                key_over: key_value
                for key_over, key_value in zip(
                    pl_view.select(col).to_series().to_list(), pl_view.select(f"counts_per_{col}").to_series().to_list()
                )
            }
        else:
            pl_res = pl_view.select([col, f"counts_per_{col}"])
        return pl_res

    def parse_metrics_per_col(self, col):
        pass

    def samples_per_col(self, col, n_samples):
        pass


__all__ = ["POLARStore"]
