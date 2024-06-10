from justatom.storing.mask import IDFDocStore
import polars as pl
import numpy as np
from typing import Union
import re


class POLARStore(IDFDocStore):

    def __init__(self, df: Union[pl.DataFrame, pl.Series]):
        self.df = df

    def counts_per_col(self, col: str, view_as_dict: bool = False):
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

    def count_words_per_col(self, col: str, sort=True, stopchars: str = None):

        df = self.df

        def clear_doc(text):
            return re.sub(regex_pattern, "", text)

        if stopchars is not None:
            regex_pattern = f"[{re.escape(stopchars)}]"
            cur_df = df[col].apply(clear_doc)
        else:
            cur_df = df[col]

        cur_df = df.drop([col]).with_columns(cur_df.alias(col))

        pl_counts_view = (
            cur_df.select(pl.col(col).str.split(" ").flatten().alias("words"))
            .to_series()
            .value_counts()
            .filter(pl.col("words").str.lengths() > 0)
            .sort("counts", descending=True)
        )

        return pl_counts_view

    def parse_metrics_per_col(self, col):
        pass

    def samples_per_col(self, col, n_samples):
        pass

    def select_only(self, over_col: str, value: str, with_row: bool = False):
        view = self.df.filter(pl.col(over_col) == value)
        if with_row:
            return view
        return view.drop(["row_nr"])

    def prepare_2d(self):
        raise NotImplementedError()

    def merge_and_replace(self, pl_other: pl.DataFrame, using_col: str, how="outer"):
        # NOTE: https://stackoverflow.com/questions/73427091/polars-replace-part-of-string-in-column-with-value-of-other-column
        # TODO: Add naming and double-check the polar(s) performance using this version
        pl_result = (
            self.df.join(pl_other, on=using_col, how=how)
            .with_row_count()
            .with_columns([pl.col("description").str.split_exact("TODO", 1)])
            .unnest("description")
            .with_columns(
                [
                    pl.when(pl.col("field_1").is_null())
                    .then(pl.col("field_0"))
                    .otherwise(pl.concat_str(["field_1", "description_right"]))
                    .alias("response_connected")
                ]
            )
            .drop(["row_nr", "field_0", "field_1", "description_right"])
            .rename({"response_connected": "description"})
        )
        return pl_result

    def _sample_it(s: pl.Series) -> pl.Series:
        return pl.Series(
            values=np.random.binomial(1, 0.7, s.len()),
            dtype=pl.Boolean,
        )

    def random_sample(self, method: str = "binomial", sample_size: int = 100, num_obs: int = 1000):
        """
        We want to sample the given population size (len(self.df)) taking `num_obs` observations with each
        observaion being `sample_size` size.
        """
        # [self.df.sample(sample_size) for _ in range(num_obs)]
        # Lazy API. See https://stackoverflow.com/a/76359078/22622408 for more info
        pl_lazy_obs = [
            self.df.lazy().select(row=pl.struct(pl.all()).sample(sample_size)).unnest("row") for _ in range(num_obs)
        ]

        return pl.collect_all(pl_lazy_obs)


__all__ = ["POLARStore"]
