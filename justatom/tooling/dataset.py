import polars as pl

from justatom.storing.dataset import API as DatasetApi


def source_from_dataset(dataset_name_or_path, **props):
    maybe_df_or_iter = DatasetApi.named(dataset_name_or_path).iterator(**props)
    if isinstance(maybe_df_or_iter, pl.DataFrame):
        pl_data = maybe_df_or_iter
    else:
        dataset = list(maybe_df_or_iter)
        pl_data = pl.from_dicts(dataset)
    return pl_data
