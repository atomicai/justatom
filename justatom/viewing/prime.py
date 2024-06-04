from justatom.viewing.mask import IChart
import polars as pl
import pandas as pd
import altair as alt
from loguru import logger
import plotly.express as px
from justatom.etc.types import LineSmoothChoice
from typing import Optional, Union, List, Any
from typing_extensions import Self


class ILineChart(IChart):

    def __init__(self):
        super().__init__()

    def _check_column(self, col: str, columns: list, param_name: str) -> bool:

        return True

    def _check_smoothing_options(self):
        pass

    def view(
        self,
        df: pl.DataFrame,
        x_axis: Optional[str] = None,
        x_title: Optional[str] = "X",
        y_axis: Optional[str] = "y",
        y_title: Optional[str] = "Y",
        scale_domain: Optional[List[int]] = None,
        overlay_mark: Optional[bool] = False,
        smooth: Optional[Union[int, float]] = 1,
        smooth_choice: Optional[str] = None,
        return_only_smooth: Optional[bool] = False,
        smooth_color: Optional[str] = "red",
    ) -> Self:
        columns = df.columns
        x_axis = x_axis or "row_nr"
        if not self._check_column(x_axis, columns, "x_axis") or not self._check_column(y_axis, columns, "y_axis"):
            raise ValueError(f"One or multiple columns is missing in {','.join(columns)}")
        chart_view = None
        if overlay_mark:
            self.chart = (
                alt.Chart(df)
                .mark_line(point=alt.OverlayMarkDef(filled=False, fill="white"))
                .encode(x=alt.X(x_axis).title(x_title), y=alt.Y(y_axis, scale=scale_domain).title(y_title))
            )
        else:
            self.chart = alt.Chart(df).encode(x=alt.X(x_axis).title(x_title), y=alt.Y(y_axis).title(y_title))
        if smooth_choice == "poly":
            chart_view = self.chart.transform_regression(
                x_axis, y_axis, method="poly", order=smooth, as_=[x_axis, y_axis]
            ).mark_line(color=smooth_color)
        elif smooth_choice == "loess":
            chart_view = self.chart.transform_loess(x_axis, y_axis, bandwidth=smooth, as_=[x_axis, y_axis]).mark_line(
                color=smooth_color
            )
        elif smooth_choice == "moving_average":
            window_size = max(1, smooth)
            chart_view = self.chart.transform_window(
                rolling_mean=f"mean({y_axis})", frame=[-(window_size // 2), window_size // 2], sort=[{"field": x_axis}]
            ).mark_line(color=smooth_color)
        else:
            return self.chart

        if return_only_smooth and chart_view is not None:
            return chart_view
        return alt.layer(self.chart, chart_view)


class IPlotlyChart(IChart):

    def __init__(self) -> None:
        super().__init__()

    def scatter(self, data: Union[pl.DataFrame, pd.DataFrame], **props) -> Any:
        self.chart = px.scatter(data, **props)
        return self.chart

    def save(self, filename, ppi=200):
        self.chart.save(filename, ppi=ppi)
