from typing import Any, List, Optional, Union

import altair as alt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from loguru import logger
from plotly.subplots import make_subplots
from typing_extensions import Self

from justatom.etc.types import LineSmoothChoice
from justatom.viewing.mask import IChart


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
        if not self._check_column(x_axis, columns, "x_axis") or not self._check_column(
            y_axis, columns, "y_axis"
        ):
            raise ValueError(
                f"One or multiple columns is missing in {','.join(columns)}"
            )
        chart_view = None
        if overlay_mark:
            self.chart = (
                alt.Chart(df)
                .mark_line(point=alt.OverlayMarkDef(filled=False, fill="white"))
                .encode(
                    x=alt.X(x_axis).title(x_title),
                    y=alt.Y(y_axis, scale=scale_domain).title(y_title),
                )
            )
        else:
            self.chart = alt.Chart(df).encode(
                x=alt.X(x_axis).title(x_title), y=alt.Y(y_axis).title(y_title)
            )
        if smooth_choice == "poly":
            chart_view = self.chart.transform_regression(
                x_axis, y_axis, method="poly", order=smooth, as_=[x_axis, y_axis]
            ).mark_line(color=smooth_color)
        elif smooth_choice == "loess":
            chart_view = self.chart.transform_loess(
                x_axis, y_axis, bandwidth=smooth, as_=[x_axis, y_axis]
            ).mark_line(color=smooth_color)
        elif smooth_choice == "moving_average":
            window_size = max(1, smooth)
            chart_view = self.chart.transform_window(
                rolling_mean=f"mean({y_axis})",
                frame=[-(window_size // 2), window_size // 2],
                sort=[{"field": x_axis}],
            ).mark_line(color=smooth_color)
        else:
            return self.chart

        if return_only_smooth and chart_view is not None:
            return chart_view
        return alt.layer(self.chart, chart_view)


class PlotlyScatterChart(IChart):

    def view(
        self,
        data: Union[pl.DataFrame, pd.DataFrame],
        label_to_view: str = "title",
        max_label_length: int = 22,
        logo_path: Optional[str] = None,
        **props,
    ) -> Any:
        pl_data = data.rename({"label": label_to_view})
        pl_data = pl_data.with_columns(
            pl.col(label_to_view).apply(
                lambda x: (
                    (x[:max_label_length] + "...") if len(x) > max_label_length else x
                )
            )
        )
        pd_data = pl_data.to_pandas()
        fig = px.scatter(
            pd_data,
            x="x",
            y="y",
            color=label_to_view,
            hover_data={label_to_view: True, "text": False},
        )
        fig.update_layout(
            plot_bgcolor="black", paper_bgcolor="black", font=dict(color="white")
        )

        if logo_path is not None:
            fig.add_layout_image(
                dict(
                    source=str(
                        Path(os.getcwd()) / ".data" / "polaroids.ai.logo.png"
                    ),  # Path to your logo file
                    xref="paper",
                    yref="paper",
                    x=1,
                    y=1,
                    sizex=0.1,
                    sizey=0.1,
                    xanchor="right",
                    yanchor="bottom",
                )
            )

        # # Add a "Powered by" text next to the logo
        # fig.add_annotation(
        # dict(
        #     x=0.98,
        #     y=1.1,
        #     xref='paper',
        #     yref='paper',
        #     text="Powered by",
        #     showarrow=False,
        #     font=dict(
        #         family="Arial",
        #         size=12,
        #         color="yellow"
        #     ),
        #     align="right"
        # )
        # )

        return fig

    def save(self, filename, **kwargs):
        self.chart.write_image(filename, **kwargs)


class PlotlyBarChart(IChart):

    def view(self, data: Union[pl.DataFrame, pd.DataFrame], **props) -> Any:
        self.chart = px.bar(data, **props)
        return self.chart

    def save(self, filename, **kwargs):
        self.chart.write_image(filename, **kwargs)


class PlotlyGroupedBarChart(IChart):
    def __init__(
        self,
        group_col_a: str,
        group_col_b: str,
        distance_col: str = "distance",
        dist_threshold: float = 0.9,
        height: int = 1000,
        width: int = 1600,
        font_size: int = 10,
        max_len_for_name: int = 50,
    ) -> None:
        super().__init__()

        self.group_col_a = group_col_a
        self.group_col_b = group_col_b
        self.distance_col = distance_col
        self.dist_threshold = dist_threshold
        self.height = height
        self.width = width
        self.font_size = font_size
        self.max_len_for_name = max_len_for_name

    def _prepare_group_inter_dataframe(
        self, data: pl.DataFrame, group_val_a: str, group_val_b: str
    ) -> pl.DataFrame:
        sub_df = data.filter(
            pl.col(self.group_col_a) == group_val_a,
            pl.col(self.group_col_b) == group_val_b,
        )

        sub_df = sub_df.with_columns(
            (pl.col(self.distance_col) >= self.dist_threshold)
            .map_dict({True: ">=", False: "<"})
            .alias("thresholded")
        )

        sub_df = sub_df.sort(pl.col(self.distance_col))

        return sub_df

    def view(
        self,
        data: Union[pl.DataFrame, pd.DataFrame],
        histfunc="count",
        histnorm="percent",
        **props,
    ) -> Any:
        if isinstance(data, pd.DataFrame):
            raise NotImplementedError("not implemented for pandas.DataFrame")

        if (
            self.group_col_a not in data.schema
            or self.group_col_b not in data.schema
            or self.distance_col not in data.schema
        ):
            raise KeyError(
                f"one of columns ({self.group_col_a}, {self.group_col_b}, {self.distance_col}) is not found in data"
            )

        figs = []
        subplot_titles = []

        cuts_group_a = data.select(self.group_col_a).to_series().unique().to_list()
        cuts_group_b = data.select(self.group_col_b).to_series().unique().to_list()

        n, m = len(cuts_group_a), len(cuts_group_b)

        for pos_a in range(n):
            for pos_b in range(m):

                sub_df = self._prepare_group_inter_dataframe(
                    data, cuts_group_a[pos_a], cuts_group_b[pos_b]
                )

                sub_size = len(sub_df)
                sub_ratio = len(sub_df) / len(data)

                if len(sub_df) == 0:
                    sub_fig = go.Histogram()
                    figs.append(sub_fig)
                    subplot_titles.append("")
                    continue

                cur_short_name_group_a = (
                    sub_df.select(self.group_col_a)
                    .to_series()
                    .item(0)[: self.max_len_for_name]
                )

                cur_short_name_group_b = (
                    sub_df.select(self.group_col_b)
                    .to_series()
                    .item(0)[: self.max_len_for_name]
                )

                sub_fig = go.Histogram(
                    histfunc=histfunc,
                    x=sub_df["thresholded"],
                    name=f"{cur_short_name_group_a} + {cur_short_name_group_b}",
                    histnorm=histnorm,
                    **props,
                )
                figs.append(sub_fig)
                subplot_titles.append(f"{sub_size}({sub_ratio:0.1%})")

        fig = make_subplots(rows=n, cols=m, subplot_titles=subplot_titles)

        for pos_a in range(n):
            for pos_b in range(m):
                fig.add_trace(figs[pos_a * m + pos_b], row=pos_a + 1, col=pos_b + 1)

        fig.update_layout(height=self.height, width=self.width)
        fig.update_annotations(font_size=self.font_size)

        self.chart = fig

        return self.chart

    def save(self, filename, **kwargs):
        self.chart.write_image(filename, **kwargs)
