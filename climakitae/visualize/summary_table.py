"""Summary metrics table rendered as a matplotlib figure.

Reproduces the Fig 1 style of Cal-Adapt county reports: a clean table where
each row is a metric and each column is a time period, with large bold
values and thin rule separators.
"""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .style import COLORS, cae_report_style


def _draw_summary_table(
    ax: Axes,
    df: pd.DataFrame,
    *,
    value_fmt: str = "{:.1f}",
    header_color: str = "#EDE0C4",
    value_color: str = COLORS["orange"],
    title: str | None = None,
) -> None:
    """Draw a metric × period table directly onto ``ax``.

    The axis is reconfigured (limits set, ticks/spines hidden) so callers
    should pass a dedicated axis.
    """
    if df.empty:
        raise ValueError("_draw_summary_table requires a non-empty DataFrame")

    n_rows, n_cols = df.shape
    ax.set_xlim(0, n_cols + 1)
    ax.set_ylim(0, n_rows + 1.2)
    ax.invert_yaxis()
    ax.set_axis_off()

    ax.add_patch(
        plt.Rectangle(
            (0.95, 0.0),
            n_cols,
            1.0,
            facecolor=header_color,
            edgecolor="none",
        )
    )
    for j, col in enumerate(df.columns):
        ax.text(
            1.5 + j,
            0.5,
            str(col),
            ha="center",
            va="center",
            color=COLORS["navy"],
            fontsize=12,
            fontweight="bold",
        )

    for i, metric in enumerate(df.index):
        y = 1.0 + i + 0.5
        label_txt = (
            textwrap.fill(str(metric), width=22)
            if len(str(metric)) > 22
            else str(metric)
        )
        ax.text(
            0.9,
            y,
            label_txt,
            ha="right",
            va="center",
            multialignment="right",
            color=COLORS["navy"],
            fontsize=10,
            fontweight="bold",
        )
        ax.plot(
            [0.95, n_cols + 0.95],
            [1.0 + i + 1.0, 1.0 + i + 1.0],
            color=COLORS["rule"],
            linewidth=0.8,
            linestyle=(0, (2, 2)),
        )
        for j, col in enumerate(df.columns):
            v = df.iat[i, j]
            txt = value_fmt.format(v) if isinstance(v, (int, float)) else str(v)
            ax.text(
                1.5 + j,
                y,
                txt,
                ha="center",
                va="center",
                color=value_color,
                fontsize=18,
                fontweight="bold",
            )

    ax.plot([0.95, n_cols + 0.95], [1.0, 1.0], color=COLORS["orange"], linewidth=1.2)
    if title:
        ax.set_title(title, loc="left", pad=12)


def render_summary_table(
    df: pd.DataFrame,
    *,
    title: str | None = None,
    value_fmt: str = "{:.1f}",
    figsize: tuple[float, float] | None = None,
    header_color: str = "#EDE0C4",
    value_color: str = COLORS["orange"],
) -> Figure:
    """Render a metric × period summary table as a standalone figure."""
    if df.empty:
        raise ValueError("render_summary_table requires a non-empty DataFrame")
    n_rows, n_cols = df.shape
    if figsize is None:
        figsize = (2.8 + 1.8 * n_cols, 1.2 + 0.7 * n_rows)

    with cae_report_style():
        fig, ax = plt.subplots(figsize=figsize)
        _draw_summary_table(
            ax,
            df,
            value_fmt=value_fmt,
            header_color=header_color,
            value_color=value_color,
            title=title,
        )
        fig.tight_layout()
    return fig


__all__ = ["_draw_summary_table", "render_summary_table"]
