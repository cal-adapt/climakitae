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
    col_w = 1.0
    label_w = 1.2
    total_w = label_w + n_cols * col_w
    row_h = 1.0
    header_h = 1.2

    ax.set_xlim(0, total_w)
    ax.set_ylim(0, n_rows * row_h + header_h)
    ax.invert_yaxis()
    ax.set_axis_off()

    # Outer border
    ax.add_patch(
        plt.Rectangle(
            (0.0, 0.0),
            total_w,
            n_rows * row_h + header_h,
            facecolor="none",
            edgecolor=COLORS["table_border"],
            linewidth=0.8,
        )
    )

    # Header row background
    ax.add_patch(
        plt.Rectangle(
            (0.0, 0.0),
            total_w,
            header_h,
            facecolor=header_color,
            edgecolor="none",
        )
    )

    # Column header labels
    for j, col in enumerate(df.columns):
        # Wrap long headers cleanly
        wrapped = textwrap.fill(str(col), width=16)
        ax.text(
            label_w + (j + 0.5) * col_w,
            header_h / 2,
            wrapped,
            ha="center",
            va="center",
            multialignment="center",
            color=COLORS["navy"],
            fontsize=10,
            fontweight="bold",
        )

    # Vertical dividers between columns
    for j in range(1, n_cols):
        ax.plot(
            [label_w + j * col_w, label_w + j * col_w],
            [0, n_rows * row_h + header_h],
            color=COLORS["table_border"],
            linewidth=0.5,
        )

    # Label / data column divider
    ax.plot(
        [label_w, label_w],
        [0, n_rows * row_h + header_h],
        color=COLORS["table_border"],
        linewidth=0.8,
    )

    # Bottom rule for header
    ax.plot(
        [0.0, total_w],
        [header_h, header_h],
        color=COLORS["orange"],
        linewidth=1.5,
    )

    # Rows
    for i, metric in enumerate(df.index):
        y_top = header_h + i * row_h

        # Alternating row tint
        if i % 2 == 1:
            ax.add_patch(
                plt.Rectangle(
                    (0.0, y_top),
                    total_w,
                    row_h,
                    facecolor=COLORS["row_alt"],
                    edgecolor="none",
                    zorder=0,
                )
            )

        # Row separator (solid, subtle)
        ax.plot(
            [0.0, total_w],
            [y_top + row_h, y_top + row_h],
            color=COLORS["table_border"],
            linewidth=0.5,
            zorder=1,
        )

        # Metric label
        label_txt = (
            textwrap.fill(str(metric), width=20)
            if len(str(metric)) > 20
            else str(metric)
        )
        ax.text(
            label_w - 0.12,
            y_top + row_h / 2,
            label_txt,
            ha="right",
            va="center",
            multialignment="right",
            color=COLORS["navy"],
            fontsize=10,
            fontweight="bold",
            zorder=2,
        )

        # Cell values
        for j, col in enumerate(df.columns):
            v = df.iat[i, j]
            txt = value_fmt.format(v) if isinstance(v, (int, float)) else str(v)
            ax.text(
                label_w + (j + 0.5) * col_w,
                y_top + row_h / 2,
                txt,
                ha="center",
                va="center",
                color=value_color,
                fontsize=16,
                fontweight="bold",
                zorder=2,
            )

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
        figsize = (1.6 + 2.0 * n_cols, 0.9 + 0.65 * n_rows)

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
