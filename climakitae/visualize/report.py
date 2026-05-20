"""Composite report-page figure that assembles all the report components.

Stacks: title strip → stat cards → summary table → grouped threshold bars
into a single :class:`matplotlib.figure.Figure` suitable for export.

All components are drawn directly onto a shared parent figure via nested
:class:`matplotlib.gridspec.GridSpec` regions so the composite composes
cleanly at any figsize.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from .stat_cards import StatItem, _draw_card
from .style import COLORS, cae_report_style
from .summary_table import _draw_summary_table, render_summary_table  # noqa: F401
from .threshold_bars import _draw_threshold_bars, render_threshold_bars  # noqa: F401


def build_report_figure(
    *,
    title: str,
    subtitle: str | None,
    stat_items: Sequence[StatItem],
    summary_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    bars_historical_col: str = "Historical",
    bars_projection_col: str = "Projection",
    bars_title: str = "Extreme Heat Thresholds: Historical vs Projected",
    bars_ymin: float | None = None,
    bars_ymax: float | None = None,
    figsize: tuple[float, float] = (11.0, 13.0),
    footnote: str | None = None,
    tagline: str | None = None,
) -> Figure:
    """Build a single-page report-style figure.

    Layout (top → bottom): title strip, optional tagline sentence, stat cards
    row, summary table, grouped historical-vs-projection bar chart.

    Parameters
    ----------
    title : str
        Big page title.
    subtitle : str, optional
        Smaller line under the title.
    stat_items : sequence of (value, caption)
        Items for the stat cards row.
    summary_df : pandas.DataFrame
        Metric × period table.
    bars_df : pandas.DataFrame
        Location × (Historical, Projection) for the bar chart.
    bars_historical_col, bars_projection_col, bars_title : str
        Forwarded to the bar chart.
    bars_ymin, bars_ymax : float, optional
        Y-axis bounds for the bar chart.  When ``bars_ymin > 0`` a
        double-slash break mark is drawn at the base of the y-axis.
    figsize : tuple, default (11, 13)
        Full figure size in inches.
    tagline : str, optional
        One-sentence interpretive summary rendered in italic below the title
        strip and above the stat cards.  When omitted the row is not drawn.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_cards = len(stat_items)
    if n_cards == 0:
        raise ValueError("stat_items must be non-empty")
    if summary_df.empty:
        raise ValueError("summary_df must be non-empty")

    n_rows = summary_df.shape[0]
    # Compute title-strip height from subtitle line count so the rule never
    # collides with the subtitle text regardless of wrapping.
    subtitle_lines = subtitle.count("\n") + 1 if subtitle else 0
    title_h = 0.55 + 0.28 * subtitle_lines

    # When a tagline is present, insert an extra thin row between the title
    # strip and the stat cards; track row indices symbolically so the rest of
    # the layout code stays readable.
    _TAGLINE_H = 0.5
    if tagline:
        _nrows = 6
        _ratios = [title_h, _TAGLINE_H, 1.6, 0.52 * n_rows + 0.9, 0.25, 4.5]
        _card_row, _table_row, _bars_row = 2, 3, 5
    else:
        _nrows = 5
        _ratios = [title_h, 1.6, 0.52 * n_rows + 0.9, 0.25, 4.5]
        _card_row, _table_row, _bars_row = 1, 2, 4

    with cae_report_style():
        fig = plt.figure(figsize=figsize)
        outer = GridSpec(
            nrows=_nrows,
            ncols=1,
            height_ratios=_ratios,
            hspace=0.18,
            left=0.07,
            right=0.95,
            top=0.96,
            bottom=0.07,
            figure=fig,
        )

        ax_title = fig.add_subplot(outer[0, 0])
        ax_title.set_axis_off()
        ax_title.set_xlim(0, 1)
        ax_title.set_ylim(0, 1)
        # Anchor title at the top; subtitle flows downward from a fixed y so it
        # never overlaps the rule regardless of how many lines it wraps to.
        ax_title.text(
            0.0,
            0.96,
            title,
            color=COLORS["navy"],
            fontsize=22,
            fontweight="bold",
            va="top",
        )
        if subtitle:
            ax_title.text(
                0.0,
                0.52,
                subtitle,
                color=COLORS["muted"],
                fontsize=12,
                va="top",
            )
        ax_title.plot([0.0, 1.0], [0.05, 0.05], color=COLORS["orange"], linewidth=2)

        if tagline:
            ax_tagline = fig.add_subplot(outer[1, 0])
            ax_tagline.set_axis_off()
            ax_tagline.set_xlim(0, 1)
            ax_tagline.set_ylim(0, 1)
            ax_tagline.text(
                0.0,
                0.5,
                tagline,
                color=COLORS["navy"],
                fontsize=11,
                style="italic",
                va="center",
                wrap=True,
            )

        cards_gs = GridSpecFromSubplotSpec(
            1, n_cards, subplot_spec=outer[_card_row, 0], wspace=0.14
        )
        for i, (value, caption) in enumerate(stat_items):
            ax_c = fig.add_subplot(cards_gs[0, i])
            _draw_card(
                ax_c,
                value,
                caption,
                value_color=COLORS["orange"],
                card_color=COLORS["cream"],
            )

        ax_table = fig.add_subplot(outer[_table_row, 0])
        _draw_summary_table(ax_table, summary_df)

        ax_bars = fig.add_subplot(outer[_bars_row, 0])
        _draw_threshold_bars(
            ax_bars,
            bars_df,
            historical_col=bars_historical_col,
            projection_col=bars_projection_col,
            title=bars_title,
            ymin=bars_ymin,
            ymax=bars_ymax,
        )

        if footnote:
            fig.text(
                0.06,
                0.015,
                footnote,
                color=COLORS["muted"],
                fontsize=8,
                ha="left",
                va="bottom",
                wrap=True,
            )

    return fig


__all__ = [
    "build_report_figure",
    "render_summary_table",
    "render_threshold_bars",
]
