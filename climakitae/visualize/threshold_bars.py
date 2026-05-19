"""Grouped historical-vs-projection bar chart.

Reproduces the Fig 4 style of Cal-Adapt county reports: a grouped bar
chart with a Historical (gold) and Projection (orange) bar per location,
value labels on top.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .style import COLORS, cae_report_style


def _draw_axis_break(ax: Axes) -> None:
    """Draw // break markers at the base of the y-axis spine.

    Indicates to the reader that the y-axis does not start at zero.
    Uses the matplotlib marker technique: a custom diagonal-line path
    placed at two axes-fraction coordinates near the bottom-left corner
    of the axes, so it scales correctly with any figure size.

    See also
    --------
    https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
    """
    d = 0.4  # slant ratio (height / width of the diagonal mark)
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=10,
        linestyle="none",
        color=COLORS["navy"],
        mec=COLORS["navy"],
        mew=1.5,
        clip_on=False,
    )
    # Two // marks side-by-side at the base of the y-axis spine
    ax.plot([-0.022, 0.022], [0.0, 0.0], transform=ax.transAxes, **kwargs)


# Colour ramp for up to 4 warming-level periods (Historic → Near → Mid → Late-century)
_PERIOD_COLORS: list[str] = [
    COLORS["historical"],  # Historic baseline — gold
    "#E89C41",  # Near-century — light orange
    COLORS["projection"],  # Mid-century — orange
    "#B35520",  # Late-century — deep orange
]


def _bar_colors(n: int) -> list[str]:
    """Return ``n`` bar colours cycling through the warming-level colour ramp."""
    return [_PERIOD_COLORS[i % len(_PERIOD_COLORS)] for i in range(n)]


def _draw_threshold_bars(
    ax: Axes,
    df: pd.DataFrame,
    *,
    historical_col: str = "Historical",
    projection_col: str = "Projection",
    ylabel: str = "°F",
    value_fmt: str = "{:.1f}",
    ymin: float | None = None,
    ymax: float | None = None,
    title: str | None = None,
    show_legend: bool = True,
) -> None:
    """Draw grouped bars (one per period) for each location onto ``ax``.

    Supports any number of columns in ``df`` — one bar group per row
    (location), one bar per column (period).  Bars are coloured using
    a gold-to-deep-orange ramp so the progression across warming levels
    is visually intuitive.  The y-axis starts at 0 by default so bar
    heights represent honest absolute values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    df : pandas.DataFrame
        Rows = locations, columns = period labels.  Values must be numeric.
    historical_col, projection_col : str
        Retained for backward compatibility; used when ``df`` has exactly
        two columns.  Ignored when ``df`` has more columns.
    ylabel : str
        Y-axis label.
    value_fmt : str
        Format string applied to each bar-top label.
    ymin, ymax : float, optional
        Y-axis bounds.  ``ymin`` defaults to 0 so bars read honestly.
    title : str, optional
        Axes title (left-aligned).
    show_legend : bool
        Whether to show a legend beneath the chart.
    """
    cols = list(df.columns)
    n_series = len(cols)
    if n_series == 0:
        raise ValueError("_draw_threshold_bars: df must have at least one column")

    locations: list[str] = list(df.index.astype(str))
    n_locs = len(locations)
    if n_locs == 0:
        raise ValueError("_draw_threshold_bars requires at least one location")

    # Ensure grid lines render behind bars (default matplotlib behaviour is
    # to draw grid above patches when axes.axisbelow is 'line').
    ax.set_axisbelow(True)

    colors = _bar_colors(n_series)
    x = np.arange(n_locs)
    width = min(0.36, 0.72 / n_series)
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * (width + 0.04)
    label_fontsize = max(7, 11 - n_series)

    for k, col in enumerate(cols):
        vals = df[col].to_numpy(dtype=float)
        bars = ax.bar(
            x + offsets[k],
            vals,
            width,
            color=colors[k],
            label=col,
            edgecolor="none",
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                value_fmt.format(v),
                ha="center",
                va="bottom",
                color=COLORS["navy"],
                fontsize=label_fontsize,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(locations, fontweight="bold")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", pad=12)

    all_vals = df.to_numpy(dtype=float).ravel()
    lo = 0.0 if ymin is None else ymin
    hi = float(np.nanmax(all_vals)) if ymax is None else ymax
    pad = max(2.0, 0.05 * hi)
    ax.set_ylim(lo, hi + pad)

    if lo > 0:
        _draw_axis_break(ax)

    if show_legend:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=min(n_series, 4),
        )
    ax.spines["bottom"].set_color(COLORS["navy"])


def render_threshold_bars(
    df: pd.DataFrame,
    *,
    historical_col: str = "Historical",
    projection_col: str = "Projection",
    title: str | None = None,
    ylabel: str = "°F",
    value_fmt: str = "{:.1f}",
    figsize: tuple[float, float] | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
) -> Figure:
    """Render grouped historical vs projection bars as a standalone figure."""
    n = len(df.index)
    if figsize is None:
        figsize = (2.0 + 1.8 * max(n, 1), 5.0)
    with cae_report_style():
        fig, ax = plt.subplots(figsize=figsize)
        _draw_threshold_bars(
            ax,
            df,
            historical_col=historical_col,
            projection_col=projection_col,
            ylabel=ylabel,
            value_fmt=value_fmt,
            ymin=ymin,
            ymax=ymax,
            title=title,
        )
        fig.tight_layout()
    return fig


__all__ = ["_draw_threshold_bars", "render_threshold_bars"]
