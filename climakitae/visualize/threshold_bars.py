"""Grouped historical-vs-projection bar chart.

Reproduces the Fig 4 style of Cal-Adapt county reports: a grouped bar
chart with a Historical (gold) and Projection (orange) bar per location,
value labels on top.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .style import COLORS, cae_report_style


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
    """Draw grouped Historical vs Projection bars directly onto ``ax``."""
    for col in (historical_col, projection_col):
        if col not in df.columns:
            raise ValueError(f"missing required column: {col!r}")

    locations: Sequence[str] = list(df.index.astype(str))
    n = len(locations)
    if n == 0:
        raise ValueError("_draw_threshold_bars requires at least one location")

    hist = df[historical_col].to_numpy(dtype=float)
    proj = df[projection_col].to_numpy(dtype=float)
    x = np.arange(n)
    width = 0.36

    bars_h = ax.bar(
        x - width / 2,
        hist,
        width,
        color=COLORS["historical"],
        label=historical_col,
        edgecolor="none",
    )
    bars_p = ax.bar(
        x + width / 2,
        proj,
        width,
        color=COLORS["projection"],
        label=projection_col,
        edgecolor="none",
    )
    for bars, vals in ((bars_h, hist), (bars_p, proj)):
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                value_fmt.format(v),
                ha="center",
                va="bottom",
                color=COLORS["navy"],
                fontsize=10,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(locations, fontweight="bold")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", pad=12)

    all_vals = np.concatenate([hist, proj])
    lo = float(np.nanmin(all_vals)) if ymin is None else ymin
    hi = float(np.nanmax(all_vals)) if ymax is None else ymax
    pad = max(2.0, 0.08 * (hi - lo if hi > lo else hi))
    ax.set_ylim(
        lo - pad if ymin is None else ymin,
        hi + pad if ymax is None else ymax,
    )
    if show_legend:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2)
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
