"""Stat-card panel for report-style figures.

Renders a horizontal row of "callout" cards, each showing a big bold value
and a short caption — the visual idiom used at the top of Cal-Adapt
county climate reports.
"""

from __future__ import annotations

import textwrap
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from .style import COLORS, cae_report_style

StatItem = tuple[str, str]


def _draw_card(
    ax: Axes,
    value: str,
    caption: str,
    *,
    value_color: str,
    card_color: str,
    value_fontsize: float = 32,
    caption_fontsize: float = 10,
) -> None:
    """Draw one stat card on ``ax`` (expects an empty axis)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    ax.add_patch(
        FancyBboxPatch(
            (0.04, 0.08),
            0.92,
            0.84,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=0,
            facecolor=card_color,
        )
    )
    ax.text(
        0.5,
        0.66,
        value,
        ha="center",
        va="center",
        color=value_color,
        fontsize=value_fontsize,
        fontweight="bold",
    )
    wrapped = textwrap.fill(caption, width=22)
    ax.text(
        0.5,
        0.26,
        wrapped,
        ha="center",
        va="center",
        multialignment="center",
        color=COLORS["navy"],
        fontsize=caption_fontsize,
    )


def render_stat_cards(
    items: Sequence[StatItem],
    *,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    value_color: str = COLORS["orange"],
    card_color: str = COLORS["cream"],
) -> Figure:
    """Render a row of stat callout cards as a standalone figure.

    Parameters
    ----------
    items : sequence of (value, caption) tuples
        Each tuple becomes one card.
    figsize : tuple, optional
        Defaults to a width that scales with the number of cards.
    title : str, optional
        Optional row title rendered above the cards.
    value_color, card_color : str
        Big-number color and card background color.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not items:
        raise ValueError("render_stat_cards requires at least one item")

    n = len(items)
    if figsize is None:
        figsize = (3.6 * n, 2.6)

    with cae_report_style():
        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]
        for ax, (value, caption) in zip(axes, items):
            _draw_card(
                ax, value, caption, value_color=value_color, card_color=card_color
            )
        if title:
            fig.suptitle(title, color=COLORS["navy"], fontweight="bold")
        fig.tight_layout()
    return fig


__all__ = ["StatItem", "_draw_card", "render_stat_cards"]
