"""Shared visual style for climakitae report-style figures.

Provides a small, opinionated matplotlib palette inspired by Cal-Adapt
report graphics — navy text on cream/white panels with orange (projection)
and gold (historical) accents.

Examples
--------
>>> from climakitae.visualize.style import apply_style, cae_report_style, COLORS
>>> apply_style()                       # set rcParams globally
>>> with cae_report_style():            # temporary scope
...     fig, ax = plt.subplots()
...     ax.bar(["a", "b"], [1, 2], color=[COLORS["historical"], COLORS["projection"]])
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import matplotlib as mpl
from matplotlib import pyplot as plt

COLORS: dict[str, str] = {
    "navy": "#1F2A44",
    "orange": "#E07A2B",
    "gold": "#F2B441",
    "cream": "#FBF4E8",
    "ink": "#1F2A44",
    "muted": "#6B7280",
    "rule": "#E5DCC8",
    "historical": "#F2B441",
    "projection": "#E07A2B",
    "accent": "#1F2A44",
}

REPORT_RCPARAMS: dict[str, object] = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": COLORS["navy"],
    "axes.labelcolor": COLORS["navy"],
    "axes.titlecolor": COLORS["navy"],
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": COLORS["rule"],
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "text.color": COLORS["navy"],
    "xtick.color": COLORS["navy"],
    "ytick.color": COLORS["navy"],
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
    "font.size": 11,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
}


def apply_style() -> None:
    """Apply the report style to the global matplotlib rcParams."""
    mpl.rcParams.update(REPORT_RCPARAMS)


@contextmanager
def cae_report_style() -> Iterator[None]:
    """Context manager that temporarily applies the report style.

    Yields
    ------
    None
        Restores the previous rcParams on exit.
    """
    with plt.rc_context(REPORT_RCPARAMS):
        yield


__all__ = ["COLORS", "REPORT_RCPARAMS", "apply_style", "cae_report_style"]
