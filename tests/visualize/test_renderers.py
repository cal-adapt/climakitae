"""Smoke tests for :mod:`climakitae.visualize` renderers.

Verifies that each renderer produces a :class:`matplotlib.figure.Figure`
without errors using ``matplotlib`` in a non-interactive backend.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from climakitae.visualize import build_report_figure  # noqa: E402
from climakitae.visualize import (render_stat_cards, render_summary_table,
                                  render_threshold_bars)


def test_render_stat_cards_returns_figure() -> None:
    fig = render_stat_cards(
        [("2-3°F", "Average summer rise"), ("10x", "Heat waves"), ("33", "Hot days")]
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3


def test_render_stat_cards_requires_items() -> None:
    with pytest.raises(ValueError):
        render_stat_cards([])


def test_render_summary_table_returns_figure() -> None:
    df = pd.DataFrame(
        {
            "Historic": [72.4, 48.2, 24.0, 0.0],
            "1.5°C": [75.5, 51.2, 51.0, 8.0],
            "2.0°C": [78.6, 54.0, 78.0, 15.0],
        },
        index=["Avg High", "Avg Low", "Hot Days/yr", "Heat Waves/yr"],
    )
    fig = render_summary_table(df, title="Test Summary")
    assert isinstance(fig, Figure)


def test_render_threshold_bars_returns_figure() -> None:
    df = pd.DataFrame(
        {"Historical": [93.2, 97.6, 100.8], "Projection": [98.6, 103.7, 106.8]},
        index=["Countywide", "Reseda", "Lancaster"],
    )
    fig = render_threshold_bars(df, title="Extreme Heat")
    assert isinstance(fig, Figure)


def test_render_threshold_bars_validates_columns() -> None:
    df = pd.DataFrame({"foo": [1, 2]}, index=["a", "b"])
    with pytest.raises(ValueError):
        render_threshold_bars(df)


def test_build_report_figure_returns_figure() -> None:
    stats = [("2-3°F", "Summer rise"), ("10x", "Heat waves"), ("33", "Hot days")]
    summary = pd.DataFrame(
        {"Historic": [72.4, 48.2], "1.5°C": [75.5, 51.2], "2.0°C": [78.6, 54.0]},
        index=["Avg High", "Avg Low"],
    )
    bars = pd.DataFrame(
        {"Historical": [93.2, 97.6], "Projection": [98.6, 103.7]},
        index=["Countywide", "Reseda"],
    )
    fig = build_report_figure(
        title="Test County Climate Report",
        subtitle="Synthetic data",
        stat_items=stats,
        summary_df=summary,
        bars_df=bars,
    )
    assert isinstance(fig, Figure)
