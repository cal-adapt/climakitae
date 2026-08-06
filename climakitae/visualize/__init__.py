"""Report-style figure components for climakitae.

Public entry points
-------------------
- :func:`render_stat_cards` — top-of-page callout cards
- :func:`render_summary_table` — metric × period table
- :func:`render_threshold_bars` — grouped historical-vs-projection bars
- :func:`build_report_figure` — composite single-page figure
- :func:`compute_report_metrics` — turn xarray temperature data into the
  metric table used by :func:`render_summary_table`
- :func:`apply_style`, :func:`cae_report_style`, :data:`COLORS` — palette
  and matplotlib rcParams helpers
"""

from .metrics import (
    PeriodInputs,
    avg_heat_wave_length,
    average_summer,
    compute_report_metrics,
    extreme_threshold,
    hot_days_per_year,
)
from .report import build_report_figure
from .stat_cards import render_stat_cards
from .style import COLORS, apply_style, cae_report_style
from .summary_table import render_summary_table
from .threshold_bars import render_threshold_bars

__all__ = [
    "COLORS",
    "PeriodInputs",
    "apply_style",
    "average_summer",
    "build_report_figure",
    "cae_report_style",
    "compute_report_metrics",
    "avg_heat_wave_length",
    "extreme_threshold",
    "hot_days_per_year",
    "render_stat_cards",
    "render_summary_table",
    "render_threshold_bars",
]
