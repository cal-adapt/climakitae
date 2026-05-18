"""Compute report-style climate metrics from ClimateData results.

This module turns raw daily xarray temperature data (typically WRF ``t2max``
and ``t2min``) into the summary numbers that drive
:mod:`climakitae.visualize` figures:

* Average summer high / low temperature
* Hot days per year (days above a configurable threshold)
* Heat-wave events per year (runs of N+ consecutive hot days)
* Return-period extreme threshold (e.g. 1-in-10-yr daily max)

The functions are pure xarray/numpy — they do not fetch data themselves, so
they can be unit-tested with synthetic fixtures.

Notes
-----
All metrics expect temperatures in degrees Fahrenheit. Use the
``convert_units`` processor on the upstream :class:`ClimateData` query, or
call :func:`climakitae.util.unit_conversions.convert_units` ahead of time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
import xarray as xr

SUMMER_MONTHS: tuple[int, ...] = (6, 7, 8)

_REDUCE_DIMS_DEFAULT: tuple[str, ...] = ("lat", "lon", "x", "y", "simulation", "sim")


def _spatial_mean(
    da: xr.DataArray, dims: tuple[str, ...] = _REDUCE_DIMS_DEFAULT
) -> xr.DataArray:
    """Mean over any of the given dims that are present on ``da``."""
    present = [d for d in dims if d in da.dims]
    return da.mean(dim=present) if present else da


def _summer(da: xr.DataArray) -> xr.DataArray:
    """Subset to boreal summer (Jun-Aug)."""
    if "time" not in da.dims:
        return da
    return da.where(da["time"].dt.month.isin(SUMMER_MONTHS), drop=True)


def average_summer(da: xr.DataArray) -> float:
    """Return the area- and time-averaged summer value as a float (°F)."""
    return float(_spatial_mean(_summer(da)).mean(dim="time", skipna=True).values)


def hot_days_per_year(da_tmax: xr.DataArray, threshold_F: float = 90.0) -> float:
    """Mean number of days per year above ``threshold_F``.

    Parameters
    ----------
    da_tmax : xr.DataArray
        Daily maximum temperature in °F with a ``time`` dimension.
    threshold_F : float, default 90
        Hot-day threshold in °F.
    """
    da = _spatial_mean(da_tmax)
    counts = (da > threshold_F).groupby("time.year").sum("time")
    return float(counts.mean().values)


def heat_waves_per_year(
    da_tmax: xr.DataArray,
    threshold_F: float = 90.0,
    min_consecutive_days: int = 4,
) -> float:
    """Mean count of heat-wave events per year.

    A heat-wave event is a run of ``min_consecutive_days`` or more days where
    daily max exceeds ``threshold_F``. Each qualifying run counts once.

    Notes
    -----
    Uses a simple run-length approach: events are counted as transitions from
    non-hot to a long-enough hot run.
    """
    da = _spatial_mean(da_tmax)
    hot = (da > threshold_F).astype("int8")
    if "time" not in hot.dims:
        raise ValueError("expected a 'time' dimension on da_tmax")

    def _count_events(year_hot: np.ndarray) -> int:
        events = 0
        run = 0
        for v in year_hot:
            if v:
                run += 1
                if run == min_consecutive_days:
                    events += 1
            else:
                run = 0
        return events

    per_year = hot.groupby("time.year").map(
        lambda g: xr.DataArray(_count_events(g.values))
    )
    return float(per_year.mean().values)


def extreme_threshold(
    da_tmax: xr.DataArray,
    return_period_years: int = 10,
) -> float:
    """Empirical 1-in-N-year daily maximum (°F) via annual maxima percentile.

    A pragmatic estimator that does not require scipy/GEV fitting: takes the
    annual maxima series (area-averaged) and returns the
    ``1 - 1/N`` quantile.  For tail estimates the user should prefer LOCA2
    and the ``metric_calc`` ``one_in_x`` processor; this helper is a
    lightweight fallback usable in figure code.
    """
    da = _spatial_mean(da_tmax)
    annual_max = da.groupby("time.year").max("time")
    q = 1.0 - 1.0 / float(return_period_years)
    return float(annual_max.quantile(q).values)


@dataclass(frozen=True)
class PeriodInputs:
    """Container for the daily temperature data of a single period."""

    tmax: xr.DataArray
    tmin: xr.DataArray | None = None


def compute_report_metrics(
    periods: Mapping[str, PeriodInputs],
    hot_day_threshold_F: float = 90.0,
    heatwave_min_days: int = 4,
    return_period_years: int = 10,
) -> pd.DataFrame:
    """Compute the full report-metrics table.

    Parameters
    ----------
    periods : mapping
        Ordered mapping of period label → :class:`PeriodInputs`. Keys appear
        as columns in the returned table — e.g.
        ``{"Historic (1981-2010)": ..., "1.5°C": ..., "2.0°C": ...}``.
    hot_day_threshold_F : float, default 90
    heatwave_min_days : int, default 4
    return_period_years : int, default 10

    Returns
    -------
    pandas.DataFrame
        Rows = metric names, columns = period labels, values are floats.
    """
    rows: dict[str, dict[str, float]] = {
        "Average High (°F)": {},
        "Average Low (°F)": {},
        f"Hot Days >{int(hot_day_threshold_F)}°F / yr": {},
        f"Heat Waves ({heatwave_min_days}+ days) / yr": {},
        f"1-in-{return_period_years}-yr Daily Max (°F)": {},
    }
    for label, p in periods.items():
        rows["Average High (°F)"][label] = average_summer(p.tmax)
        rows["Average Low (°F)"][label] = (
            average_summer(p.tmin) if p.tmin is not None else float("nan")
        )
        rows[f"Hot Days >{int(hot_day_threshold_F)}°F / yr"][label] = hot_days_per_year(
            p.tmax, hot_day_threshold_F
        )
        rows[f"Heat Waves ({heatwave_min_days}+ days) / yr"][label] = (
            heat_waves_per_year(p.tmax, hot_day_threshold_F, heatwave_min_days)
        )
        rows[f"1-in-{return_period_years}-yr Daily Max (°F)"][label] = (
            extreme_threshold(p.tmax, return_period_years)
        )

    return pd.DataFrame(rows).T[list(periods.keys())]


__all__ = [
    "PeriodInputs",
    "SUMMER_MONTHS",
    "average_summer",
    "compute_report_metrics",
    "extreme_threshold",
    "heat_waves_per_year",
    "hot_days_per_year",
]
