"""Compute report-style climate metrics from ClimateData results.

This module turns raw daily xarray temperature data (typically WRF ``t2max``
and ``t2min``) into the summary numbers that drive
:mod:`climakitae.visualize` figures:

* Average summer high / low temperature
* Hot days per year (days above a configurable threshold)
* Average heat-wave duration in days (runs of N+ consecutive hot days)
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

# Spatial and simulation dimension names handled separately so that the
# simulation median is always taken *last*, after temporal aggregation.
_SPATIAL_DIMS: tuple[str, ...] = ("lat", "lon", "x", "y")
_SIM_DIMS: tuple[str, ...] = ("simulation", "sim")


def _spatial_mean(da: xr.DataArray) -> xr.DataArray:
    """Mean over spatial dims only; simulation dimension is preserved."""
    present = [d for d in _SPATIAL_DIMS if d in da.dims]
    return da.mean(dim=present) if present else da


def _sim_median(da: xr.DataArray) -> xr.DataArray:
    """Median over simulation dims if present; otherwise identity.

    Notes
    -----
    xarray's ``median`` is not implemented for dask-backed arrays.  By the
    time this helper is called the array has already been reduced to at most
    one remaining (simulation) dimension, so ``load()`` is cheap.
    """
    present = [d for d in _SIM_DIMS if d in da.dims]
    if not present:
        return da
    return da.load().median(dim=present)


def _summer(da: xr.DataArray) -> xr.DataArray:
    """Subset to boreal summer (Jun-Aug)."""
    if "time" not in da.dims:
        return da
    return da.where(da["time"].dt.month.isin(SUMMER_MONTHS), drop=True)


def average_summer(da: xr.DataArray) -> float:
    """Return the area- and time-averaged summer value as a float (°F).

    Reduction order: spatial mean → time mean → simulation median.
    """
    da = _spatial_mean(_summer(da))
    per_sim = da.mean(dim="time", skipna=True)
    return float(_sim_median(per_sim).values)


def hot_days_per_year(da_tmax: xr.DataArray, threshold_F: float = 90.0) -> float:
    """Median-across-simulations mean number of days per year above ``threshold_F``.

    Parameters
    ----------
    da_tmax : xr.DataArray
        Daily maximum temperature in °F with a ``time`` dimension.
    threshold_F : float, default 90
        Hot-day threshold in °F.

    Notes
    -----
    Reduction order: spatial mean → annual sum → mean over years →
    median across simulations.
    """
    da = _spatial_mean(da_tmax)
    counts = (da > threshold_F).groupby("time.year").sum("time")
    per_sim = counts.mean("year")
    return float(_sim_median(per_sim).values)


def avg_heat_wave_length(
    da_tmax: xr.DataArray,
    threshold_F: float = 90.0,
    min_consecutive_days: int = 4,
) -> float:
    """Median-across-simulations mean heat-wave duration in days.

    A heat-wave event is a run of ``min_consecutive_days`` or more
    consecutive days where the spatially-averaged daily max exceeds
    ``threshold_F``. The mean length (in days) of all qualifying events
    across the full period is computed per simulation; the simulation
    median is returned.

    Returns 0.0 when no qualifying heat-wave events exist.

    Parameters
    ----------
    da_tmax : xr.DataArray
        Daily maximum temperature in °F with a ``time`` dimension.
    threshold_F : float, default 90
        Hot-day threshold in °F.
    min_consecutive_days : int, default 4
        Minimum run length to qualify as a heat-wave event.

    Notes
    -----
    Reduction order: spatial mean → mean event length over full period →
    median across simulations.
    """
    da = _spatial_mean(da_tmax)

    def _mean_event_len(arr: np.ndarray) -> float:
        lengths: list[int] = []
        run = 0
        for v in arr:
            if v:
                run += 1
            else:
                if run >= min_consecutive_days:
                    lengths.append(run)
                run = 0
        if run >= min_consecutive_days:  # trailing run
            lengths.append(run)
        return float(np.mean(lengths)) if lengths else 0.0

    sim_dim = next((d for d in _SIM_DIMS if d in da.dims), None)
    if sim_dim is not None:
        hot = (da > threshold_F).astype("int8")
        per_sim = [
            _mean_event_len(hot.isel({sim_dim: i}).values)
            for i in range(hot.sizes[sim_dim])
        ]
        return float(np.median(per_sim))

    hot = (da > threshold_F).astype("int8")
    return _mean_event_len(hot.values)


def extreme_threshold(
    da_tmax: xr.DataArray,
    return_period_years: int = 10,
) -> float:
    """Empirical 1-in-N-year daily maximum (°F) via annual maxima quantile.

    A pragmatic estimator that does not require scipy/GEV fitting: takes the
    annual maxima series (spatially averaged) and returns the
    ``1 - 1/N`` quantile per simulation, then takes the simulation median.
    For tail estimates the user should prefer LOCA2 and the ``metric_calc``
    ``one_in_x`` processor; this helper is a lightweight fallback usable in
    figure code.

    Notes
    -----
    Reduction order: spatial mean → annual max → return-period quantile →
    median across simulations.
    """
    da = _spatial_mean(da_tmax)
    annual_max = da.groupby("time.year").max("time")
    q = 1.0 - 1.0 / float(return_period_years)
    per_sim = annual_max.quantile(q, dim="year")
    return float(_sim_median(per_sim).values)


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
        f"Avg Heat Wave Duration (days)": {},
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
        rows[f"Avg Heat Wave Duration (days)"][label] = avg_heat_wave_length(
            p.tmax, hot_day_threshold_F, heatwave_min_days
        )
        rows[f"1-in-{return_period_years}-yr Daily Max (°F)"][label] = (
            extreme_threshold(p.tmax, return_period_years)
        )

    return pd.DataFrame(rows).T[list(periods.keys())]


__all__ = [
    "PeriodInputs",
    "SUMMER_MONTHS",
    "avg_heat_wave_length",
    "average_summer",
    "compute_report_metrics",
    "extreme_threshold",
    "hot_days_per_year",
]
