"""Tests for :mod:`climakitae.visualize.metrics`.

Uses synthetic xarray fixtures so the metrics can be checked against known
analytical values without touching real climate data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.visualize.metrics import (PeriodInputs, average_summer,
                                          compute_report_metrics,
                                          extreme_threshold,
                                          heat_waves_per_year,
                                          hot_days_per_year)


@pytest.fixture
def daily_tmax_F() -> xr.DataArray:
    """Two years of daily tmax (°F), area-averaged (no spatial dims)."""
    time = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    rng = np.random.default_rng(42)
    base = 50 + 30 * np.sin(2 * np.pi * (time.dayofyear / 365.25))
    noise = rng.normal(0, 3, size=len(time))
    return xr.DataArray(base + noise, dims="time", coords={"time": time})


@pytest.fixture
def known_tmax_F() -> xr.DataArray:
    """Deterministic series: 100 °F for days 0-9 of each year, 70 °F otherwise."""
    time = pd.date_range("2000-01-01", "2002-12-31", freq="D")
    values = np.full(len(time), 70.0)
    for year in (2000, 2001, 2002):
        mask = (time.year == year) & (time.dayofyear <= 10)
        values[mask] = 100.0
    return xr.DataArray(values, dims="time", coords={"time": time})


def test_average_summer_uses_only_jja(daily_tmax_F: xr.DataArray) -> None:
    summer = daily_tmax_F.where(
        daily_tmax_F["time"].dt.month.isin([6, 7, 8]), drop=True
    )
    assert average_summer(daily_tmax_F) == pytest.approx(float(summer.mean()), rel=1e-6)


def test_hot_days_per_year_counts_deterministic(known_tmax_F: xr.DataArray) -> None:
    assert hot_days_per_year(known_tmax_F, threshold_F=90.0) == pytest.approx(10.0)


def test_hot_days_per_year_below_threshold_is_zero(known_tmax_F: xr.DataArray) -> None:
    assert hot_days_per_year(known_tmax_F, threshold_F=200.0) == 0.0


def test_heat_waves_per_year_one_long_event(known_tmax_F: xr.DataArray) -> None:
    assert (
        heat_waves_per_year(known_tmax_F, threshold_F=90.0, min_consecutive_days=4)
        == 1.0
    )


def test_heat_waves_per_year_requires_min_length(known_tmax_F: xr.DataArray) -> None:
    assert (
        heat_waves_per_year(known_tmax_F, threshold_F=90.0, min_consecutive_days=20)
        == 0.0
    )


def test_extreme_threshold_returns_finite_value(daily_tmax_F: xr.DataArray) -> None:
    val = extreme_threshold(daily_tmax_F, return_period_years=10)
    assert np.isfinite(val)
    assert val > float(daily_tmax_F.mean())


def test_compute_report_metrics_shape(known_tmax_F: xr.DataArray) -> None:
    tmin = known_tmax_F - 25.0
    periods = {
        "Historic": PeriodInputs(tmax=known_tmax_F, tmin=tmin),
        "1.5C": PeriodInputs(tmax=known_tmax_F + 2, tmin=tmin + 2),
        "2.0C": PeriodInputs(tmax=known_tmax_F + 3, tmin=tmin + 3),
    }
    df = compute_report_metrics(periods)
    assert list(df.columns) == ["Historic", "1.5C", "2.0C"]
    assert df.shape[0] == 5
    assert df.loc["Average High (°F)", "1.5C"] > df.loc["Average High (°F)", "Historic"]


def test_spatial_dims_are_reduced() -> None:
    time = pd.date_range("2000-01-01", "2000-12-31", freq="D")
    arr = np.full((len(time), 3, 3), 80.0)
    da = xr.DataArray(arr, dims=("time", "lat", "lon"), coords={"time": time})
    assert average_summer(da) == pytest.approx(80.0)
    assert hot_days_per_year(da, threshold_F=70.0) == pytest.approx(366.0)
