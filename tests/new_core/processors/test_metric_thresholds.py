"""
Unit tests for the threshold exceedance count feature in `metric_calc` processor.

Tests cover threshold configuration, direction (above/below), duration filtering,
period grouping, NaN handling, and execute() dispatch.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY
from climakitae.new_core.processors.metric_calc import MetricCalc


class TestMetricCalcThresholds:
    """Unit tests for the threshold exceedance count feature."""

    @pytest.fixture
    def daily_da(self):
        """DataArray with daily time steps spanning two calendar years."""
        # 2020 (leap, 366 days) + 2021 (365 days) = 731 days total
        dates = pd.date_range("2020-01-01", periods=731, freq="D")
        values = np.zeros(731, dtype=float)
        # Set 3 exceedances in 2020 and 2 in 2021
        values[:3] = 100.0
        values[366:368] = 100.0
        return xr.DataArray(
            values, dims=["time"], coords={"time": dates}, name="tasmax"
        )

    @pytest.fixture
    def single_year_da(self):
        """DataArray with 10 daily values in a single year for simple counting."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        values = np.arange(10, dtype=float)  # 0, 1, ..., 9
        return xr.DataArray(
            values, dims=["time"], coords={"time": dates}, name="tasmax"
        )

    def test_init_sets_thresholds(self):
        """MetricCalc stores normalised threshold config after init."""
        processor = MetricCalc(
            {
                "thresholds": {
                    "threshold_value": 305.0,
                    "threshold_direction": "above",
                    "period": (1, "year"),
                }
            }
        )
        assert processor.thresholds["threshold_value"] == 305.0
        assert processor.thresholds["threshold_direction"] == "above"
        assert processor.thresholds["period"] == (1, "year")

    def test_missing_direction_raises(self):
        """Missing threshold_direction should raise ValueError."""
        with pytest.raises(ValueError, match="threshold_direction"):
            MetricCalc({"thresholds": {"threshold_value": 0.0}})

    def test_missing_threshold_value_raises(self):
        """Missing threshold_value should raise ValueError."""
        with pytest.raises(ValueError, match="threshold_value"):
            MetricCalc({"thresholds": {"threshold_direction": "above"}})

    def test_invalid_direction_raises(self):
        """Invalid threshold_direction should raise ValueError."""
        with pytest.raises(ValueError, match="threshold_direction"):
            MetricCalc(
                {
                    "thresholds": {
                        "threshold_value": 5.0,
                        "threshold_direction": "sideways",
                    }
                }
            )

    def test_both_thresholds_and_one_in_x_raises(self):
        """Setting both thresholds and one_in_x should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot set both"):
            MetricCalc(
                {
                    "thresholds": {
                        "threshold_value": 5.0,
                        "threshold_direction": "above",
                    },
                    "one_in_x": {"return_periods": [10]},
                }
            )

    def test_both_thresholds_and_metric_raises(self):
        """Setting both thresholds and metric should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot set both"):
            MetricCalc(
                {
                    "thresholds": {
                        "threshold_value": 5.0,
                        "threshold_direction": "above",
                    },
                    "metric": "mean",
                }
            )

    def test_both_thresholds_and_percentiles_raises(self):
        """Setting both thresholds and percentiles should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot set both"):
            MetricCalc(
                {
                    "thresholds": {
                        "threshold_value": 5.0,
                        "threshold_direction": "above",
                    },
                    "percentiles": [10, 50, 90],
                }
            )

    def test_above_threshold_count_dataarray(self, single_year_da):
        """Count timesteps above threshold for a DataArray."""
        processor = MetricCalc(
            {"thresholds": {"threshold_value": 5.0, "threshold_direction": "above"}}
        )
        result = processor._calculate_threshold_single(single_year_da)

        # values 0-9: those > 5 are 6,7,8,9 → count = 4
        assert isinstance(result, xr.Dataset)
        assert "tasmax" in result.data_vars
        assert int(result["tasmax"].sum()) == 4

    def test_below_threshold_count_dataarray(self, single_year_da):
        """Count timesteps below threshold for a DataArray."""
        processor = MetricCalc(
            {"thresholds": {"threshold_value": 3.0, "threshold_direction": "below"}}
        )
        result = processor._calculate_threshold_single(single_year_da)

        # values 0-9: those < 3 are 0,1,2 → count = 3
        assert int(result["tasmax"].sum()) == 3

    def test_unnamed_dataarray_uses_default_name(self, single_year_da):
        """Unnamed DataArray should produce a variable named 'exceedance_count'."""
        da_unnamed = single_year_da.rename(None)
        processor = MetricCalc(
            {"thresholds": {"threshold_value": 5.0, "threshold_direction": "above"}}
        )
        result = processor._calculate_threshold_single(da_unnamed)
        assert "exceedance_count" in result.data_vars

    def test_threshold_count_dataset(self, single_year_da):
        """Count exceedances for each variable in a Dataset."""
        ds = xr.Dataset({"tasmax": single_year_da, "tasmin": single_year_da * 0.5})
        processor = MetricCalc(
            {"thresholds": {"threshold_value": 3.0, "threshold_direction": "above"}}
        )
        result = processor._calculate_threshold_single(ds)

        assert isinstance(result, xr.Dataset)
        # tasmax > 3: values 4,5,6,7,8,9 → 6
        assert int(result["tasmax"].sum()) == 6
        # tasmin (0-4.5 in 0.5 steps): > 3 are 3.5, 4.0, 4.5 → 3
        assert int(result["tasmin"].sum()) == 3

    def test_counts_per_year(self, daily_da):
        """Counts are grouped by calendar year."""
        processor = MetricCalc(
            {"thresholds": {"threshold_value": 50.0, "threshold_direction": "above"}}
        )
        result = processor._calculate_threshold_single(daily_da)

        counts = result["tasmax"].values
        # 2020: first 3 days = 100 → 3 exceedances
        # 2021: days 366-367 = 100 → 2 exceedances
        assert counts[0] == 3
        assert counts[1] == 2

    def test_duration_filter(self):
        """duration=(2, 'day') requires 2 consecutive exceedances."""
        # mask will be [0,1,1,0,0,0,0] → rolling(2).min = [NaN,0,1,0,0,0,0] → sum=1
        dates = pd.date_range("2020-01-01", periods=7, freq="D")
        values = np.array([1.0, 6.0, 6.0, 1.0, 1.0, 1.0, 1.0])
        da = xr.DataArray(values, dims=["time"], coords={"time": dates}, name="tasmax")

        processor = MetricCalc(
            {
                "thresholds": {
                    "threshold_value": 5.0,
                    "threshold_direction": "above",
                    "duration": (2, "day"),
                }
            }
        )
        result = processor._calculate_threshold_single(da)
        assert int(result["tasmax"].sum()) == 1

    def test_duration_unit_converted_to_timesteps(self):
        """duration=(2, 'day') on hourly data should roll 48 steps, not 2."""
        dates = pd.date_range("2020-01-01", periods=72, freq="h")
        # exceedances at hours 24-47 (24 consecutive hours = 1 day)
        values = np.where((dates.hour >= 0) & (dates.dayofyear == 2), 10.0, 1.0)
        da = xr.DataArray(values, dims=["time"], coords={"time": dates}, name="tasmax")

        processor = MetricCalc(
            {
                "thresholds": {
                    "threshold_value": 5.0,
                    "threshold_direction": "above",
                    "duration": (2, "day"),
                }
            }
        )
        result = processor._calculate_threshold_single(da)
        # 24 consecutive exceedances = 1 day, duration requires 2 days → 0 events
        assert int(result["tasmax"].sum()) == 0

    def test_all_nan_period_returns_nan(self):
        """A period with all-NaN data returns NaN, not 0."""
        dates = pd.date_range("2020-01-01", periods=365, freq="D")
        values = np.full(365, np.nan)
        da = xr.DataArray(values, dims=["time"], coords={"time": dates}, name="tasmax")

        processor = MetricCalc(
            {"thresholds": {"threshold_value": 5.0, "threshold_direction": "above"}}
        )
        result = processor._calculate_threshold_single(da)
        assert np.isnan(result["tasmax"].values).all()

    @pytest.mark.parametrize("period", [(1, "week"), (1, "day"), (1, "hour")])
    def test_invalid_period_raises(self, period):
        """Only 'year' and 'month' are valid period units."""
        with pytest.raises(ValueError, match="period"):
            MetricCalc(
                {
                    "thresholds": {
                        "threshold_value": 5.0,
                        "threshold_direction": "above",
                        "period": period,
                    }
                }
            )

    def test_nan_in_data_not_counted(self):
        """NaN values in the data are preserved as NaN in the output mask."""
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        values = np.array([np.nan, 10.0, 10.0, np.nan, 1.0])
        da = xr.DataArray(values, dims=["time"], coords={"time": dates}, name="tasmax")

        processor = MetricCalc(
            {"thresholds": {"threshold_value": 5.0, "threshold_direction": "above"}}
        )
        result = processor._calculate_threshold_single(da)
        # Only the two 10.0 values should be counted; NaN positions must not count
        assert int(result["tasmax"].sum()) == 2

    def test_execute_routes_to_threshold(self, single_year_da):
        """execute() dispatches to _calculate_threshold_single when thresholds is set."""
        processor = MetricCalc(
            {"thresholds": {"threshold_value": 5.0, "threshold_direction": "above"}}
        )
        result = processor.execute(single_year_da, context={})
        assert isinstance(result, xr.Dataset)
        assert int(result["tasmax"].sum()) == 4

    def test_update_context_threshold(self):
        """update_context records threshold info in the context."""
        processor = MetricCalc(
            {
                "thresholds": {
                    "threshold_value": 305.0,
                    "threshold_direction": "above",
                    "period": (1, "year"),
                }
            }
        )
        context: Dict[str, Any] = {}
        processor.update_context(context)
        description = context[_NEW_ATTRS_KEY]["metric_calc"]
        assert "305.0" in description
        assert "above" in description
