"""
Add test coverage for code that is not already tested in
test_thresholds_returns.py or test_threshold_exceedence.py.
"""

import os

import numpy as np
import pandas as pd
import pytest
import scipy
import xarray as xr

from climakitae.explore import threshold_tools
from climakitae.explore.threshold_tools import (
    _calc_average_ess_gridded_data,
    _calc_average_ess_timeseries_data,
    _exceedance_count_name,
    _get_distr_func,
    _get_exceedance_events,
    _get_fitted_distr,
)


class TestThresholdTools:
    """Test effective sample size and hidden functions in
    threshold_tools.
    """

    def test_calculate_ess(self):
        """Test that effective sample size runs without validating returned values."""
        test = xr.DataArray(data=np.arange(0, 100, 1))
        result = threshold_tools.calculate_ess(test)
        assert isinstance(result, xr.DataArray)
        assert not np.isnan(result.data)
        assert result.name == "ess"

        result = threshold_tools.calculate_ess(test, 3)
        assert isinstance(result, xr.DataArray)
        assert not np.isnan(result.data)
        assert result.name == "ess"

    def test__calc_average_ess_gridded_data(self):
        """Test that effective sample size runs without validating returned values."""
        test_data = test_data = np.random.rand(10, 10, 365 * 3) * 100
        test = xr.DataArray(
            data=test_data,
            coords={
                "x": np.arange(10, 20, 1),
                "y": np.arange(10, 20, 1),
                "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
            },
        )
        result = _calc_average_ess_gridded_data(test, 1)
        assert not np.isnan(result)
        assert isinstance(result, float)

    def test__calc_average_ess_timeseries_data(self):
        """Test that effective sample size runs without validating returned values."""
        test_data = test_data = np.random.rand(365 * 3) * 100
        test = xr.DataArray(
            data=test_data,
            coords={
                "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
            },
        )
        result = _calc_average_ess_timeseries_data(test, 1)
        assert not np.isnan(result)
        assert isinstance(result, float)

    @pytest.mark.advanced
    def test__get_fitted_distr(self):
        """Check that fitted distr returns correct parameter set for gev."""
        test_data = test_data = np.random.rand(10, 10, 365 * 3) * 100
        test = xr.DataArray(
            data=test_data,
            coords={
                "x": np.arange(10, 20, 1),
                "y": np.arange(10, 20, 1),
                "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
            },
        )
        result = _get_fitted_distr(test, "gev", scipy.stats.genextreme)
        assert isinstance(result, tuple)
        assert isinstance(result[0], dict)
        assert isinstance(
            result[1], scipy.stats._distn_infrastructure.rv_continuous_frozen
        )
        for param in ["c", "loc", "scale"]:
            assert param in result[0]
            assert isinstance(result[0][param], float)

    def test__get_distr_func(self):
        """Check that get_distr_func returns correct distribution for input value."""
        diststr = ["gev", "gumbel", "weibull", "pearson3", "genpareto", "gamma"]
        diststat = [
            scipy.stats.genextreme,
            scipy.stats.gumbel_r,
            scipy.stats.weibull_min,
            scipy.stats.pearson3,
            scipy.stats.genpareto,
            scipy.stats.gamma,
        ]

        for dstr, dstat in zip(diststr, diststat):
            assert _get_distr_func(dstr) == dstat

        with pytest.raises(ValueError):
            _get_distr_func("")

    def test__exceedance_count_name(self):
        """Test that correct string is returned for various parameter values."""
        item1 = 4
        item2 = "day"
        test = xr.DataArray()
        test.attrs["duration2"] = (item1, item2)
        result = _exceedance_count_name(test)
        assert result == (f"Number of {item1}-{item2} events")

        test = xr.DataArray()
        test.attrs["duration2"] = None
        test.attrs["group"] = (item1, item2)
        result = _exceedance_count_name(test)
        assert result == (f"Number of {item1}-{item2} events")

        test = xr.DataArray()
        test.attrs["duration2"] = None
        test.attrs["group"] = None
        tscales = {"hourly": "hours", "monthly": "months", "daily": "days"}
        for item in tscales:
            test.attrs["frequency"] = item
            result = _exceedance_count_name(test)
            assert result == (f"Number of {tscales[item]}")

    def test__get_exceedance_events(self):
        """Check exceedence for minimum case and for error cases."""
        test_data = test_data = np.random.rand(10, 10, 365 * 3) * 100
        test_data[0, 0, 0] = 1  # make sure at least one instance is <10
        test = xr.DataArray(
            data=test_data,
            coords={
                "x": np.arange(10, 20, 1),
                "y": np.arange(10, 20, 1),
                "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
            },
        )
        result = _get_exceedance_events(test, 10, threshold_direction="below")
        assert result.sum().data == xr.where(test < 10, 1, 0).sum().data

        with pytest.raises(ValueError):
            _get_exceedance_events(test, 10, threshold_direction="belw")

        with pytest.raises(ValueError):
            _get_exceedance_events(test, 90, duration1=(3, "month"))

    def test_get_return_value_keeps_one_in_x_shape_with_invalid_series(self):
        """Test return-value output keeps consistent one_in_x shape even with bad series."""
        rng = np.random.default_rng(42)
        time = pd.date_range(start="1980-01-01", periods=30, freq="YS")
        valid_series = rng.gamma(shape=2.0, scale=1.0, size=len(time))
        invalid_series = np.full(len(time), np.nan)

        bms = xr.DataArray(
            np.stack([valid_series, invalid_series], axis=1),
            dims=["time", "simulation"],
            coords={"time": time, "simulation": ["good", "bad"]},
            name="bms",
        )

        periods = [2, 5, 10, 25, 50, 100, 150, 200]
        result = threshold_tools.get_return_value(
            bms,
            return_period=periods,
            distr="gev",
            bootstrap_runs=2,
            multiple_points=False,
        )

        assert result["return_value"].sizes["one_in_x"] == len(periods)
        assert result["conf_int_lower_limit"].sizes["one_in_x"] == len(periods)
        assert result["conf_int_upper_limit"].sizes["one_in_x"] == len(periods)
        assert result["return_value"].sel(simulation="bad").isnull().all()

    def test_get_ks_stat_handles_non_finite_values(self):
        """Test KS-stat gracefully handles NaN/Inf values without raising."""
        bms = xr.DataArray(
            np.array([1.0, 2.0, np.nan, np.inf, 3.0, 4.0, 5.0]), dims=["time"]
        )

        result = threshold_tools.get_ks_stat(bms, distr="gev", multiple_points=False)

        assert isinstance(result, xr.Dataset)
        assert "d_statistic" in result
        assert "p_value" in result

        all_bad = xr.DataArray(np.array([np.nan, np.inf, -np.inf]), dims=["time"])
        all_bad_result = threshold_tools.get_ks_stat(
            all_bad, distr="gev", multiple_points=False
        )
        assert np.isnan(all_bad_result["p_value"].values[()])


@pytest.mark.advanced
class TestKsStat:
    """Test the ks_stat function. These are longer running tests
    which have been separated to run in the advanced tier.
    """

    def test_ks_stat(self):
        """Verify that ks_stat runs for various distribution cases."""
        test_data = test_data = np.random.rand(10, 10, 365 * 3) * 100
        test = xr.DataArray(
            data=test_data,
            coords={
                "x": np.arange(10, 20, 1),
                "y": np.arange(10, 20, 1),
                "time": pd.date_range(start="2001-01-01", end="2003-12-31"),
            },
        )
        result = threshold_tools.get_ks_stat(test)
        assert isinstance(result, xr.Dataset)
        assert result.distribution == "gev"
        assert "d_statistic" in result
        assert "p_value" in result

        # Check that other distributions run
        result = threshold_tools.get_ks_stat(test, "gumbel")
        assert isinstance(result, xr.Dataset)

        result = threshold_tools.get_ks_stat(test, "weibull")
        assert isinstance(result, xr.Dataset)

        result = threshold_tools.get_ks_stat(test, "pearson3")
        assert isinstance(result, xr.Dataset)

        result = threshold_tools.get_ks_stat(test, "genpareto")
        assert isinstance(result, xr.Dataset)

        result = threshold_tools.get_ks_stat(test, "gamma")
        assert isinstance(result, xr.Dataset)
