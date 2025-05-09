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

    def test_calculate_ess(self):
        # Not validating the result, just checking this runs
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

    def test__get_fitted_distr(self):
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
        diststr = ["gev", "gumbel", "weibull", "pearson3", "genpareto", "gamma"]
        diststat = [
            scipy.stats.genextreme,
            scipy.stats.gumbel_r,
            scipy.stats.weibull_min,
            scipy.stats.pearson3,
            scipy.stats.genpareto,
            scipy.stats.gamma,
        ]
        for ind in range(0, len(diststr)):
            assert _get_distr_func(diststr[ind]) == diststat[ind]

        with pytest.raises(ValueError):
            _get_distr_func("")

    def test__exceedance_count_name(self):
        item1 = 4
        item2 = "day"
        test = xr.DataArray()
        test.attrs["duration2"] = (item1, item2)
        result = _exceedance_count_name(test)
        assert result == "Number of {0}-{1} events".format(item1, item2)

        test = xr.DataArray()
        test.attrs["duration2"] = None
        test.attrs["group"] = (item1, item2)
        result = _exceedance_count_name(test)
        assert result == "Number of {0}-{1} events".format(item1, item2)

        test = xr.DataArray()
        test.attrs["duration2"] = None
        test.attrs["group"] = None
        tscales = {"hourly": "hours", "monthly": "months", "daily": "days"}
        for item in tscales:
            test.attrs["frequency"] = item
            result = _exceedance_count_name(test)
            assert result == "Number of {0}".format(tscales[item])

    def test__get_exceedance_events(self):
        test_data = test_data = np.random.rand(10, 10, 365 * 3) * 100
        test_data[0, 0, 0] = 1  # making sure something is <10
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


@pytest.mark.advanced
class TestKsStat:
    def test_ks_stat(self):
        # Not verifying results, just checking that it runs
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
