"""
Test the available extreme value theory functions for calculating return
values and periods. These tests do not check the correctness of the
calculations; they just ensure that the functions run without error, or raise
the expected error messages for invalid argument specifications.
"""

import os

import numpy as np
import pytest
import xarray as xr

from climakitae.explore import threshold_tools

# ------------- Data for testing -----------------------------------------------


@pytest.fixture
def T2_ams(rootdir: str) -> xr.DataArray:
    """Generate an annual maximum series (ams) datarray for testing."""
    # This data is generated in "create_test_data.py"
    test_filename = "test_data/timeseries_data_T2_2014_2016_monthly_45km.nc"
    test_filepath = os.path.join(rootdir, test_filename)
    test_data = (
        xr.open_dataset(test_filepath)
        .T2.mean(dim=("x", "y"), skipna=True)
        .isel(scenario=0, simulation=0)
    )
    return threshold_tools.get_block_maxima(test_data, block_size=1, check_ess=False)


# ------------- Test return values and periods ----------------------------------


@pytest.mark.advanced
def test_return_value(T2_ams: xr.DataArray):
    """Test Return Values."""
    rvs = threshold_tools.get_return_value(
        T2_ams, return_period=10, distr="gev", bootstrap_runs=1, multiple_points=False
    )
    assert not np.isnan(rvs["return_value"].values[()])


@pytest.mark.advanced
def test_return_value_dropna(T2_ams: xr.DataArray):
    """Test Return Values with dropna_time option."""
    T2_ams.data[2] = np.nan
    rvs = threshold_tools.get_return_value(
        T2_ams,
        return_period=10,
        distr="gev",
        bootstrap_runs=1,
        multiple_points=False,
        dropna_time=True,
    )
    assert not np.isnan(rvs["return_value"].values[()])


@pytest.mark.advanced
def test_return_value_invalid_distr(T2_ams: xr.DataArray):
    """# Test invalid distribution argument for Return Values."""
    with pytest.raises(ValueError, match="invalid distribution type"):
        rvs = threshold_tools.get_return_value(
            T2_ams,
            return_period=10,
            distr="foo",
            bootstrap_runs=1,
            multiple_points=False,
        )


@pytest.mark.advanced
def test_return_period(T2_ams: xr.DataArray):
    """Test Return Periods."""
    rvs = threshold_tools.get_return_period(
        T2_ams,
        return_value=290,
        distr="gumbel",
        bootstrap_runs=1,
        multiple_points=False,
    )
    assert not np.isnan(rvs["return_period"].values[()])


@pytest.mark.advanced
def test_return_period_dropna(T2_ams: xr.DataArray):
    """Test Return Periods."""
    T2_ams.data[2] = np.nan
    rvs = threshold_tools.get_return_period(
        T2_ams,
        return_value=290,
        distr="gumbel",
        bootstrap_runs=1,
        multiple_points=False,
        dropna_time=True,
    )
    assert not np.isnan(rvs["return_period"].values[()])


@pytest.mark.advanced
def test_return_period_invalid_distr(T2_ams: xr.DataArray):
    """Test invalid distribution argument for Return Periods."""
    with pytest.raises(ValueError, match="invalid distribution type"):
        _ = threshold_tools.get_return_period(
            T2_ams,
            return_value=290,
            distr="foo",
            bootstrap_runs=1,
            multiple_points=False,
        )


@pytest.mark.advanced
def test_return_values_block_size(T2_ams: xr.DataArray):
    """Test return values for different block sizes."""
    rvs1 = threshold_tools.get_return_value(
        T2_ams, return_period=10, distr="gev", bootstrap_runs=1, multiple_points=False
    )
    # set different block size attribute to test that the calculation is handled differently:
    T2_ams.attrs["block size"] = "2 year"
    rvs2 = threshold_tools.get_return_value(
        T2_ams, return_period=10, distr="gev", bootstrap_runs=1, multiple_points=False
    )
    # test that the return values from longer block sizes should be smaller:
    assert rvs1["return_value"].values[()] >= rvs2["return_value"].values[()]


@pytest.mark.advanced
def test_return_periods_block_size(T2_ams: xr.DataArray):
    """Test return periods for different block sizes."""
    rps1 = threshold_tools.get_return_period(
        T2_ams, return_value=290, distr="gev", bootstrap_runs=1, multiple_points=False
    )
    # set different block size attribute to test that the calculation is handled differently:
    T2_ams.attrs["block size"] = "2 year"
    rps2 = threshold_tools.get_return_period(
        T2_ams, return_value=290, distr="gev", bootstrap_runs=1, multiple_points=False
    )
    # test that the return periods from longer block sizes should be larger:
    assert rps1["return_period"].values[()] <= rps2["return_period"].values[()]


@pytest.mark.advanced
def test_return_probs_block_size(T2_ams: xr.DataArray):
    """Test return probabilities for different block sizes."""
    rps1 = threshold_tools.get_return_prob(
        T2_ams, threshold=290, distr="gev", bootstrap_runs=1, multiple_points=False
    )
    # set different block size attribute to test that the calculation is handled differently:
    T2_ams.attrs["block size"] = "2 year"
    rps2 = threshold_tools.get_return_prob(
        T2_ams, threshold=290, distr="gev", bootstrap_runs=1, multiple_points=False
    )
    # test that the return probs from longer block sizes should be smaller:
    assert rps1["return_prob"].values[()] >= rps2["return_prob"].values[()]


# -------------- Test AMS block maxima calculations for complex extreme events


@pytest.mark.advanced
def test_ams_ex1(T2_hourly: xr.DataArray):
    """Test that the AMS (block maxima) for a 3-day grouped event are lower than
    the simple AMS (single hottest value in each year).
    """
    T2_hourly = T2_hourly.isel(scenario=0, simulation=0)
    ams = threshold_tools.get_block_maxima(T2_hourly, check_ess=False)
    ams_3d = threshold_tools.get_block_maxima(
        T2_hourly, groupby=(1, "day"), grouped_duration=(3, "day"), check_ess=False
    )
    assert (ams >= ams_3d).all()


@pytest.mark.advanced
def test_ams_ex2(T2_hourly: xr.DataArray):
    """Test that the AMS (block maxima) for a 3-day continous event are lower than
    the AMS for a grouped 3-day event.
    """
    T2_hourly = T2_hourly.isel(scenario=0, simulation=0)
    ams_3d = threshold_tools.get_block_maxima(
        T2_hourly, groupby=(1, "day"), grouped_duration=(3, "day"), check_ess=False
    )
    ams_72h = threshold_tools.get_block_maxima(
        T2_hourly, duration=(72, "hour"), check_ess=False
    )
    assert (ams_3d >= ams_72h).all()


@pytest.mark.advanced
def test_ams_ex3(T2_hourly: xr.DataArray):
    """Test that the AMS (block maxima) for a 4-hour per day for 3 days are lower
    than the AMS for a grouped 3-day event.
    """
    T2_hourly = T2_hourly.isel(scenario=0, simulation=0)
    ams_3d = threshold_tools.get_block_maxima(
        T2_hourly, groupby=(1, "day"), grouped_duration=(3, "day"), check_ess=False
    )
    ams_3d_4h = threshold_tools.get_block_maxima(
        T2_hourly,
        duration=(4, "hour"),
        groupby=(1, "day"),
        grouped_duration=(3, "day"),
        check_ess=False,
    )
    assert (ams_3d >= ams_3d_4h).all()


def test_ams_ex4(T2_hourly: xr.DataArray):
    """Test that the AMS (block maxima) for a 4-hour per day for 3 days are greater
    than the AMS for a grouped 3-day event.
    """
    T2_hourly = T2_hourly.isel(scenario=0, simulation=0)
    ams_3d = threshold_tools.get_block_maxima(
        T2_hourly,
        extremes_type="min",
        groupby=(1, "day"),
        grouped_duration=(3, "day"),
        check_ess=False,
    )
    ams_3d_4h = threshold_tools.get_block_maxima(
        T2_hourly,
        extremes_type="min",
        duration=(4, "hour"),
        groupby=(1, "day"),
        grouped_duration=(3, "day"),
        check_ess=False,
    )
    assert (ams_3d <= ams_3d_4h).all()


def test_block_maxima_value_error(T2_hourly: xr.DataArray):
    """Test the configurations that should raise ValueError."""
    T2_hourly = T2_hourly.isel(scenario=0, simulation=0)
    with pytest.raises(ValueError):
        _ = threshold_tools.get_block_maxima(
            T2_hourly, extremes_type="mx", check_ess=False
        )

    with pytest.raises(ValueError):
        _ = threshold_tools.get_block_maxima(
            T2_hourly, duration=(4, "day"), check_ess=False
        )

    with pytest.raises(ValueError):
        _ = threshold_tools.get_block_maxima(
            T2_hourly, groupby=(4, "hr"), check_ess=False
        )

    with pytest.raises(ValueError):
        _ = threshold_tools.get_block_maxima(
            T2_hourly, grouped_duration=(3, "day"), check_ess=False
        )

    with pytest.raises(ValueError):
        _ = threshold_tools.get_block_maxima(
            T2_hourly,
            duration=(4, "month"),
            grouped_duration=(3, "day"),
            check_ess=False,
        )
