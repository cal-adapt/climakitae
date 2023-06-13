"""
Test the available extreme value theory functions for calculating return
values and periods. These tests do not check the correctness of the
calculations; they just ensure that the functions run without error, or raise
the expected error messages for invalid argument specifications.
"""

import os
import pytest
import xarray as xr

from climakitae import threshold_tools

# ------------- Data for testing -----------------------------------------------


# Generate an annual maximum series (ams) datarray for testing
@pytest.fixture
def T2_ams(rootdir):
    # This data is generated in "create_test_data.py"
    test_filename = "test_data/timeseries_data_T2_2014_2016_monthly_45km.nc"
    test_filepath = os.path.join(rootdir, test_filename)
    test_data = xr.open_dataset(test_filepath).T2
    return threshold_tools.get_block_maxima(test_data).isel(simulation=0)


# ------------- Test return values and periods ----------------------------------


# Test Return Values
def test_return_value(T2_ams):
    rvs = threshold_tools.get_return_value(
        T2_ams, return_period=10, distr="gev", bootstrap_runs=1
    )
    assert len(rvs["return_value"].shape) == 2


# Test invalid distribution argument for Return Values
def test_return_value_invalid_distr(T2_ams):
    with pytest.raises(ValueError, match="invalid distribution type"):
        rvs = threshold_tools.get_return_value(
            T2_ams, return_period=10, distr="foo", bootstrap_runs=1
        )


# Test Return Periods
def test_return_period(T2_ams):
    rvs = threshold_tools.get_return_period(
        T2_ams, return_value=290, distr="gumbel", bootstrap_runs=1
    )
    assert len(rvs["return_period"].shape) == 2


# Test invalid distribution argument for Return Periods
def test_return_period_invalid_distr(T2_ams):
    with pytest.raises(ValueError, match="invalid distribution type"):
        rvs = threshold_tools.get_return_period(
            T2_ams, return_value=290, distr="foo", bootstrap_runs=1
        )


#-------------- Test AMS block maxima calculations for complex extreme events


# Test that the AMS (block maxima) for a 3-day grouped event are lower than 
# the simple AMS (single hottest value in each year)
def test_ams_ex1(T2_hourly):
    ams = threshold_tools.get_block_maxima(T2_hourly)
    ams_3d = threshold_tools.get_block_maxima(T2_hourly, groupby=(1, 'day'), grouped_duration=(3, 'day'))
    assert (ams > ams_3d).all()


# Test that the AMS (block maxima) for a 3-day continous event are lower than 
# the AMS for a grouped 3-day event
def test_ams_ex2(T2_hourly):
    ams_3d = threshold_tools.get_block_maxima(T2_hourly, groupby=(1, 'day'), grouped_duration=(3, 'day'))
    ams_72h = threshold_tools.get_block_maxima(T2_hourly, duration=(72, 'hour'))
    assert (ams_3d > ams_72h).all()

# Test that the AMS (block maxima) for a 4-hour per day for 3 days are lower 
# than the AMS for a grouped 3-day event
def test_ams_ex3(T2_hourly):
    ams_3d = threshold_tools.get_block_maxima(T2_hourly, groupby=(1, 'day'), grouped_duration=(3, 'day'))
    ams_3d_4h = threshold_tools.get_block_maxima(T2_hourly, duration=(4, 'hour'), groupby=(1, 'day'), grouped_duration=(3, 'day'))
    assert (ams_3d > ams_3d_4h).all()
