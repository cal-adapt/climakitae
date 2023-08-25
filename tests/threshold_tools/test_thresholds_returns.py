"""
Test the available extreme value theory functions for calculating return
values and periods. These tests do not check the correctness of the
calculations; they just ensure that the functions run without error, or raise
the expected error messages for invalid argument specifications.
"""

import os
import pytest
import xarray as xr

from climakitae.explore import threshold_tools

# ------------- Data for testing -----------------------------------------------


# Generate an annual maximum series (ams) datarray for testing
@pytest.fixture
def T2_ams(rootdir):
    # This data is generated in "create_test_data.py"
    test_filename = "test_data/timeseries_data_T2_2014_2016_monthly_45km.nc"
    test_filepath = os.path.join(rootdir, test_filename)
    test_data = xr.open_dataset(test_filepath).T2
    return threshold_tools.get_ams(test_data).isel(simulation=0)


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
