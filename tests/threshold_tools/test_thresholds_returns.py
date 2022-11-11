# Test the available extreme value theory functions for calcualting 
# return values, periods, and probabilities

import numpy as np
import os
import pandas as pd
import pytest
import xarray as xr

from climakitae import threshold_tools

#------------- Data for testing -----------------------------------------------

# Generate an annual maximum series (ams) datarray for testing 
@pytest.fixture
def T2_ams(rootdir):
    # This data is generated in "create_test_data.py"
    test_filename= "test_data/timeseries_data_T2_2014_2016_monthly_45km.nc"
    test_filepath = os.path.join(rootdir, test_filename)
    test_data = xr.open_dataset(test_filepath).T2
    return threshold_tools.get_ams(test_data).isel(simulation = 0)

#------------- Test  -----------------------------------------------

# Test Return Values
def test_return_value(T2_ams):
    rvs = threshold_tools.get_return_value(
        T2_ams, return_period=10, distr='gev', bootstrap_runs=1
    )
    assert length(rvs['return_value'].shape) == 2

# Test invalid distribution argument for Return Values
def test_return_value_invalid_distr(T2_ams):
    with pytest.raises(ValueError, match="invalid distr type"):
            rvs = threshold_tools.get_return_value(
                T2_ams, return_period=10, distr='foo', bootstrap_runs=1
            )   