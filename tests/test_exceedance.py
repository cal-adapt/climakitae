import numpy as np
import os
import pytest
import xarray as xr

from climakitae import threshold_tools


#------------- Data for testing -----------------------------------------------

@pytest.fixture
def T2_monthly(test_data):
    """ Monthly temperature data for one scenario and one simulation"""
    return test_data["T2"].isel(scenario=0, simulation=0)

@pytest.fixture
def T2_hourly(rootdir):
    """ Small hourly temperature data set for one location, one scenario, and one simulation"""
    test_filename= "test_data/threshold_data_T2_2050_2060_hourly_45km.nc"
    test_filepath = os.path.join(rootdir, test_filename)
    test_data = xr.open_dataarray(test_filepath)
    return test_data.isel(scenario=0, simulation=0)

#------------- Tests

def test_hourly(T2_hourly):
    exc_counts = threshold_tools.exceedance(T2_hourly, threshold_value=305, period_length="1year")
    assert (exc_counts >= 0).all()
    assert "year" in exc_counts.coords



# test multiple scenario / simulation options with montlhy data (smaller)

# test smoothing options with monthly data

# test duration options-- probably has to be with hourly data