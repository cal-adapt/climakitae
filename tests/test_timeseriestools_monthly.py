"""
This script contains tests on various Timeseries Tools options using monthly 
data. For now, the tests only test that the various parameter combinations can 
run through `transform_data` without error, and that the data has been transformed
(not equal to original values), but does not test for exact expected values.
"""

from climakitae import timeseriestools as tst
import datetime as dt
import pytest
import os
import xarray as xr

#-------- Read in the test dataset and return a TimeSeriesParams object -------

@pytest.fixture
def test_TSP(rootdir):
    # This data is generated in "create_timeseries_test_data.py"
    test_filename= "test_data/timeseries_data_T2_2014_2016_monthly_45km.nc"
    test_filepath = os.path.join(rootdir, test_filename)
    test_data = xr.open_dataset(test_filepath).T2
    ts = tst.Timeseries(test_data) # make Timeseries object
    return ts.choices # return the underlying TimeSeriesParams object for testing

#------------- Test monthly-weighted running mean without anomaly -------------

def test_monthly_smoothing(test_TSP):
    # Specify Params options
    test_TSP.smoothing = 'running mean'
    test_TSP.num_timesteps = 3
    test_TSP.anomaly = False 

    # Transform data and test
    result = test_TSP.transform_data() # transform_data calls _running_mean()
    assert (result == test_TSP.data).sum().values.item() == 0

#------------- Test monthly weighted anomaly ----------------------------------

def test_monthly_anomaly(test_TSP):
    # Specify Params options
    test_TSP.anomaly = True
    test_TSP.reference_range = (dt.datetime(2014, 1, 1), dt.datetime(2014, 12, 31)) 

    # Transform data and test
    result = test_TSP.transform_data()
    assert (result == test_TSP.data).sum().values.item() == 0

#------------- Test anomaly and smoothing together ----------------------------

def test_monthly_anomaly_and_smoothing(test_TSP):
    # Specify Params options
    test_TSP.smoothing = 'running mean'
    test_TSP.num_timesteps = 3
    test_TSP.anomaly = True
    test_TSP.reference_range = (dt.datetime(2014, 1, 1), dt.datetime(2014, 12, 31)) 

    # Transform data and test
    result = test_TSP.transform_data()
    assert (result == test_TSP.data).sum().values.item() == 0

#------------- Test seasonal cycle removal w/ and w/o smoothing ---------------

def test_seasonal(test_TSP):
    # Specify Params options
    test_TSP.anomaly = False
    test_TSP.remove_seasonal_cycle = True 

    # Transform data and test
    result = test_TSP.transform_data()
    assert (result == test_TSP.data).sum().values.item() == 0

def test_seasonal_and_smoothing(test_TSP):
    # Specify Params options
    test_TSP.smoothing = 'running mean'
    test_TSP.num_timesteps = 3
    test_TSP.anomaly = False
    test_TSP.remove_seasonal_cycle = True 

    # Transform data and test
    result = test_TSP.transform_data()
    assert (result == test_TSP.data).sum().values.item() == 0

#------------- Test extremes options ------------------------------------------

def test_extremes_smoothing(test_TSP):
    # Specify Params options
    test_TSP.anomaly = False 
    test_TSP.smoothing = 'running mean'
    test_TSP.num_timesteps = 3
    test_TSP.extremes = "min"
    test_TSP.resample_window = 2

    # Transform data and test
    result = test_TSP.transform_data() 
    assert (result == test_TSP.data).sum().values.item() == 0

def test_extremes_min(test_TSP):
    # Specify Params options
    test_TSP.anomaly = False 
    test_TSP.extremes = "min"
    test_TSP.resample_window = 2 

    # Transform data and test
    result = test_TSP.transform_data() 
    assert (result == test_TSP.data).sum().values.item() == 0

def test_extremes_percentile(test_TSP):
    # Specify Params options
    test_TSP.anomaly = False 
    test_TSP.extremes = "percentile"
    test_TSP.resample_window = 2
    test_TSP.percentile = 0.95

    # Transform data and test
    result = test_TSP.transform_data() 
    assert (result == test_TSP.data).sum().values.item() == 0