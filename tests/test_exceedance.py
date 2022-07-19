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
    """ Small hourly temperature data set for one scenario and one simulation"""
    test_filename= "test_data/threshold_data_T2_2050_2060_hourly_45km.nc"
    test_filepath = os.path.join(rootdir, test_filename)
    test_data = xr.open_dataarray(test_filepath)
    return test_data.isel(scenario=0, simulation=0)

#------------- Tests with hourly data -----------------------------------------

# example 1: count number of hours in each year exceeding the threshold
def test_hourly_ex1(T2_hourly):
    exc_counts = threshold_tools.get_exceedance_count(T2_hourly, threshold_value=305, period_length="1year")
    assert (exc_counts >= 0).all()      # test no negative values
    assert "year" in exc_counts.coords  # test correct time transformation occured (datetime --> year)

# exmample 2: count number of days in each year that have at least one hour exceeding the threshold
def test_hourly_ex2(T2_hourly):
    exc_counts = threshold_tools.get_exceedance_count(T2_hourly, threshold_value=305, period_length="1year", groupby="1day")
    assert (exc_counts >= 0).all()      # test no negative values
    assert "year" in exc_counts.coords  # test correct time transformation occured (datetime --> year)

# exmample 3: count number of 3-day events in each year that continously exceed the threshold
def test_hourly_ex3(T2_hourly):
    exc_counts = threshold_tools.get_exceedance_count(T2_hourly, threshold_value=305, period_length="1year", duration="3day")
    assert (exc_counts >= 0).all()      # test no negative values
    assert "year" in exc_counts.coords  # test correct time transformation occured (datetime --> year)

# exmample 4: count number of 3-day events in each year that exceed the threshold once each day
def test_hourly_ex3(T2_hourly):
    exc_counts = threshold_tools.get_exceedance_count(T2_hourly, threshold_value=305, period_length="1year", duration="3day", groupby="1day")
    assert (exc_counts >= 0).all()      # test no negative values
    assert "year" in exc_counts.coords  # test correct time transformation occured (datetime --> year)


# test multiple scenario / simulation options with montlhy data (smaller)

# test smoothing options with monthly data

# test duration options-- probably has to be with hourly data