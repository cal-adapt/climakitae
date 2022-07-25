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
    # return xr.open_dataarray(test_filepath).T2
    return xr.open_dataarray(test_filepath)

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


#------------- Test helper functions for plotting -----------------------------

# example name 1: 
def test_name1():
    ex1 = xr.DataArray(attrs = {
        "group" : None,
        "frequency" : "hourly",
        "period_length" : "1year"
    })
    name1 = threshold_tools._exceedance_count_name(ex1)
    assert name1 == "Number of hours per 1year"

# example name 2: 
def test_name2():
    ex2 = xr.DataArray(attrs = {
        "group" : "1day",
        "frequency" : "hourly",
        "period_length" : "1year"
    })
    name2 = threshold_tools._exceedance_count_name(ex2)
    assert name2 == "Number of days per 1year"

# def test_title1():




# test multiple scenario / simulation options with montlhy data (smaller)

# test smoothing options with monthly data

# test duration options-- probably has to be with hourly data