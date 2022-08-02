import numpy as np
import os
import pytest
import xarray as xr

from climakitae import threshold_tools

#------------- Data for testing -----------------------------------------------

@pytest.fixture
def T2_monthly(test_data):
    """ Monthly RAINC data for one scenario and one simulation 
    (pulled from the general test data set)"""
    return test_data["RAINC"].isel(scenario=0, simulation=0)

@pytest.fixture
def T2_hourly(rootdir):
    """ Small hourly temperature data set"""
    test_filename= "test_data/threshold_data_T2_2050_2051_hourly_45km.nc"
    test_filepath = os.path.join(rootdir, test_filename)
    return xr.open_dataset(test_filepath)["Air Temperature at 2m"]

#------------- Test kwarg compatibility and Exceptions ------------------------

# incompatible: cannot specify a 1-day groupy for monthly data
def test_error1(T2_monthly):
    with pytest.raises(ValueError, match="Incompatible `groupby` specification"):
        threshold_tools.get_exceedance_count(T2_monthly,
            threshold_value=305, groupby = (1, "day")
        ) 

# incompatible: cannot specify a 3-day duration if grouped by month
# But for now, `duration` not yet implemented
def test_error2(T2_hourly):
    # with pytest.raises(ValueError, match="Incompatible `groupby` and `duration` specification"):
    with pytest.raises(ValueError, match="Duration options not yet implemented"):
        threshold_tools.get_exceedance_count(T2_hourly,
            threshold_value=305, groupby = (1, "month"), duration = (3, "day")
        )

#------------- Tests with hourly data -----------------------------------------

# example 1: count number of hours in each year exceeding the threshold
def test_hourly_ex1(T2_hourly):
    exc_counts = threshold_tools.get_exceedance_count(T2_hourly, threshold_value=305, period=(1, "year"))
    assert (exc_counts >= 0).all()      # test no negative values
    assert "year" in exc_counts.coords  # test correct time transformation occured (datetime --> year)

# exmample 2: count number of days in each year that have at least one hour exceeding the threshold
def test_hourly_ex2(T2_hourly):
    exc_counts = threshold_tools.get_exceedance_count(T2_hourly, threshold_value=305, period=(1, "year"), groupby=(1, "day"))
    assert (exc_counts >= 0).all()      # test no negative values
    assert "year" in exc_counts.coords  # test correct time transformation occured (datetime --> year)

# exmample 3: count number of 3-day events in each year that continously exceed the threshold
def test_hourly_ex3(T2_hourly):
    exc_counts = threshold_tools.get_exceedance_count(T2_hourly, threshold_value=305, period=(1, "year"), duration=(3, "day"))
    assert (exc_counts >= 0).all()      # test no negative values
    assert "year" in exc_counts.coords  # test correct time transformation occured (datetime --> year)

# exmample 4: count number of 3-day events in each year that exceed the threshold once each day
def test_hourly_ex4(T2_hourly):
    exc_counts = threshold_tools.get_exceedance_count(T2_hourly, threshold_value=305, period=(1, "year"), duration=(3, "day"), groupby=(1, "day"))
    assert (exc_counts >= 0).all()      # test no negative values
    assert "year" in exc_counts.coords  # test correct time transformation occured (datetime --> year)


#------------- Test helper functions for plotting -----------------------------

# example name 1: Number of hours per 1 year
def test_name1():
    ex1 = xr.DataArray(attrs = {
        "group" : None,
        "frequency" : "hourly",
        "period" : (1, "year"),
        "duration" : None
    })
    name1 = threshold_tools._exceedance_count_name(ex1)
    assert name1 == "Number of hours per 1 year"

# example name 2: Number of days per 1 year
def test_name2():
    ex2 = xr.DataArray(attrs = {
        "group" : (1, "day"),
        "frequency" : "hourly",
        "period" : (1, "year"),
        "duration" : None
    })
    name2 = threshold_tools._exceedance_count_name(ex2)
    assert name2 == "Number of days per 1 year"

# example name 3: Number of 3-day events per 1 year
def test_name3():
    ex3 = xr.DataArray(attrs = {
        "group" : (1, "day"),
        "frequency" : "hourly",
        "period" : (1, "year"),
        "duration" : (3, "day")
    })
    name3 = threshold_tools._exceedance_count_name(ex3)
    assert name3 == "Number of 3-day events per 1 year"

# example title 1: 
def test_title1():
    ex1 = xr.DataArray(attrs = {
        "variable_name" : "Air Temperature at 2m",
        "threshold_direction" : "above",
        "threshold_value" : 299,
        "variable_units" : "K"
    })
    title1 = threshold_tools._exceedance_plot_title(ex1)
    assert title1 == "Air Temperature at 2m: events above 299K"




# test multiple scenario / simulation options with montlhy data (smaller)

# test smoothing options with monthly data

# test duration options-- probably has to be with hourly data