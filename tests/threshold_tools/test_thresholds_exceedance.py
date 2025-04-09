import pytest
import pandas as pd
import xarray as xr

from climakitae.explore import threshold_tools

# ------------- Data for testing -----------------------------------------------


@pytest.fixture
def T2_monthly(test_data):
    """Monthly RAINC data for one scenario and one simulation
    (pulled from the general test data set)"""
    return test_data["RAINC"].isel(scenario=0, simulation=0)


# ------------- Test kwarg compatibility and Exceptions ------------------------


# incompatible: cannot specify a 1-day groupy for monthly data
def test_error1(test_data_2022_monthly_45km):
    with pytest.raises(ValueError, match="Incompatible `group` specification"):
        threshold_tools.get_exceedance_count(
            test_data_2022_monthly_45km, threshold_value=305, groupby=(1, "day")
        )


# incompatible: cannot specify a 3-day duration if grouped by month
# But for now, `duration` not yet implemented
def test_error2(T2_hourly):
    with pytest.raises(
        ValueError, match="Incompatible `group` and `duration2` specification"
    ):
        threshold_tools.get_exceedance_count(
            T2_hourly, threshold_value=305, groupby=(1, "month"), duration2=(3, "day")
        )


# ------------- Tests with hourly data -----------------------------------------


# example 1: count number of hours in each year exceeding the threshold
def test_hourly_ex1(T2_hourly):
    exc_counts = threshold_tools.get_exceedance_count(
        T2_hourly, threshold_value=305, period=(1, "year")
    )
    assert (exc_counts >= 0).all()  # test no negative values
    assert (
        len(exc_counts.time) == 2
    )  # test correct time transformation occured (collapsed to only 2 values, one for each year)


# exmample 2: count number of days in each year that have at least one hour exceeding the threshold
def test_hourly_ex2(T2_hourly):
    exc_counts = threshold_tools.get_exceedance_count(
        T2_hourly, threshold_value=305, period=(1, "year"), groupby=(1, "day")
    )
    assert (exc_counts >= 0).all()  # test no negative values
    assert (
        len(exc_counts.time) == 2
    )  # test correct time transformation occured (collapsed to only 2 values, one for each year)


# exmample 3: count number of 3-day events in each year that continously exceed the threshold
def test_hourly_ex3(T2_hourly):
    exc_counts = threshold_tools.get_exceedance_count(
        T2_hourly, threshold_value=305, period=(1, "year"), duration1=(72, "hour")
    )
    assert (exc_counts >= 0).all()  # test no negative values
    assert (
        len(exc_counts.time) == 2
    )  # test correct time transformation occured (collapsed to only 2 values, one for each year)


# exmample 4: count number of 3-day events in each year that exceed the threshold once each day
def test_hourly_ex4(T2_hourly):
    exc_counts = threshold_tools.get_exceedance_count(
        T2_hourly,
        threshold_value=305,
        period=(1, "year"),
        duration2=(3, "day"),
        groupby=(1, "day"),
    )
    assert (exc_counts >= 0).all()  # test no negative values
    assert (
        len(exc_counts.time) == 2
    )  # test correct time transformation occured (collapsed to only 2 values, one for each year)


# test current behavior of `duration` options: a six events in a row is counted as 4 3-hour events
def test_duration():
    da = xr.DataArray(
        [1, 1, 1, 1, 1, 1],
        coords={"time": pd.date_range("2000-01-01", freq="1h", periods=6)},
        attrs={"frequency": "hourly", "units": "T"},
    )
    exc_counts = threshold_tools.get_exceedance_count(da, 0, duration1=(3, "hour"))
    assert exc_counts == 4  # four of the six hours are the start of a 3-hour event


# ------------- Test helper functions for plotting -----------------------------


# example name 1: Number of hours each year
def test_name1():
    ex1 = xr.DataArray(
        attrs={
            "frequency": "hourly",
            "period": (1, "year"),
            "duration1": None,
            "group": None,
            "duration2": None,
            "variable_name": "Air Temperature at 2m",
            "threshold_direction": "above",
            "threshold_value": 299,
            "variable_units": "K",
        }
    )
    title1 = threshold_tools.exceedance_plot_title(ex1)
    assert title1 == "Air Temperature at 2m: events above 299K"
    subtitle1 = threshold_tools.exceedance_plot_subtitle(ex1)
    assert subtitle1 == "Number of hours each year"


# example name 2: Number of days each year with conditions lasting at least 1 hour
def test_name2():
    ex2 = xr.DataArray(
        attrs={
            "frequency": "hourly",
            "period": (1, "year"),
            "duration1": (1, "hour"),
            "group": (1, "day"),
            "duration2": (1, "day"),
            "variable_name": "Air Temperature at 2m",
            "threshold_direction": "above",
            "threshold_value": 299,
            "variable_units": "K",
        }
    )
    title2 = threshold_tools.exceedance_plot_title(ex2)
    assert title2 == "Air Temperature at 2m: events above 299K"
    subtitle2 = threshold_tools.exceedance_plot_subtitle(ex2)
    assert (
        subtitle2 == "Number of days each year with conditions lasting at least 1 hour"
    )


# example name 3: Number of 3-day events per 1 year
def test_name3():
    ex3 = xr.DataArray(
        attrs={
            "frequency": "hourly",
            "period": (1, "year"),
            "duration1": (4, "hour"),
            "group": (1, "day"),
            "duration2": (3, "day"),
            "variable_name": "Air Temperature at 2m",
            "threshold_direction": "above",
            "threshold_value": 299,
            "variable_units": "K",
        }
    )
    title3 = threshold_tools.exceedance_plot_title(ex3)
    assert title3 == "Air Temperature at 2m: events above 299K"
    subtitle3 = threshold_tools.exceedance_plot_subtitle(ex3)
    assert (
        subtitle3
        == "Number of 3-day events each year with conditions lasting at least 4 hours each day"
    )
