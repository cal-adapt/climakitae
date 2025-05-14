"""
Test exceedence event identification in threshold_tools functions.
Also test the plotting helper functions.
"""

import pandas as pd
import pytest
import xarray as xr

from climakitae.explore import threshold_tools

# ------------- Test kwarg compatibility and Exceptions ------------------------


def test_error1(test_data_2022_monthly_45km: xr.Dataset):
    """Test incompatible case: cannot specify a 1-day groupy for monthly data."""
    with pytest.raises(ValueError, match="Incompatible `group` specification"):
        threshold_tools.get_exceedance_count(
            test_data_2022_monthly_45km, threshold_value=305, groupby=(1, "day")
        )


def test_error2(T2_hourly: xr.DataArray):
    """Test incompatible case: incompatible: cannot specify a 3-day duration
    if grouped by month. But for now, `duration` not yet implemented.
    """
    with pytest.raises(
        ValueError, match="Incompatible `group` and `duration2` specification"
    ):
        threshold_tools.get_exceedance_count(
            T2_hourly, threshold_value=305, groupby=(1, "month"), duration2=(3, "day")
        )


# ------------- Tests with hourly data -----------------------------------------


@pytest.mark.advanced
def test_hourly_ex1(T2_hourly: xr.DataArray):
    """Example 1: count number of hours in each year exceeding the threshold."""
    exc_counts = threshold_tools.get_exceedance_count(
        T2_hourly, threshold_value=305, period=(1, "year")
    )
    assert (exc_counts >= 0).all()  # test no negative values
    assert (
        len(exc_counts.time) == 2
    )  # test correct time transformation occured (collapsed to only 2 values, one for each year)


@pytest.mark.advanced
def test_hourly_ex2(T2_hourly: xr.DataArray):
    """Example 2: count number of days in each year that have at least one hour
    exceeding the threshold."""
    exc_counts = threshold_tools.get_exceedance_count(
        T2_hourly, threshold_value=305, period=(1, "year"), groupby=(1, "day")
    )
    assert (exc_counts >= 0).all()  # test no negative values
    assert (
        len(exc_counts.time) == 2
    )  # test correct time transformation occured (collapsed to only 2 values, one for each year)


@pytest.mark.advanced
def test_hourly_ex3(T2_hourly: xr.DataArray):
    """Example 3: count number of 3-day events in each year that continously exceed the threshold."""
    exc_counts = threshold_tools.get_exceedance_count(
        T2_hourly, threshold_value=305, period=(1, "year"), duration1=(72, "hour")
    )
    assert (exc_counts >= 0).all()  # test no negative values
    assert (
        len(exc_counts.time) == 2
    )  # test correct time transformation occured (collapsed to only 2 values, one for each year)


@pytest.mark.advanced
def test_hourly_ex4(T2_hourly: xr.DataArray):
    """Example 4: count number of 3-day events in each year that exceed the threshold once each day."""
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


@pytest.mark.advanced
def test_duration():
    """Test current behavior of `duration` options: a six events in a row is
    counted as 4 3-hour events.
    """
    da = xr.DataArray(
        [1, 1, 1, 1, 1, 1],
        coords={"time": pd.date_range("2000-01-01", freq="1h", periods=6)},
        attrs={"frequency": "hourly", "units": "T"},
    )
    exc_counts = threshold_tools.get_exceedance_count(da, 0, duration1=(3, "hour"))
    assert exc_counts == 4  # four of the six hours are the start of a 3-hour event


# ------------- Test helper functions for plotting -----------------------------


def test_name1():
    """Example name 1: Number of hours each year."""
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


def test_name2():
    """Example name 2: Number of days each year with conditions lasting at least 1 hour."""
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


def test_name3():
    """Example name 3: Number of 3-day events per 1 year."""
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
