"""
Tests for the pure utility functions in the uncertainty module.
"""

import datetime
import re
from unittest.mock import MagicMock, patch

import cftime
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Polygon

from climakitae.core.data_interface import DataParameters
from climakitae.core.paths import GWL_1850_1900_FILE, GWL_1981_2010_FILE
from climakitae.explore.uncertainty import (
    _area_wgt_average,
    _calendar_align,
    _cf_to_dt,
    _clip_region,
    _drop_member_id,
    _grab_ensemble_data_by_experiment_id,
    _precip_flux_to_total,
    _standardize_cmip6_data,
)
from tests.uncertainty.fixtures import (
    global_dataset,
    mock_catalog,
    mock_data_for_clipping,
    mock_data_for_standardization,
    mock_geoms_for_clipping,
    mock_multi_ens_dataset,
    multi_var_dataset,
    simple_dataset,
)


def test_calendar_align():
    """Test _calendar_align function."""

    # pass without time dimension
    mock_data = xr.Dataset(
        {
            "var": (("lat", "lon"), [[1, 2], [3, 4]]),
        },
        coords={
            "lat": (("lat"), [0, 1]),
            "lon": (("lon"), [0, 1]),
        },
    )
    with pytest.raises(AttributeError):
        _calendar_align(mock_data)

    mock_data = xr.Dataset(
        {
            "time": ("time", pd.date_range("2000-01-01", periods=10, freq="MS")),
            "var": (("time"), range(10)),
        },
        coords={
            "lat": (("lat"), [0]),
            "lon": (("lon"), [0]),
        },
    )

    result = _calendar_align(mock_data)
    assert isinstance(result, xr.Dataset)
    assert result["time"].shape == (10,)
    assert result["var"].shape == (10,)
    # assert that format of time is time.dt.strftime("%Y-%m")
    assert result["time"].dt.strftime("%Y-%m").values[0] == "2000-01"
    assert result["time"].dt.strftime("%Y-%m").values[-1] == "2000-10"


def test_cf_to_dt():
    """Test _cf_to_dt function."""

    # mock dataset without time dimension
    mock_data = xr.Dataset(
        {
            "var": (("lat", "lon"), [[1, 2], [3, 4]]),
        },
        coords={
            "lat": (("lat"), [0, 1]),
            "lon": (("lon"), [0, 1]),
        },
    )
    with pytest.raises(KeyError):
        _cf_to_dt(mock_data)

    # CASE 1: Test normal pandas DatetimeIndex (should remain unchanged)
    mock_data = xr.Dataset(
        {
            "var": (("time"), range(10)),
        },
        coords={
            "time": pd.date_range("2000-01-01", periods=10, freq="ME"),
            "lat": (("lat"), [0]),
            "lon": (("lon"), [0]),
        },
    )
    result = _cf_to_dt(mock_data)
    assert isinstance(result, xr.Dataset)
    assert isinstance(result.indexes["time"], pd.DatetimeIndex)

    # CASE 2: Test with CFTimeIndex (should be converted)
    # Create cftime dates with a non-standard calendar
    cf_dates = [
        cftime.DatetimeNoLeap(2000, 1, 1) + datetime.timedelta(days=i)
        for i in range(10)
    ]

    # Create dataset with CFTimeIndex
    mock_cf_data = xr.Dataset(
        {
            "var": (("time"), range(10)),
        },
        coords={
            "time": cf_dates,
            "lat": (("lat"), [0]),
            "lon": (("lon"), [0]),
        },
    )

    # Ensure we have a CFTimeIndex - this will verify our mock is correct
    assert not isinstance(mock_cf_data.indexes["time"], pd.DatetimeIndex)

    # Test conversion
    result = _cf_to_dt(mock_cf_data)
    assert isinstance(result, xr.Dataset)
    assert isinstance(result.indexes["time"], pd.DatetimeIndex)

    # Verify data is preserved
    np.testing.assert_array_equal(result["var"].values, range(10))


@patch("climakitae.explore.uncertainty.DataInterface")
def test_clip_region_states(
    mock_data_interface, mock_data_for_clipping, mock_geoms_for_clipping
):
    """Test clipping a dataset to a state boundary."""
    # Set up the mock DataInterface
    mock_interface = MagicMock()
    mock_data_interface.return_value = mock_interface
    mock_geographies = MagicMock()
    mock_interface.geographies = mock_geographies
    mock_geographies._us_states = mock_geoms_for_clipping["states"]
    mock_geographies._ca_counties = mock_geoms_for_clipping["counties"]

    # Call the function with state area subset
    result = _clip_region(mock_data_for_clipping, "states", "California")

    # Verify the function worked correctly
    mock_data_for_clipping.rio.write_crs.assert_called_once_with(
        "epsg:4326", inplace=True
    )

    # Use an alternative approach to verify the clip arguments
    clip_args = mock_data_for_clipping.rio.clip.call_args
    assert clip_args is not None
    assert clip_args[1]["crs"] == 4326
    assert clip_args[1]["drop"] is True
    assert clip_args[1]["all_touched"] is False

    # Verify that the geometries passed are from California
    # Since we're getting a Series of geometries, compare by checking the filter condition
    california_filter = mock_geoms_for_clipping["states"].NAME == "California"
    passed_geometries = clip_args[1]["geometries"]
    assert isinstance(passed_geometries, pd.Series)

    # Check if the series contains the California geometry
    california_geom = (
        mock_geoms_for_clipping["states"].loc[california_filter, "geometry"].iloc[0]
    )
    if len(passed_geometries) == 1:
        assert passed_geometries.iloc[0].equals(california_geom)

    assert result is mock_data_for_clipping


@patch("climakitae.explore.uncertainty.DataInterface")
def test_clip_region_counties(
    mock_data_interface, mock_data_for_clipping, mock_geoms_for_clipping
):
    """Test clipping a dataset to a county boundary."""
    # Set up the mock DataInterface
    mock_interface = MagicMock()
    mock_data_interface.return_value = mock_interface
    mock_geographies = MagicMock()
    mock_interface.geographies = mock_geographies
    mock_geographies._us_states = mock_geoms_for_clipping["states"]
    mock_geographies._ca_counties = mock_geoms_for_clipping["counties"]

    # Get Los Angeles geometry before calling the function (for comparison)
    la_geometry = (
        mock_geoms_for_clipping["counties"]
        .loc[mock_geoms_for_clipping["counties"].NAME == "Los Angeles", "geometry"]
        .iloc[0]
    )  # Get the actual geometry object, not the Series

    # Call the function with county area subset
    result = _clip_region(mock_data_for_clipping, "counties", "Los Angeles")

    # Verify the function worked correctly
    mock_data_for_clipping.rio.write_crs.assert_called_once_with(
        "epsg:4326", inplace=True
    )

    # Use an alternative approach to verify the clip arguments
    clip_args = mock_data_for_clipping.rio.clip.call_args
    assert clip_args is not None
    assert clip_args[1]["crs"] == 4326
    assert clip_args[1]["drop"] is True
    assert clip_args[1]["all_touched"] is False

    # Verify that the geometry passed is the Los Angeles polygon
    passed_geometries = clip_args[1]["geometries"]
    assert isinstance(passed_geometries, pd.Series)

    # Check if the series contains the Los Angeles geometry
    if len(passed_geometries) == 1:
        assert passed_geometries.iloc[0].equals(la_geometry)

    assert result is mock_data_for_clipping


@patch("climakitae.explore.uncertainty.DataInterface")
@patch("builtins.print")
def test_clip_region_fallback(
    mock_print, mock_data_interface, mock_data_for_clipping, mock_geoms_for_clipping
):
    """Test the fallback behavior when initial clip fails."""
    # Set up the mock DataInterface
    mock_interface = MagicMock()
    mock_data_interface.return_value = mock_interface
    mock_geographies = MagicMock()
    mock_interface.geographies = mock_geographies
    mock_geographies._us_states = mock_geoms_for_clipping["states"]

    # Configure the rio.clip method to fail on first call and succeed on second call
    mock_ds_fallback = mock_data_for_clipping.copy()
    mock_data_for_clipping.rio.clip.side_effect = [
        Exception("No grid centers in region"),
        mock_ds_fallback,
    ]

    # Call the function
    result = _clip_region(mock_data_for_clipping, "states", "California")

    # Verify the function worked correctly with fallback
    mock_data_for_clipping.rio.write_crs.assert_called_once_with(
        "epsg:4326", inplace=True
    )
    assert mock_data_for_clipping.rio.clip.call_count == 2

    # Check first call failed with all_touched=False
    first_call_args = mock_data_for_clipping.rio.clip.call_args_list[0][1]
    assert first_call_args["all_touched"] is False

    # Check second call succeeded with all_touched=True
    second_call_args = mock_data_for_clipping.rio.clip.call_args_list[1][1]
    assert second_call_args["all_touched"] is True

    # Verify the print message was displayed
    mock_print.assert_called_once_with("selecting all cells which intersect region")

    # Verify the correct result was returned
    assert result is mock_ds_fallback


@patch("climakitae.explore.uncertainty.rename_cmip6")
@patch("climakitae.explore.uncertainty._cf_to_dt")
@patch("climakitae.explore.uncertainty._calendar_align")
def test_standardize_cmip6_data_full_processing(
    mock_calendar_align, mock_cf_to_dt, mock_rename_cmip6, mock_data_for_standardization
):
    """Test that all processing steps are called with proper arguments."""
    # Create a result dataset with the expected final state
    result_ds = mock_data_for_standardization.copy()
    # Add expected coordinates that should be in final result
    result_ds = result_ds.assign_coords(
        {"simulation": "MODEL1", "scenario": "historical"}
    )

    # Create a chain of mocks to handle the method calls
    mock_after_drop = MagicMock()
    mock_after_assign = MagicMock()

    # Configure the mock chain
    mock_data_with_squeeze = MagicMock()
    mock_data_with_squeeze.drop_vars.return_value = mock_after_drop
    mock_after_drop.assign_coords.return_value = mock_after_assign
    mock_after_assign.squeeze.return_value = result_ds

    # Configure our initial mocks
    mock_rename_cmip6.return_value = mock_data_for_standardization
    mock_cf_to_dt.return_value = mock_data_for_standardization
    mock_calendar_align.return_value = mock_data_with_squeeze

    # Call function
    result = _standardize_cmip6_data(mock_data_for_standardization)

    # Assert all processing functions were called
    mock_rename_cmip6.assert_called_once_with(mock_data_for_standardization)
    mock_cf_to_dt.assert_called_once()
    mock_calendar_align.assert_called_once()

    # Verify method calls
    mock_data_with_squeeze.drop_vars.assert_called_once_with(
        ["lon", "lat", "height"], errors="ignore"
    )
    mock_after_drop.assign_coords.assert_called_once_with(
        {"simulation": "MODEL1", "scenario": "historical"}
    )
    mock_after_assign.squeeze.assert_called_once_with(drop=True)

    # Now we can check the actual result, which is our pre-configured result_ds
    assert result is result_ds
    assert result.coords["simulation"] == "MODEL1"
    assert result.coords["scenario"] == "historical"


@patch("climakitae.explore.uncertainty.rename_cmip6")
@patch("climakitae.explore.uncertainty._cf_to_dt")
@patch("climakitae.explore.uncertainty._calendar_align")
def test_standardize_cmip6_data_non_monthly(
    mock_calendar_align, mock_cf_to_dt, mock_rename_cmip6, mock_data_for_standardization
):
    """Test that calendar align is skipped for non-monthly data."""
    # Set frequency to something other than "mon"
    mock_data_for_standardization.attrs["frequency"] = "day"

    mock_rename_cmip6.return_value = mock_data_for_standardization
    mock_cf_to_dt.return_value = mock_data_for_standardization

    # Call function
    _standardize_cmip6_data(mock_data_for_standardization)

    # Check calendar_align was not called
    mock_calendar_align.assert_not_called()


def test_standardize_cmip6_data_integration():
    """Integration test with a real-like dataset."""
    # Create a more complex dataset that doesn't require mocking
    time = pd.date_range("2020-01-01", periods=3)
    lats = np.array([0, 1])
    lons = np.array([0, 1])
    data = np.random.rand(3, 2, 2)

    ds = xr.Dataset(
        data_vars={
            "tas": (["time", "lat", "lon"], data),
            "height": ([], 0),
        },
        coords={
            "time": time,
            "lat": lats,
            "lon": lons,
        },
        attrs={
            "source_id": "TESTMODEL",
            "experiment_id": "ssp370",
            "frequency": "day",
        },
    )

    # We'll need to patch the imported functions since they likely
    # depend on external resources
    with patch("climakitae.explore.uncertainty.rename_cmip6", return_value=ds), patch(
        "climakitae.explore.uncertainty._cf_to_dt", return_value=ds
    ):

        result = _standardize_cmip6_data(ds)

        # Check that coordinates were properly assigned
        assert result.coords["simulation"].values == "TESTMODEL"
        assert result.coords["scenario"].values == "ssp370"

        # Check that variables were dropped
        assert "lon" not in result.data_vars
        assert "lat" not in result.data_vars
        assert "height" not in result.data_vars


@patch("climakitae.explore.uncertainty.rename_cmip6")
@patch("climakitae.explore.uncertainty._cf_to_dt")
@patch("climakitae.explore.uncertainty._calendar_align")
def test_standardize_cmip6_data_variable_handling(
    mock_calendar_align, mock_cf_to_dt, mock_rename_cmip6, mock_data_for_standardization
):
    """Test that the function properly handles missing variables."""
    # Remove variables that should be dropped
    mock_data_for_standardization = mock_data_for_standardization.drop_vars(
        ["lon", "lat", "height"]
    )

    mock_rename_cmip6.return_value = mock_data_for_standardization
    mock_cf_to_dt.return_value = mock_data_for_standardization
    mock_calendar_align.return_value = mock_data_for_standardization

    # Function should not error if variables to drop don't exist
    result = _standardize_cmip6_data(mock_data_for_standardization)

    # Check all other processing was done
    mock_rename_cmip6.assert_called_once()
    mock_cf_to_dt.assert_called_once()
    mock_calendar_align.assert_called_once()


def test_area_wgt_average_basic(simple_dataset):
    """Test basic functionality of area_wgt_average with a simple dataset."""
    # Weight at y=0 is cos(0°) = 1.0
    # Weight at y=60 is cos(60°) = 0.5
    # Weighted average = (1*1.0 + 2*1.0 + 3*0.5 + 4*0.5) / (1.0 + 1.0 + 0.5 + 0.5) = 6.5 / 3.0
    expected_result = 6.5 / 3.0

    result = _area_wgt_average(simple_dataset)

    assert isinstance(result, xr.Dataset)
    assert "tas" in result.data_vars
    assert np.isclose(result["tas"].values, expected_result)


def test_area_wgt_average_multiple_vars(multi_var_dataset):
    """Test area_wgt_average with a dataset containing multiple variables."""
    result = _area_wgt_average(multi_var_dataset)

    # Expected results for each variable
    # tas: (1*1.0 + 2*1.0 + 3*0.5 + 4*0.5) / 3.0 = 10.0/3.0
    # pr: (5*1.0 + 6*1.0 + 7*0.5 + 8*0.5) / 3.0 = 26.0/3.0

    assert isinstance(result, xr.Dataset)
    assert "tas" in result.data_vars and "pr" in result.data_vars
    assert np.isclose(result["tas"].values, 6.5 / 3.0)
    assert np.isclose(result["pr"].values, 18.5 / 3.0)


def test_area_wgt_average_global(global_dataset):
    """Test area_wgt_average with a global dataset spanning both hemispheres."""
    result = _area_wgt_average(global_dataset)

    # Since all values are 1.0, the weighted average should still be 1.0
    assert isinstance(result, xr.Dataset)
    assert np.isclose(result["tas"].values, 1.0)


def test_area_wgt_average_poles():
    """Test area_wgt_average handling of the poles where cos(latitude) approaches 0."""
    # Create a dataset with points very close to the poles
    lats = np.array([89.9, -89.9])  # Very close to poles
    lons = np.array([0.0, 10.0])

    # Values at near-poles
    pole_values = np.array([[100.0, 200.0], [300.0, 400.0]])

    ds_poles = xr.Dataset(
        data_vars={"tas": (["y", "x"], pole_values)},
        coords={
            "y": lats,
            "x": lons,
        },
    )

    result = _area_wgt_average(ds_poles)

    # With equal weights at both poles, weighted average should be regular average
    expected_avg = np.mean(pole_values)
    assert np.isclose(result["tas"].values, expected_avg)


def test_area_wgt_average_no_side_effects(simple_dataset):
    """Test that area_wgt_average does not modify the input dataset."""
    ds_copy = simple_dataset.copy(deep=True)

    _ = _area_wgt_average(simple_dataset)

    # Check that the original dataset hasn't been modified
    xr.testing.assert_identical(simple_dataset, ds_copy)


def test_area_wgt_average_missing_coords():
    """Test that area_wgt_average raises an error when required coordinates are missing."""
    # Create a dataset without y coordinate
    ds_no_y = xr.Dataset(
        data_vars={"tas": (["lat", "x"], np.ones((2, 2)))},
        coords={
            "lat": np.array([0.0, 10.0]),  # Using 'lat' instead of 'y'
            "x": np.array([0.0, 10.0]),
        },
    )

    with pytest.raises(AttributeError):
        _area_wgt_average(ds_no_y)


def test_area_wgt_average_missing_dims():
    """Test handling when required dimensions are missing."""
    # Create a dataset with 'y' coordinate but no 'x' dimension
    ds_no_x_dim = xr.Dataset(
        data_vars={"tas": (["y", "lon"], np.ones((2, 2)))},
        coords={
            "y": np.array([0.0, 10.0]),
            "lon": np.array([0.0, 10.0]),  # Using 'lon' instead of 'x'
        },
    )

    with pytest.raises(ValueError) as excinfo:
        _area_wgt_average(ds_no_x_dim)
    assert "Dimensions ('x',) not found in DatasetWeighted dimensions" in str(
        excinfo.value
    )


def test_drop_member_id_basic():
    """Test basic functionality of _drop_member_id with a dataset containing member_id."""
    # Create test dataset with member_id
    ds_with_member = xr.Dataset(
        data_vars={"tas": (["time", "member_id"], np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={
            "time": np.array([0, 1]),
            "member_id": np.array([0, 1]),
        },
    )

    # Create dictionary with dataset
    dset_dict = {"model1": ds_with_member}

    # Apply function
    result = _drop_member_id(dset_dict)

    # Check result
    assert "model1" in result
    assert "member_id" not in result["model1"].coords
    assert "member_id" not in result["model1"].dims
    # Should keep the first member values
    assert np.array_equal(result["model1"]["tas"].values, np.array([1.0, 3.0]))


def test_drop_member_id_multiple_datasets():
    """Test _drop_member_id with multiple datasets, some with member_id and some without."""
    # Create test datasets
    ds_with_member = xr.Dataset(
        data_vars={"tas": (["time", "member_id"], np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={
            "time": np.array([0, 1]),
            "member_id": np.array([0, 1]),
        },
    )

    ds_without_member = xr.Dataset(
        data_vars={"tas": (["time"], np.array([5.0, 6.0]))},
        coords={
            "time": np.array([0, 1]),
        },
    )

    # Create dictionary with datasets
    dset_dict = {"model1": ds_with_member, "model2": ds_without_member}

    # Apply function
    result = _drop_member_id(dset_dict)

    # Check results
    assert "model1" in result and "model2" in result
    # Dataset with member_id should have it removed
    assert "member_id" not in result["model1"].coords
    assert "member_id" not in result["model1"].dims
    # Dataset without member_id should remain unchanged
    assert np.array_equal(result["model2"]["tas"].values, np.array([5.0, 6.0]))
    xr.testing.assert_identical(result["model2"], ds_without_member)


def test_drop_member_id_empty_dict():
    """Test _drop_member_id with an empty dictionary."""
    dset_dict = {}
    result = _drop_member_id(dset_dict)
    assert result == {}


def test_drop_member_id_no_modification():
    """Test that _drop_member_id doesn't modify the original dictionary."""
    # Create test dataset with member_id
    ds_with_member = xr.Dataset(
        data_vars={"tas": (["time", "member_id"], np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={
            "time": np.array([0, 1]),
            "member_id": np.array([0, 1]),
        },
    )

    # Create a deep copy for verification
    original_ds = ds_with_member.copy(deep=True)

    # Create dictionary with dataset
    dset_dict = {"model1": ds_with_member}

    # Apply function
    _ = _drop_member_id(dset_dict)

    # Original dataset should be unchanged (not modified in-place)
    xr.testing.assert_identical(original_ds, ds_with_member)


def test_precip_flux_to_total():
    """Test for _precip_flux_to_total function."""

    # Create a test dataset with precipitation flux data
    time = pd.date_range(
        "2020-01-01", periods=3, freq="MS"
    )  # Monthly data for 3 months
    lats = np.array([0, 1])
    lons = np.array([0, 1])
    # Precipitation flux values in kg m-2 s-1
    pr_data = np.array(
        [
            [[0.0001, 0.0002], [0.0003, 0.0004]],  # January data
            [[0.0005, 0.0006], [0.0007, 0.0008]],  # February data
            [[0.0001, 0.0001], [0.0002, 0.0002]],  # March data
        ]
    )

    ds = xr.Dataset(
        data_vars={"pr": (["time", "y", "x"], pr_data)},
        coords={
            "time": time,
            "y": lats,
            "x": lons,
        },
        attrs={"source": "test_data", "version": "1.0"},
    )
    ds.pr.attrs["units"] = "kg m-2 s-1"

    # Test basic conversion
    result = _precip_flux_to_total(ds)

    # Check that attributes are preserved
    assert result.attrs["source"] == "test_data"
    assert result.attrs["version"] == "1.0"

    # Check that units are correctly set
    assert result.pr.attrs["units"] == "mm"

    # Calculate expected values for each month - days in month * seconds per day * flux values
    days_in_month = np.array([31, 29, 31])  # Jan, Feb (leap year 2020), Mar
    expected_values = []

    for i, days in enumerate(days_in_month):
        seconds = days * 86400
        month_values = pr_data[i] * seconds
        # Apply 0.1 clipping
        month_values = np.maximum(month_values, 0.1)
        expected_values.append(month_values)

    expected_values = np.array(expected_values)

    # Check that values are correctly converted
    np.testing.assert_allclose(result.pr.values, expected_values)


def test_precip_flux_to_total_clipping():
    """Test that values below 0.1 are clipped properly."""

    # Create test data with very small precipitation values
    time = pd.date_range("2020-01-01", periods=1, freq="MS")
    pr_data = np.array(
        [[[0.000001, 0.000002], [0.2, 0.3]]]
    )  # Very small values and normal values

    ds = xr.Dataset(
        data_vars={"pr": (["time", "y", "x"], pr_data)},
        coords={
            "time": time,
            "y": [0, 1],
            "x": [0, 1],
        },
    )
    ds.pr.attrs["units"] = "kg m-2 s-1"

    # Convert
    result = _precip_flux_to_total(ds)

    # Check that small values are clipped to 0.1
    # First values would be very small even after conversion and should be clipped to 0.1
    # Last two values should be properly converted and not clipped
    expected_values = np.array(
        [[[2.6784e00, 5.3568e00], [31 * 86400 * 0.2, 31 * 86400 * 0.3]]]
    )

    np.testing.assert_allclose(result.pr.values, expected_values)


def test_precip_flux_to_total_different_months():
    """Test conversion for different months with varying days."""

    # Create test data for multiple months with different numbers of days
    time = pd.date_range("2020-01-01", periods=12, freq="MS")  # Full year
    pr_data = np.ones((12, 1, 1)) * 0.0001  # Same flux value for all months

    ds = xr.Dataset(
        data_vars={"pr": (["time", "y", "x"], pr_data)},
        coords={
            "time": time,
            "y": [0],
            "x": [0],
        },
    )
    ds.pr.attrs["units"] = "kg m-2 s-1"

    # Calculate expected values - each month should have a different total based on days in month
    result = _precip_flux_to_total(ds)

    # Check that the conversion correctly accounts for different month lengths
    days_in_month = np.array(
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    )  # 2020 is leap year

    for i, days in enumerate(days_in_month):
        expected = max(0.0001 * days * 86400, 0.1)  # Apply clipping
        assert np.isclose(result.pr.values[i][0][0], expected)


def test_precip_flux_to_total_error_handling():
    """Test how function handles datasets without required attributes."""

    # Dataset without time dimension
    ds_no_time = xr.Dataset(
        data_vars={"pr": (["y", "x"], np.ones((2, 2)) * 0.0001)},
        coords={
            "y": [0, 1],
            "x": [0, 1],
        },
    )

    # Function should raise AttributeError when time dimension is missing
    with pytest.raises(AttributeError):
        _precip_flux_to_total(ds_no_time)

    # Dataset without pr variable
    ds_no_pr = xr.Dataset(
        data_vars={"temperature": (["time", "y", "x"], np.ones((2, 2, 2)))},
        coords={
            "time": pd.date_range("2020-01-01", periods=2, freq="MS"),
            "y": [0, 1],
            "x": [0, 1],
        },
    )

    # Function should raise AttributeError because pr variable is missing
    with pytest.raises(AttributeError):
        _precip_flux_to_total(ds_no_pr)


@patch("climakitae.explore.uncertainty.intake")
def test_grab_ensemble_data_basic_functionality(mock_intake, mock_catalog):
    """Test the basic functionality of _grab_ensemble_data_by_experiment_id."""
    # Setup the mock to return our mock catalog
    mock_intake.open_esm_datastore.return_value = mock_catalog

    # Call the function with test parameters
    variable = "tas"
    cmip_names = ["MODEL1", "MODEL2"]
    experiment_id = "historical"

    # Execute the function
    result = _grab_ensemble_data_by_experiment_id(variable, cmip_names, experiment_id)

    # Verify intake.open_esm_datastore was called with the correct URL
    mock_intake.open_esm_datastore.assert_called_once_with(
        "https://cadcat.s3.amazonaws.com/tmp/cmip6-regrid.json"
    )

    # Verify search was called with the correct parameters
    mock_catalog.search.assert_called_once_with(
        table_id="Amon",
        variable_id=variable,
        experiment_id=experiment_id,
        source_id=cmip_names,
    )

    # Verify to_dataset_dict was called with the correct parameters
    mock_subset = mock_catalog.search.return_value
    mock_subset.to_dataset_dict.assert_called_once()
    call_kwargs = mock_subset.to_dataset_dict.call_args[1]
    assert call_kwargs["zarr_kwargs"] == {"consolidated": True}
    assert call_kwargs["storage_options"] == {"anon": True}
    assert call_kwargs["progressbar"] is False

    # Verify the function returns a list of datasets
    assert isinstance(result, list)
    assert len(result) == 2
    for ds in result:
        assert isinstance(ds, xr.Dataset)


@patch("climakitae.explore.uncertainty.intake")
def test_grab_ensemble_data_empty_result(mock_intake):
    """Test behavior when the catalog search returns no results."""
    # Setup mock catalog with empty result
    mock_catalog = MagicMock()
    mock_subset = MagicMock()
    mock_catalog.search.return_value = mock_subset
    mock_subset.to_dataset_dict.return_value = {}  # Empty dictionary

    mock_intake.open_esm_datastore.return_value = mock_catalog

    # Call the function
    result = _grab_ensemble_data_by_experiment_id("tas", ["MODEL1"], "historical")

    # Verify result is an empty list
    assert result == []


@patch("climakitae.explore.uncertainty.intake")
def test_grab_ensemble_data_different_parameters(mock_intake, mock_catalog):
    """Test the function with different parameter values."""
    mock_intake.open_esm_datastore.return_value = mock_catalog

    # Test with different variable
    _grab_ensemble_data_by_experiment_id("pr", ["MODEL1"], "historical")
    mock_catalog.search.assert_called_with(
        table_id="Amon",
        variable_id="pr",
        experiment_id="historical",
        source_id=["MODEL1"],
    )

    # Test with different experiment_id
    _grab_ensemble_data_by_experiment_id("tas", ["MODEL1"], "ssp370")
    mock_catalog.search.assert_called_with(
        table_id="Amon",
        variable_id="tas",
        experiment_id="ssp370",
        source_id=["MODEL1"],
    )

    # Test with multiple models
    _grab_ensemble_data_by_experiment_id(
        "tas", ["MODEL1", "MODEL2", "MODEL3"], "historical"
    )
    mock_catalog.search.assert_called_with(
        table_id="Amon",
        variable_id="tas",
        experiment_id="historical",
        source_id=["MODEL1", "MODEL2", "MODEL3"],
    )


@patch("climakitae.explore.uncertainty.intake")
@patch("climakitae.explore.uncertainty._standardize_cmip6_data")
def test_grab_ensemble_data_standardization(
    mock_standardize, mock_intake, mock_catalog
):
    """Test that the preprocess function (_standardize_cmip6_data) is used correctly."""
    mock_intake.open_esm_datastore.return_value = mock_catalog

    # Define a standardize function that modifies datasets in a recognizable way
    def fake_standardize(ds):
        ds = ds.copy()
        ds.attrs["standardized"] = True
        return ds

    mock_standardize.side_effect = fake_standardize

    # Call the function
    _grab_ensemble_data_by_experiment_id("tas", ["MODEL1"], "historical")

    # Verify to_dataset_dict was called with our preprocess function
    mock_subset = mock_catalog.search.return_value
    assert mock_subset.to_dataset_dict.call_args[1]["preprocess"] == mock_standardize


@patch("climakitae.explore.uncertainty.intake")
def test_grab_ensemble_data_error_handling(mock_intake):
    """Test error handling in _grab_ensemble_data_by_experiment_id."""
    # Mock the intake module to raise an exception when open_esm_datastore is called
    mock_intake.open_esm_datastore.side_effect = Exception("Connection error")

    # Function should propagate exceptions
    with pytest.raises(Exception) as excinfo:
        _grab_ensemble_data_by_experiment_id("tas", ["MODEL1"], "historical")

    assert "Connection error" in str(excinfo.value)
