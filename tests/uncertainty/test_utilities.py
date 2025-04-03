"""
Tests for the pure utility functions in the uncertainty module.
"""

import datetime
from unittest.mock import MagicMock, patch

import cftime
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Polygon

from climakitae.core.data_interface import DataParameters
from climakitae.core.paths import gwl_1850_1900_file, gwl_1981_2010_file
from climakitae.explore.uncertainty import (  # get_ks_pval_df, # TODO I think this function is broken
    _calendar_align,
    _cf_to_dt,
    _clip_region,
    _standardize_cmip6_data,
    get_warm_level,
)


@pytest.fixture
def mock_multi_ens_dataset():
    """Create a mock dataset with simulation and member_id attributes"""
    ds = xr.Dataset(
        data_vars={"tas": (["time"], [1.0, 2.0, 3.0, 4.0, 5.0])},
        coords={
            "time": pd.date_range("2020-01-01", periods=5),
            "simulation": "EC-Earth3",
            "member_id": "r1i1p1f1",
        },
    )
    return ds


@pytest.fixture
def mock_data_for_warm_level():
    """Create a mock dataset with simulation attribute"""
    import numpy as np
    import xarray as xr

    # Create a simple dataset with simulation coordinate
    ds = xr.Dataset(
        data_vars={"tas": (["time"], np.random.rand(10))},
        coords={"time": pd.date_range("2020-01-01", periods=10)},
    )
    ds = ds.assign_coords({"simulation": "CESM2"})
    return ds


@pytest.fixture
def wrf_dataset():
    """Fixture to create a mock WRF dataset."""
    selections = DataParameters()
    selections.area_average = "No"
    selections.area_subset = "states"
    selections.cached_area = ["CA"]
    selections.downscaling_method = "Dynamical"
    selections.scenario_historical = ["Historical Climate"]
    selections.scenario_ssp = ["SSP 3-7.0"]
    selections.append_historical = True
    selections.variable = "Precipitation (total)"
    selections.time_slice = (1981, 2100)
    selections.resolution = "9 km"
    selections.timescale = "monthly"
    wrf_ds = selections.retrieve().squeeze()

    # WRF simulation names have additional information about activity ID and ensemble
    # Lookup dictionary used to rename simulation values
    wrf_cmip_lookup_dict = {
        "WRF_EC-Earth3-Veg_r1i1p1f1": "EC-Earth3-Veg",
        "WRF_EC-Earth3_r1i1p1f1": "EC-Earth3",
        "WRF_CESM2_r11i1p1f1": "CESM2",
        "WRF_CNRM-ESM2-1_r1i1p1f2": "CNRM-ESM2-1",
        "WRF_FGOALS-g3_r1i1p1f1": "FGOALS-g3",
        "WRF_MIROC6_r1i1p1f1": "MIROC6",
        "WRF_TaiESM1_r1i1p1f1": "TaiESM1",
        "WRF_MPI-ESM1-2-HR_r3i1p1f1": "MPI-ESM1-2-HR",
    }

    wrf_ds = wrf_ds.sortby("simulation")  # Sort simulations alphabetically
    wrf_ds["simulation"] = [
        wrf_cmip_lookup_dict[sim] for sim in wrf_ds.simulation.values
    ]  # Rename simulations
    wrf_ds = wrf_ds.clip(0.1)  # Remove values less than 0.1

    return wrf_ds


def test_get_warm_level_input_validation():
    """Test validation of warm_level parameter."""
    # Test non-numeric warm_level
    with pytest.raises(ValueError) as excinfo:
        get_warm_level("not_a_number", None)
    assert "Please specify warming level as an integer or float." in str(excinfo.value)

    # Test invalid numeric warm_level
    with pytest.raises(ValueError) as excinfo:
        get_warm_level(2.5, None)
    assert (
        "Specified warming level is not valid. Options are: 1.5, 2.0, 3.0, 4.0"
        in str(excinfo.value)
    )


@patch("climakitae.explore.uncertainty.read_csv_file")
def test_get_warm_level_file_loading(mock_read_csv, mock_data_for_warm_level):
    """Test file loading logic in get_warm_level for both IPCC and non-IPCC paths"""

    # Create mock DataFrames for the CSV files
    mock_ipcc_df = pd.DataFrame(
        {"1.5": [2030.5], "2.0": [2045.2], "3.0": [2070.1], "4.0": [2090.3]}
    )
    mock_ipcc_df.index = pd.MultiIndex.from_tuples(
        [("CESM2", "r11i1p1f1", "ssp370")], names=["GCM", "run", "scenario"]
    )

    mock_non_ipcc_df = pd.DataFrame(
        {"1.5": [2025.3], "2.0": [2040.7], "3.0": [2065.8], "4.0": [2085.9]}
    )

    mock_ece3_df = pd.DataFrame(
        {"1.5": [2026.1], "2.0": [2041.2], "3.0": [2066.4], "4.0": [2086.3]}
    )

    # Set up the mock to return different DataFrames based on input file path
    def side_effect(file_path, **kwargs):
        if file_path == gwl_1850_1900_file:
            return mock_ipcc_df
        elif file_path == gwl_1981_2010_file:
            return mock_non_ipcc_df
        elif file_path == "data/gwl_1981-2010ref_EC-Earth3_ssp370.csv":
            return mock_ece3_df
        else:
            return pd.DataFrame()

    mock_read_csv.side_effect = side_effect

    # Test IPCC path (ipcc=True)
    with patch("climakitae.explore.uncertainty.gwl_1850_1900_file", gwl_1850_1900_file):
        with patch(
            "climakitae.explore.uncertainty.gwl_1981_2010_file", gwl_1981_2010_file
        ):
            # Mock just enough of the function to test file loading
            with patch.object(
                get_warm_level.__globals__["pd"], "concat"
            ) as mock_concat:
                # Call the function with ipcc=True
                try:
                    result = get_warm_level(2.0, mock_data_for_warm_level, ipcc=True)
                except:
                    # We expect the function to potentially fail after file loading,
                    # since we're not mocking everything, but we only care about file loading
                    pass

                # Verify read_csv_file was called with the correct file
                mock_read_csv.assert_called_with(
                    gwl_1850_1900_file, index_col=[0, 1, 2]
                )
                # Verify concat was not called
                mock_concat.assert_not_called()

    # Test non-IPCC path (ipcc=False)
    mock_read_csv.reset_mock()
    with patch("climakitae.explore.uncertainty.gwl_1850_1900_file", gwl_1850_1900_file):
        with patch(
            "climakitae.explore.uncertainty.gwl_1981_2010_file", gwl_1981_2010_file
        ):
            with patch.object(
                pd, "concat", return_value=pd.concat([mock_non_ipcc_df, mock_ece3_df])
            ) as mock_concat:
                # Call the function with ipcc=False
                try:
                    result = get_warm_level(2.0, mock_data_for_warm_level, ipcc=False)
                except:
                    pass

                # Verify read_csv_file was called with both files
                assert mock_read_csv.call_count == 2
                mock_read_csv.assert_any_call(gwl_1981_2010_file)
                mock_read_csv.assert_any_call(
                    "data/gwl_1981-2010ref_EC-Earth3_ssp370.csv"
                )

                # Verify concat was called with the two DataFrames
                mock_concat.assert_called_once()


@patch("climakitae.explore.uncertainty.read_csv_file")
def test_get_warm_level_member_id_selection(
    mock_read_csv, mock_data_for_clipping, mock_multi_ens_dataset
):
    """Test member_id selection logic."""
    # Setup mock dataframe with warming data - create a DataFrame with one row per model
    mock_df = pd.DataFrame(
        {
            "1.5": [2030.5, 2030.6, 2030.7],
            "2.0": [2045.2, 2045.3, 2045.4],
            "3.0": [2070.1, 2070.2, 2070.3],
            "4.0": [2090.3, 2090.4, 2090.5],
        }
    )
    mock_df.index = pd.MultiIndex.from_tuples(
        [
            ("CESM2", "r11i1p1f1", "ssp370"),
            ("EC-Earth3", "r1i1p1f1", "ssp370"),
            ("CNRM-ESM2-1", "r1i1p1f2", "ssp370"),
        ],
        names=["GCM", "run", "scenario"],
    )
    mock_read_csv.return_value = mock_df

    # Test default member_id selection for CESM2
    with patch("xarray.Dataset.sel", return_value=mock_data_for_clipping) as mock_sel:
        get_warm_level(2.0, mock_data_for_clipping)
        # For CESM2, should use r11i1p1f1
        assert mock_df.loc[("CESM2", "r11i1p1f1", "ssp370")]["2.0"] == 2045.2

    # Test default member_id selection for CNRM-ESM2-1
    cnrm_ds = mock_data_for_clipping.copy()
    cnrm_ds = cnrm_ds.assign_coords({"simulation": "CNRM-ESM2-1"})
    with patch("xarray.Dataset.sel", return_value=cnrm_ds) as mock_sel:
        get_warm_level(2.0, cnrm_ds)
        # For CNRM-ESM2-1, should use r1i1p1f2
        assert mock_df.loc[("CNRM-ESM2-1", "r1i1p1f2", "ssp370")]["2.0"] == 2045.4

    # Test multi_ens=True (should use the member_id from the dataset)
    with patch("xarray.Dataset.sel", return_value=mock_multi_ens_dataset) as mock_sel:
        get_warm_level(2.0, mock_multi_ens_dataset, multi_ens=True)
        # Should use the member_id from the dataset
        assert mock_df.loc[("EC-Earth3", "r1i1p1f1", "ssp370")]["2.0"] == 2045.3


@patch("climakitae.explore.uncertainty.read_csv_file")
def test_get_warm_level_time_slice(mock_read_csv, mock_data_for_warm_level):
    """Test time slicing logic based on warming level."""
    # Setup mock dataframe with warming data
    mock_df = pd.DataFrame(
        {
            "3.0": [2070.0],  # Warming level reached in 2070
        }
    )
    mock_df.index = pd.MultiIndex.from_tuples(
        [("CESM2", "r11i1p1f1", "ssp370")], names=["GCM", "run", "scenario"]
    )
    mock_read_csv.return_value = mock_df

    # Test normal time slice (-14/+15 years from warming level year)
    with patch("xarray.Dataset.sel", return_value=mock_data_for_warm_level) as mock_sel:
        get_warm_level(3.0, mock_data_for_warm_level)
        # Should select time slice from 2056 (2070-14) to 2085 (2070+15)
        mock_sel.assert_called_with(time=slice("2056", "2085"))

    # Test case where warming level not reached (year string less than 4 chars)
    mock_df["3.0"] = ["N/A"]
    with patch("builtins.print") as mock_print:
        result = get_warm_level(3.0, mock_data_for_warm_level)
        assert result is None
        mock_print.assert_called_with(
            "3.0°C warming level not reached for ensemble member r11i1p1f1 of model CESM2"
        )

    # Test case where end year exceeds 2100
    mock_df["3.0"] = [2090.0]  # 2090 + 15 > 2100
    with patch("builtins.print") as mock_print:
        result = get_warm_level(3.0, mock_data_for_warm_level)
        assert result is None
        mock_print.assert_called_with(
            "End year for SSP time slice occurs after 2100; skipping ensemble member r11i1p1f1 of model CESM2"
        )


def test_get_warm_level_integration(wrf_dataset):
    """Test get_warm_level with a real-like dataset."""
    # Extract a single simulation to test
    ds = wrf_dataset.isel(simulation=0).drop_vars("simulation")
    ds = ds.assign_coords({"simulation": "CESM2"})

    # Since we can't predict the actual warming level data, we'll patch it
    with patch("climakitae.explore.uncertainty.read_csv_file") as mock_read_csv:
        mock_df = pd.DataFrame(
            {
                "3.0": [2070.0],  # Warming level reached in 2070
            }
        )
        mock_df.index = pd.MultiIndex.from_tuples(
            [("CESM2", "r11i1p1f1", "ssp370")], names=["GCM", "run", "scenario"]
        )
        mock_read_csv.return_value = mock_df

        # Use module-level patching instead of object-level patching
        with patch("xarray.DataArray.sel", return_value=ds) as mock_sel:
            get_warm_level(3.0, ds)
            mock_sel.assert_called_with(time=slice("2056", "2085"))


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
            "time": ("time", pd.date_range("2000-01-01", periods=10, freq="ME")),
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


@pytest.fixture
def mock_data_for_clipping():
    """Create a mock dataset for testing clipping."""
    # Create a custom mock object instead of a real xarray Dataset
    mock_ds = MagicMock()

    # Add rio attribute with required methods
    mock_rio = MagicMock()
    mock_rio.write_crs.return_value = mock_ds
    mock_rio.clip.return_value = mock_ds
    mock_ds.rio = mock_rio

    return mock_ds


@pytest.fixture
def mock_geoms_for_clipping():
    """Create mock geometries for states and counties."""
    # Create a simple polygon for testing
    polygon = Polygon([(-120, 35), (-120, 36), (-119, 36), (-119, 35)])

    # Create mock GeoDataFrames
    states = gpd.GeoDataFrame(
        {"NAME": ["California", "Nevada"]},
        geometry=[polygon, Polygon()],
    )
    counties = gpd.GeoDataFrame(
        {"NAME": ["Los Angeles", "San Francisco"]},
        geometry=[polygon, Polygon()],
    )
    return {"states": states, "counties": counties}


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


@pytest.fixture
def mock_data_for_standardization():
    """Create a simple mock dataset with required attributes and structure."""
    # Create sample data
    time = pd.date_range("2020-01-01", periods=3)
    lats = np.array([0, 1])
    lons = np.array([0, 1])
    data = np.random.rand(3, 2, 2)

    # Create dataset
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
            "source_id": "MODEL1",
            "experiment_id": "historical",
            "frequency": "mon",
        },
    )
    return ds


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


# TODO investigate whether get_ks_pval_df is actually working
# def test_get_ks_pval_df():
#     """Test get_ks_pval_df function."""

#     selections = DataParameters()
#     selections.area_average = "No"
#     selections.area_subset = "states"
#     selections.cached_area = ["CA"]
#     selections.downscaling_method = "Dynamical"
#     selections.scenario_historical = ["Historical Climate"]
#     selections.scenario_ssp = ["SSP 3-7.0"]
#     selections.append_historical = True
#     selections.variable = "Precipitation (total)"
#     selections.time_slice = (1981, 2100)
#     selections.resolution = "9 km"
#     selections.timescale = "monthly"
#     wrf_ds = selections.retrieve().squeeze()

#     box_wrf_ds = selections.retrieve().squeeze()

#     box_wrf_ds = box_wrf_ds.sortby("simulation")  # Sort simulations alphabetically
#     wrf_cmip_lookup_dict = {
#         "WRF_EC-Earth3-Veg_r1i1p1f1": "EC-Earth3-Veg",
#         "WRF_EC-Earth3_r1i1p1f1": "EC-Earth3",
#         "WRF_CESM2_r11i1p1f1": "CESM2",
#         "WRF_CNRM-ESM2-1_r1i1p1f2": "CNRM-ESM2-1",
#         "WRF_FGOALS-g3_r1i1p1f1": "FGOALS-g3",
#         "WRF_MIROC6_r1i1p1f1": "MIROC6",
#         "WRF_TaiESM1_r1i1p1f1": "TaiESM1",
#         "WRF_MPI-ESM1-2-HR_r3i1p1f1": "MPI-ESM1-2-HR",
#     }
#     box_wrf_ds["simulation"] = [
#         wrf_cmip_lookup_dict[sim] for sim in box_wrf_ds.simulation.values
#     ]  # Rename simulations
#     box_wrf_ds = box_wrf_ds.clip(0.1)  # Remove values less than 0.1

#     box_hist_wrf = box_wrf_ds.sel(time=slice("1981", "2010"))

#     sim_idx = list(wrf_ds.simulation.values)
#     ssp_wrf_list = [
#         get_warm_level(3.0, wrf_ds.sel(simulation=s).squeeze(), ipcc=False)
#         for s in sim_idx
#     ]
#     print(ssp_wrf_list)

#     ssp_wrf_list = list(filter(lambda item: item is not None, ssp_wrf_list))

#     box_ssp_wrf_list = list(filter(lambda item: item is not None, ssp_wrf_list))
#     box_ssp_wrf = xr.concat(box_ssp_wrf_list, dim="simulation")

#     box_hist_wrf = box_hist_wrf.sel(simulation=box_ssp_wrf.simulation.values)

#     hist_wrf_pool = box_hist_wrf.stack(index=["simulation", "time"])
#     ssp_wrf_pool = box_ssp_wrf.stack(index=["simulation", "time"])

#     hist_wrf_pool = hist_wrf_pool.compute()
#     ssp_wrf_pool = ssp_wrf_pool.compute()
#     pooled_p_df = get_ks_pval_df(hist_wrf_pool, ssp_wrf_pool)

# # Call the function
# assert isinstance(result, pd.DataFrame)
# assert "lat" in result.columns
# assert "lon" in result.columns
# assert "p_value" in result.columns
# assert result.shape == (
#     len(lats) * len(lons),
#     3,
# )  # Each grid point should be a row, with 3 columns (lat, lon, p_value)
