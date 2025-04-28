from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import shapely
import xarray as xr
from pytest import approx

from climakitae.core.data_interface import DataParameters
from climakitae.core.data_load import (
    _check_valid_unit_selection,
    _get_data_attributes,
    _get_eff_temp,
    _get_fosberg_fire_index,
    _get_hourly_dewpoint,
    _get_hourly_rh,
    _get_monthly_daily_dewpoint,
    _get_noaa_heat_index,
    _get_Uearth,
    _get_Vearth,
    _get_wind_speed_derived,
    _override_unit_defaults,
    _time_slice,
    area_subset_geometry,
    read_catalog_from_select,
)


def mock_data():
    # Used as a return value for mocking data download
    da = xr.DataArray(np.ones((1)), coords={"time": np.zeros((1))})
    da.attrs = {"units": ""}
    da.name = "data"
    return da


def mock_wind_var():
    # Simplify testing wind speed functions
    # with single gridpoint dataset
    lon = -123.521255
    lat = 9.475632
    da = xr.DataArray(
        data=np.array([[10]]),
        dims=["y", "x"],
        coords={
            "lon": (("y", "x"), np.array([[lon]])),
            "lat": (("y", "x"), np.array([[lat]])),
        },
    )
    da.name = "wind speed"
    da.attrs = {"units": "m/s"}
    return da


def mock_wrf_angles_ds():
    # Simplify testing wind speed functions with
    # single gridpoint dataset
    lon = -123.521255
    lat = 9.475632
    sinalpha = 0.6197577
    cosalpha = 0.78479314
    ds = xr.Dataset(
        data_vars={
            "COSALPHA": ((lat, lon), np.array([[cosalpha]])),
            "SINALPHA": ((lat, lon), np.array([[sinalpha]])),
        },
        coords={
            "lon": (("y", "x"), np.array([[lon]])),
            "lat": (("y", "x"), np.array([[lat]])),
        },
    )
    return ds


@pytest.fixture
def selections():
    selections = DataParameters()
    return selections


class TestDataLoadHidden:

    def test__override_unit_defaults(self):
        da = xr.DataArray()
        da.attrs = {}

        result1 = _override_unit_defaults(da, "pr")
        assert result1.attrs == da.attrs

        result2 = _override_unit_defaults(da, "huss")
        assert result2.attrs == {"units": "kg/kg"}

        result3 = _override_unit_defaults(da, "rsds")
        assert result3.attrs == {"units": "W/m2"}

    def test__check_valid_unit_selection(self, selections):
        result = _check_valid_unit_selection(selections)
        assert result is None

        # Edit a unit to cause exception
        selections.variable_options_df.loc[29, "unit"] = "C"
        with pytest.raises(ValueError):
            result = _check_valid_unit_selection(selections)

    def test__get_data_attributes(self, selections):
        result = _get_data_attributes(selections)

        # Check that dict with correct keys in returned.
        kwlist = [
            "variable_id",
            "extended_description",
            "units",
            "data_type",
            "resolution",
            "frequency",
            "location_subset",
            "approach",
            "downscaling_method",
        ]
        for item in kwlist:
            assert item in result

    def test__time_slice(self, selections):
        selections.time_slice = (1990, 1991)
        time = pd.date_range(start="1990-01-01", end="1992-12-31", freq="d")
        ds = xr.DataArray(
            np.ones((365 + 365 + 366)), coords={"time": time}, name="data"
        ).to_dataset()
        result = _time_slice(ds, selections)
        assert len(result.time) == 365 * 2


class TestAreaSubset:

    def test_area_subset_geometry_latlon(self, selections):
        selections.area_subset = "lat/lon"
        result = area_subset_geometry(selections)
        assert isinstance(result, list)
        assert isinstance(result[0], shapely.geometry.polygon.Polygon)

    def test_area_subset_geometry_latlon(self, selections):
        selections.area_subset = "states"
        selections.cached_area = ["CA"]
        selections.data_type = "Gridded"
        result = area_subset_geometry(selections)
        assert isinstance(result, list)
        assert isinstance(result[0], shapely.geometry.multipolygon.MultiPolygon)

    def test_area_subset_geometry_counties(self, selections):
        selections.area_subset = "CA counties"
        selections.cached_area = ["San Bernardino County"]
        selections.data_type = "Gridded"
        result = area_subset_geometry(selections)
        assert isinstance(result, list)
        assert isinstance(result[0], shapely.geometry.polygon.Polygon)

    def test_area_subset_geometry_watersheds(self, selections):
        selections.area_subset = "CA watersheds"
        selections.cached_area = ["Antelope-Fremont Valleys"]
        selections.data_type = "Gridded"
        result = area_subset_geometry(selections)
        assert isinstance(result, list)
        assert isinstance(result[0], shapely.geometry.polygon.Polygon)

    def test_area_subset_geometry_utilities(self, selections):
        selections.area_subset = "CA Electric Load Serving Entities (IOU & POU)"
        selections.cached_area = ["Redding Electric Utility"]
        selections.data_type = "Gridded"
        result = area_subset_geometry(selections)
        assert isinstance(result, list)
        assert isinstance(result[0], shapely.geometry.multipolygon.MultiPolygon)

    def test_area_subset_geometry_forecast_zones(self, selections):
        selections.area_subset = "CA Electricity Demand Forecast Zones"
        selections.cached_area = ["Central Valley"]
        selections.data_type = "Gridded"
        result = area_subset_geometry(selections)
        assert isinstance(result, list)
        assert isinstance(result[0], shapely.geometry.multipolygon.MultiPolygon)

    def test_area_subset_geometry_forecast_zones(self, selections):
        selections.area_subset = "CA Electric Balancing Authority Areas"
        selections.cached_area = ["IID"]
        selections.data_type = "Gridded"
        result = area_subset_geometry(selections)
        assert isinstance(result, list)
        assert isinstance(result[0], shapely.geometry.polygon.Polygon)


class TestDataLoadDerived:

    @patch("climakitae.core.data_load._get_data_one_var", return_value=mock_data())
    def test__get_hourly_rh(self, mock_get_data_one_var, selections):
        # Not testing correctness of calculation, just that function runs
        # and returns data.
        result = _get_hourly_rh(selections)

        assert isinstance(result, xr.core.dataarray.DataArray)

    @patch("climakitae.core.data_load._get_data_one_var", return_value=mock_data())
    def test__get_monthly_daily_dewpoint(self, mock_get_data_one_var, selections):
        # Not testing correctness of calculation, just that function runs
        # and returns data.
        result = _get_monthly_daily_dewpoint(selections)

        assert isinstance(result, xr.core.dataarray.DataArray)
        assert result.name == "dew_point_derived"

    @patch("climakitae.core.data_load._get_data_one_var", return_value=mock_data())
    def test__get_hourly_dewpoint(self, mock_get_data_one_var, selections):
        # Not testing correctness of calculation, just that function runs
        # and returns data.
        result = _get_hourly_dewpoint(selections)

        assert isinstance(result, xr.core.dataarray.DataArray)
        assert result.name == "dew_point_derived"

    @patch("climakitae.core.data_load._get_data_one_var", return_value=mock_data())
    def test__get_noaa_heat_index(self, mock_get_data_one_var, selections):
        # Not testing correctness of calculation, just that function runs
        # and returns data.
        result = _get_noaa_heat_index(selections)

        assert isinstance(result, xr.core.dataarray.DataArray)

    @patch("climakitae.core.data_load._get_data_one_var", return_value=mock_data())
    def test__get_eff_temp(self, mock_get_data_one_var, selections):
        # Not testing correctness of calculation, just that function runs
        # and returns data.
        result = _get_eff_temp(selections)

        assert isinstance(result, xr.core.dataarray.DataArray)

    @patch("climakitae.core.data_load._get_data_one_var", return_value=mock_data())
    def test__get_fosberg_fire_index(self, mock_get_data_one_var, selections):
        # Not testing correctness of calculation, just that function runs
        # and returns data.
        result = _get_fosberg_fire_index(selections)

        assert isinstance(result, xr.core.dataarray.DataArray)
        assert result.name == "Fosberg Fire Weather Index"

    @patch("xarray.backends.zarr.open_zarr", return_value=mock_wrf_angles_ds())
    @patch("climakitae.core.data_load._get_data_one_var", return_value=mock_wind_var())
    @patch(
        "climakitae.core.data_load._spatial_subset", return_value=mock_wrf_angles_ds()
    )
    def test__get_Uearth(
        self, mock_open_zarr, mock_get_data_one_var, mock_spatial_subset, selections
    ):
        # Function is heavily patched so we aren't downloading any data
        # Just stepping through the function with small canned datasets
        result = _get_Uearth(selections)
        assert isinstance(result, xr.core.dataarray.DataArray)
        assert result.name == selections.variable
        assert approx(result.data.item(), rel=1e-7) == 1.6503544

    @patch("xarray.backends.zarr.open_zarr", return_value=mock_wrf_angles_ds())
    @patch("climakitae.core.data_load._get_data_one_var", return_value=mock_wind_var())
    @patch(
        "climakitae.core.data_load._spatial_subset", return_value=mock_wrf_angles_ds()
    )
    def test__get_Vearth(
        self, mock_open_zarr, mock_get_data_one_var, mock_spatial_subset, selections
    ):
        # Function is heavily patched so we aren't downloading any data
        # Just stepping through the function with small canned datasets
        result = _get_Vearth(selections)
        assert isinstance(result, xr.core.dataarray.DataArray)
        assert result.name == selections.variable
        assert approx(result.data.item(), rel=1e-7) == 14.0455084

    @patch("climakitae.core.data_load._get_Uearth", return_value=mock_wind_var())
    @patch("climakitae.core.data_load._get_Vearth", return_value=mock_wind_var())
    def test__get_wind_speed_derived(
        self, mock_get_Uearth, mock_get_Vearth, selections
    ):
        # Function is heavily patched so we aren't downloading any data
        # Just stepping through the function with small canned datasets
        result = _get_wind_speed_derived(selections)
        expected = np.sqrt(np.square(1.6503544) + np.square(14.0455084))
        assert isinstance(result, xr.core.dataarray.DataArray)
        assert approx(result.data.item(), rel=1e-7) == expected
        assert result.name == "wind_speed_derived"
        assert result.attrs["units"] == "m s-1"

def TestCatalogFromSelect(self):

    def test_read_catalog_from_select_defaults(self,selections):
        result = read_catalog_from_select(selections)
        assert isinstance(result,xr.core.dataarray.DataArray)
        assert result.name == selections.variable
        assert result.attrs["variable_id"] == selections.variable_id
        # Check that there's at least one variant for each model in selections
        for sim in selections.simulation if sim != "ERA5":
            found = [x for x in result.simulation.data if sim in x]
            assert len(found) > 0
        assert result.attrs["data_type"] == selections.data_type
        assert result.attrs["downscaling_method"] == selections.downscaling_method
        assert result.attrs["units"] == selections.units
        assert result.attrs["units"] == selections.units
        # Check that all requested scenarios are present
        assert result.scenario.data[0] == selections.scenario_historical

    def test_read_catalog_from_select_ssp(self,selections):
        selections.time_slice=(1990,2050)
        selections.scenario_ssp = ['SSP 2-4.5']
        result = read_catalog_from_select(selections)
        assert result.scenario.data == 'Historical + SSP 2-4.5'