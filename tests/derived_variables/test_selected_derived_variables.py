"""Test functions for derived variables that are not loadable via get_data or new core.

Currently includes sea level pressure and geostrophic wind functions.
"""

import numpy as np
import pytest
import xarray as xr
from pytest import approx
from unittest.mock import patch

from climakitae.tools.derived_variables import (
    compute_sea_level_pressure,
    _wrf_deltas,
    _align_dim,
    _get_spatial_derivatives,
    _get_rotated_geostrophic_wind,
    compute_geostrophic_wind,
)


@pytest.fixture
def coords():
    return {
        "lon": -123.521255,
        "lat": 9.475632,
        "time_range": xr.date_range("2000-01-01 00:00", "2000-01-01 23:00", freq="h"),
    }


def _make_dataarray(data, coords, name, units, include_time=True, time_dim="time"):
    """Helper to create consistent mock DataArrays."""
    if include_time:
        if time_dim == "time":
            time_coord = coords["time_range"]
        elif time_dim == "time_delta":
            time_coord = [-1000 + h for h in range(0, len(coords["time_range"]))]
        da = xr.DataArray(
            data=np.expand_dims(np.tile([[data]], 24), 0),
            dims=["y", "x", time_dim],
            coords={
                "lon": (("y", "x"), np.array([[coords["lon"]]])),
                "lat": (("y", "x"), np.array([[coords["lat"]]])),
                time_dim: ((time_dim), time_coord),
            },
        )
    else:
        da = xr.DataArray(
            data=np.array([[data]]),
            dims=["y", "x"],
            coords={
                "lon": (("y", "x"), np.array([[coords["lon"]]])),
                "lat": (("y", "x"), np.array([[coords["lat"]]])),
            },
        )
    da.name = name
    da.attrs = {"units": units}
    return da


@pytest.fixture
def mock_psfc(coords) -> xr.DataArray:
    return _make_dataarray(85000, coords, "surface pressure", "Pa")


@pytest.fixture
def mock_t2(coords) -> xr.DataArray:
    return _make_dataarray(293.15, coords, "Air Temperature at 2m", "K")


@pytest.fixture
def mock_t2_no_time(coords) -> xr.DataArray:
    return _make_dataarray(
        293.15, coords, "Air Temperature at 2m", "K", include_time=False
    )


@pytest.fixture
def mock_q2(coords) -> xr.DataArray:
    return _make_dataarray(0.004, coords, "Water Vapor Mixing Ratio at 2m", "kg kg-1")


@pytest.fixture
def mock_psfc_time_delta(coords) -> xr.DataArray:
    return _make_dataarray(
        85000, coords, "surface pressure", "Pa", time_dim="time_delta"
    )


@pytest.fixture
def mock_t2_time_delta(coords) -> xr.DataArray:
    return _make_dataarray(
        293.15, coords, "Air Temperature at 2m", "K", time_dim="time_delta"
    )


@pytest.fixture
def mock_q2_time_delta(coords) -> xr.DataArray:
    return _make_dataarray(
        0.004,
        coords,
        "Water Vapor Mixing Ratio at 2m",
        "kg kg-1",
        time_dim="time_delta",
    )


@pytest.fixture
def mock_lapse_rate(coords) -> xr.DataArray:
    return _make_dataarray(0.009, coords, "lapse rate", "K/m", include_time=False)


@pytest.fixture
def mock_elevation(coords) -> xr.DataArray:
    return _make_dataarray(500, coords, "elevation", "m", include_time=False)


@pytest.fixture
def wrf_coords_d01() -> dict:
    wrf_coords = {
        "y": (("y"), np.array([-1.126089e06, -1.081089e06, -1.036089e06])),
        "x": (("x"), np.array([-6285114.165919, -6240114.165919, -6195114.165919])),
        "lon": (
            ("y", "x"),
            np.array(
                [
                    [-123.521255, -123.24165, -122.96051],
                    [-123.74222, -123.46225, -123.180725],
                    [-123.96474, -123.6844, -123.4025],
                ]
            ),
        ),
        "lat": (
            ("y", "x"),
            np.array(
                [
                    [9.475632, 9.692574, 9.909004],
                    [9.75074, 9.969017, 10.186775],
                    [10.025963, 10.245583, 10.464706],
                ]
            ),
        ),
    }
    return wrf_coords


@pytest.fixture
def wrf_coords_d03() -> dict:
    wrf_coords = {
        "y": (("y"), np.array([454911.730699, 457911.730699, 460911.730699])),
        "x": (("x"), np.array([-4335113.661861, -4332113.661861, -4329113.661861])),
        "lon": (
            ("y", "x"),
            np.array(
                [
                    [-117.80029, -117.774536, -117.74875],
                    [-117.81781, -117.79204, -117.76625],
                    [-117.83533, -117.809555, -117.78377],
                ]
            ),
        ),
        "lat": (
            ("y", "x"),
            np.array(
                [
                    [29.978943, 29.994099, 30.009262],
                    [30.001251, 30.016422, 30.031597],
                    [30.023571, 30.038742, 30.053913],
                ]
            ),
        ),
    }
    return wrf_coords


def _make_mock_wrf_angles_ds(wrf_coords: dict) -> xr.Dataset:
    """Helper to make fake WRF angles datasets with different grids."""
    cosdata = [
        [0.78479314, 0.7869512, 0.78911465],
        [0.78307813, 0.7852431, 0.78742146],
        [0.7813307, 0.7835257, 0.78571546],
    ]
    sindata = [
        [0.6197577, 0.6170152, 0.6142459],
        [0.6219233, 0.6191876, 0.61641496],
        [0.6241172, 0.6213593, 0.6185881],
    ]
    wrf_angles_ds = xr.Dataset(
        data_vars={
            "COSALPHA": (("y", "x"), cosdata),
            "SINALPHA": (("y", "x"), sindata),
        },
        coords=wrf_coords,
    )
    return wrf_angles_ds


@pytest.fixture
def mock_wrf_angles_ds_d01(wrf_coords_d01) -> xr.Dataset:
    return _make_mock_wrf_angles_ds(wrf_coords_d01)


@pytest.fixture
def mock_wrf_angles_ds_d03(wrf_coords_d03) -> xr.Dataset:
    return _make_mock_wrf_angles_ds(wrf_coords_d03)


class TestSeaLevelPressure:
    """Test class for compute_sea_level_pressure."""

    def test_defaults(self, mock_psfc, mock_t2, mock_q2, mock_elevation):
        """Test SLP calculation with default arguments for single point."""
        result = compute_sea_level_pressure(mock_psfc, mock_t2, mock_q2, mock_elevation)
        assert isinstance(result, xr.DataArray)
        # Since time averaging is default, the first valid value is time=11
        assert approx(result[0, 0, 12].data.item(), rel=1e-4) == 90060.53
        assert np.isnan(result[0, 0, 0:11]).all()

        # Check metadata
        assert result.attrs["units"] == "Pa"
        assert result.name == "slp_derived"

    def test_no_time_average(self, mock_psfc, mock_t2, mock_q2, mock_elevation):
        """Test slp calculation without 12-hour temperature mean."""
        result = compute_sea_level_pressure(
            mock_psfc, mock_t2, mock_q2, mock_elevation, average_t2=False
        )
        assert isinstance(result, xr.DataArray)

        # Should be no NaNs; because test data is constant in time, the
        # time-averaged result is same as non-time-averaged
        assert approx(result[0, 0, 0].data.item(), rel=1e-4) == 90060.53
        assert ~np.isnan(result).all()

    def test_lapse_rate(
        self, mock_psfc, mock_t2, mock_q2, mock_elevation, mock_lapse_rate
    ):
        """Test user-defined lapse rate."""
        # Lapse rate is float 7K/km
        result = compute_sea_level_pressure(
            mock_psfc, mock_t2, mock_q2, mock_elevation, lapse_rate=0.007
        )
        assert approx(result[0, 0, 11].data.item(), rel=1e-4) == 90058.33

        # Lapse rate is spatial data array 9K/km
        result = compute_sea_level_pressure(
            mock_psfc, mock_t2, mock_q2, mock_elevation, lapse_rate=mock_lapse_rate
        )
        assert approx(result[0, 0, 11].data.item(), rel=1e-4) == 90049.54

    def test_no_time_axis(
        self, mock_psfc, mock_t2_no_time, mock_q2, mock_elevation, mock_lapse_rate
    ):
        """Test that error is thrown when average_t2 is True with no time axis."""
        with pytest.raises(
            KeyError,
            match="No time or time_delta axis found in t2. Use `average_t2=False` for data without time axis.",
        ):
            result = compute_sea_level_pressure(
                mock_psfc, mock_t2_no_time, mock_q2, mock_elevation
            )

    def test_with_time_delta(
        self,
        mock_psfc_time_delta,
        mock_t2_time_delta,
        mock_q2_time_delta,
        mock_elevation,
        mock_lapse_rate,
    ):
        """Test that slp works with time_delta dimension."""
        result = compute_sea_level_pressure(
            mock_psfc_time_delta, mock_t2_time_delta, mock_q2_time_delta, mock_elevation
        )
        assert isinstance(result, xr.DataArray)
        # Since time averaging is default, the first valid value is time=11
        assert approx(result[0, 0, 12].data.item(), rel=1e-4) == 90060.53
        assert np.isnan(result[0, 0, 0:11]).all()


class TestGeostrophicWind:
    """Test class for geostrophic wind functions."""

    def test__wrf_deltas(self):
        """Test function to find distance between lat/lon points on grid."""
        da = xr.DataArray(
            data=np.ones((2, 2)),
            dims=["y", "x"],
            coords={
                "y": (("y"), np.array([0, 1])),
                "x": (("x"), np.array([0, 1])),
                "lon": (("y", "x"), np.array([[-117, -118], [-117, -118]])),
                "lat": (("y", "x"), np.array([[36, 36], [35, 35]])),
            },
        )
        dx, dy = deltas = _wrf_deltas(da)

        assert dx.shape == (2, 1)
        assert dy.shape == (1, 2)

        assert approx(dx[0].data, 1e-6) == 89958.1484979
        assert approx(dy[:, 1].data, 1e-6) == -111194.87428468

    def test__align_dim(self):
        """Test the _align_dim helper function with toy arrays."""
        da_to_copy = xr.DataArray(
            data=np.array([[0]]),
            dims=["y", "x"],
            coords={
                "y": (("y"), np.array([10])),
                "x": (("x"), np.array([10])),
                "lon": (("y", "x"), np.array([[-118.0]])),
                "lat": (("y", "x"), np.array([[35]])),
            },
        )
        da_shifted = xr.DataArray(
            data=np.array([[1]]),
            dims=["y", "x"],
            coords={
                "y": (("y"), np.array([11])),
                "x": (("x"), np.array([11])),
                "lon": (("y", "x"), np.array([[-118.6]])),
                "lat": (("y", "x"), np.array([[35.4]])),
            },
        )
        result = _align_dim(da_shifted, da_to_copy, "x")

        # Result should match original along selected x dim
        assert result.x.data == da_to_copy.x.data
        assert result.lat.data == da_to_copy.lat.data
        assert result.lon.data == da_to_copy.lon.data

    def test__get_spatial_derivatives(self):
        """Test the spatial derivatives function with a correctly sized array."""
        # Needs to be at least 3x3 for derivative to work
        da = xr.DataArray(
            data=np.array([[[1, 2, 3], [2, 3, 4], [4, 5, 6]]]),
            dims=["time", "y", "x"],
            coords={
                "time": (("time"), ["2020-01-01"]),
                "y": (("y"), np.array([0, 1, 2])),
                "x": (("x"), np.array([0, 1, 2])),
                "lon": (
                    ("y", "x"),
                    np.array(
                        [[-117, -118, -119], [-117, -118, -119], [-117, -118, -119]]
                    ),
                ),
                "lat": (
                    ("y", "x"),
                    np.array([[36, 36, 36], [35, 35, 35], [34, 34, 34]]),
                ),
            },
        )
        dx, dy = _get_spatial_derivatives(da)

        for derivative in [dx, dy]:
            assert derivative.dims == da.dims
            assert derivative.shape == da.shape
            assert (derivative.x == da.x).all()
            assert (derivative.y == da.y).all()

        # Our data decreases in the y direction
        # and increases in the x direction
        assert (dy.data < 0).all()
        assert (dx.data > 0).all()

    def test__get_spatial_derivatives_small_array(self):
        """Test spatial derivative with a too small array."""
        da = xr.DataArray(
            data=np.array([[1]]),
            dims=["y", "x"],
            coords={
                "y": (("y"), np.array([11])),
                "x": (("x"), np.array([11])),
                "lon": (("y", "x"), np.array([[-118.6]])),
                "lat": (("y", "x"), np.array([[35.4]])),
            },
        )
        with pytest.raises(
            ValueError,
            match="Spatial derivative requires a minimum length of 3 on both x and y dimensions.",
        ):
            _, _ = _get_spatial_derivatives(da)

    def test__get_rotated_geostrophic_wind_d01(
        self, wrf_coords_d01, mock_wrf_angles_ds_d01
    ):
        """Test wind rotation with correct gridlabel."""
        coords = {"time": (("time"), ["2020-01-01"])}
        coords.update(wrf_coords_d01)
        u_wrf = xr.DataArray(
            data=np.ones((1, 3, 3)), dims=["time", "y", "x"], coords=coords
        )
        v_wrf = u_wrf
        with patch("xarray.open_zarr", return_value=mock_wrf_angles_ds_d01):
            u_rot, v_rot = _get_rotated_geostrophic_wind(u_wrf, v_wrf, "d01")

        assert approx(u_rot.data[0, 1, 1], 1e-6) == 0.1660555
        assert approx(v_rot.data[0, 1, 1], 1e-6) == 1.4044307

    def test__get_rotated_geostrophic_wind_d03(
        self, wrf_coords_d01, mock_wrf_angles_ds_d03
    ):
        """Test that error raised when grid label does not align with u/v grid type."""
        coords = {"time": (("time"), ["2020-01-01"])}
        coords.update(wrf_coords_d01)
        u_wrf = xr.DataArray(
            data=np.ones((1, 3, 3)), dims=["time", "y", "x"], coords=coords
        )
        v_wrf = u_wrf
        with patch("xarray.open_zarr", return_value=mock_wrf_angles_ds_d03):
            with pytest.raises(
                ValueError,
                match="Cannot multiply wind array by WRF angles array. This is likely due to the `gridlabel` parameter not matching the u and v grid type.",
            ):
                u_rot, v_rot = _get_rotated_geostrophic_wind(u_wrf, v_wrf, "d03")

    def test_compute_geostrophic_wind(self, wrf_coords_d01, mock_wrf_angles_ds_d01):
        """Test the geostrophic wind function with a small array."""
        # Here we've set up with actual coordinates from the d01 grid.
        coords = {"time": (("time"), ["2020-01-01"])}
        coords.update(wrf_coords_d01)
        da = xr.DataArray(
            data=np.array([[[1, 2, 3], [2, 3, 4], [4, 5, 6]]]),
            dims=["time", "y", "x"],
            coords=coords,
        )
        with patch("xarray.open_zarr", return_value=mock_wrf_angles_ds_d01):
            u, v = compute_geostrophic_wind(da)

        # Check that results are the right shape
        for component in [u, v]:
            assert component.dims == da.dims
            assert component.shape == da.shape
            assert (component.time == da.time).all()
            assert (component.y == da.y).all()

        # Spot check center values
        assert approx(u.data[0, 1, 1], 1e-6) == -17.82301021
        assert approx(v.data[0, 1, 1], 1e-6) == -1.41620402
