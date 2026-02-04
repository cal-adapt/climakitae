"""Test functions for derived variables that are not loadable via get_data or new core.

Currently, only compute_sea_level_pressure() needs to be tested here.
"""

import numpy as np
import pytest
import xarray as xr
from pytest import approx

from climakitae.tools.derived_variables import compute_sea_level_pressure


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
