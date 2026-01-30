"""Test functions for derived variables that are not loadable via get_data or new core."""

import numpy as np
import pytest
import xarray as xr
from pytest import approx

from climakitae.tools.derived_variables import compute_sea_level_pressure


@pytest.fixture
def mock_psfc() -> xr.DataArray:
    # Single grid point surface pressure
    lon = -123.521255
    lat = 9.475632
    time_range = xr.date_range("2000-01-01 00:00", "2000-01-01 23:00", freq="h")
    da = xr.DataArray(
        data=np.expand_dims(np.tile([[850]], 24), 0),
        dims=["y", "x", "time"],
        coords={
            "lon": (("y", "x"), np.array([[lon]])),
            "lat": (("y", "x"), np.array([[lat]])),
            "time": (("time"), time_range),
        },
    )
    da.name = "surface pressure"
    da.attrs = {"units": "Pa", "frequency": "hourly"}
    return da


@pytest.fixture
def mock_t2(frequency: str = "hourly") -> xr.DataArray:
    # Single grid point air temperature
    lon = -123.521255
    lat = 9.475632
    time_range = xr.date_range("2000-01-01 00:00", "2000-01-01 23:00", freq="h")
    da = xr.DataArray(
        data=np.expand_dims(np.tile([[293.15]], 24), 0),
        dims=["y", "x", "time"],
        coords={
            "lon": (("y", "x"), np.array([[lon]])),
            "lat": (("y", "x"), np.array([[lat]])),
            "time": (("time"), time_range),
        },
    )
    da.name = "Air Temperature at 2m"
    da.attrs = {"units": "K", "frequency": frequency}
    if frequency == "":
        da.attrs.pop("frequency")
    return da


@pytest.fixture
def mock_q2() -> xr.DataArray:
    # Single grid point mixing ratio
    lon = -123.521255
    lat = 9.475632
    time_range = xr.date_range("2000-01-01 00:00", "2000-01-01 23:00", freq="h")
    da = xr.DataArray(
        data=np.expand_dims(np.tile([[0.04]], 24), 0),
        dims=["y", "x", "time"],
        coords={
            "lon": (("y", "x"), np.array([[lon]])),
            "lat": (("y", "x"), np.array([[lat]])),
            "time": (("time"), time_range),
        },
    )
    da.name = "Water Vapor Mixing Ratio at 2m"
    da.attrs = {"units": "kg kg-1", "frequency": "hourly"}
    return da


@pytest.fixture
def mock_elevation() -> xr.DataArray:
    # Single grid point elevation
    lon = -123.521255
    lat = 9.475632
    da = xr.DataArray(
        data=np.array([[500]]),
        dims=["y", "x"],
        coords={
            "lon": (("y", "x"), np.array([[lon]])),
            "lat": (("y", "x"), np.array([[lat]])),
        },
    )
    da.name = "elevation"
    da.attrs = {"units": "m"}
    return da


@pytest.fixture
def mock_lapse_rate() -> xr.DataArray:
    # Single grid point lapse rate as array
    lon = -123.521255
    lat = 9.475632
    da = xr.DataArray(
        data=np.array([[0.009]]),
        dims=["y", "x"],
        coords={
            "lon": (("y", "x"), np.array([[lon]])),
            "lat": (("y", "x"), np.array([[lat]])),
        },
    )
    da.name = "elevation"
    da.attrs = {"units": "m"}
    return da


class TestSeaLevelPressure:

    def test_defaults(self, mock_psfc, mock_t2, mock_q2, mock_elevation):
        """Test SLP calculation with default arguments for single point."""
        result = compute_sea_level_pressure(mock_psfc, mock_t2, mock_q2, mock_elevation)
        assert isinstance(result, xr.DataArray)
        # Since time averaging is default, the first valid value is time=11
        assert approx(result[0, 0, 12].data.item(), rel=1e-7) == 899.54320333
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
        assert approx(result[0, 0, 0].data.item(), rel=1e-7) == 899.54320333
        assert ~np.isnan(result).all()

    def test_lapse_rate(
        self, mock_psfc, mock_t2, mock_q2, mock_elevation, mock_lapse_rate
    ):
        """Test user-defined lapse rate."""
        # Lapse rate is float 7K/km
        result = compute_sea_level_pressure(
            mock_psfc, mock_t2, mock_q2, mock_elevation, lapse_rate=0.007
        )
        assert approx(result[0, 0, 11].data.item(), rel=1e-7) == 899.52209463

        # Lapse rate is spatial data array 9K/km
        result = compute_sea_level_pressure(
            mock_psfc, mock_t2, mock_q2, mock_elevation, lapse_rate=mock_lapse_rate
        )
        assert approx(result[0, 0, 11].data.item(), rel=1e-7) == 899.43783935

    @pytest.mark.parametrize("mock_t2", ["", "day"], indirect=True)
    def test_bad_time_frequency(self, mock_psfc, mock_t2, mock_q2, mock_elevation):
        """Test SLP calculation with non-hourly time frequencies and check that averaging is skipped."""
        result = compute_sea_level_pressure(mock_psfc, mock_t2, mock_q2, mock_elevation)
        # There should be no NaN values because time was not averaged.
        assert ~np.isnan(result).all()
