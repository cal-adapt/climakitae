"""This script tests that the functions used to derive indices perform as expected."""

import pytest
import xarray as xr

from climakitae.tools.derived_variables import (
    compute_relative_humidity,
    compute_wind_mag,
)
from climakitae.tools.indices import effective_temp, fosberg_fire_index, noaa_heat_index
from climakitae.util.unit_conversions import convert_units


class TestNOAAHeatIndex:
    @pytest.fixture
    def NOAA_heat_index(self, test_dataset_01Jan2015_LAcounty_45km_hourly):
        """Derive NOAA heat index from hourly data"""

        # Parse out variables from the Dataset
        p_Pa = test_dataset_01Jan2015_LAcounty_45km_hourly["Surface Pressure"]
        t_K = test_dataset_01Jan2015_LAcounty_45km_hourly["Air Temperature at 2m"]
        q2 = test_dataset_01Jan2015_LAcounty_45km_hourly[
            "Water Vapor Mixing Ratio at 2m"
        ]

        # Derive relative humidity
        # Returned in units of [0-100]
        rh_da = compute_relative_humidity(
            pressure=p_Pa, temperature=t_K, mixing_ratio=q2
        )

        # Convert units to proper units
        t2_da_F = convert_units(t_K, "degF")

        # Derive index
        # Returned in units of F
        da = noaa_heat_index(T=t2_da_F, RH=rh_da)
        return da

    def test_expected_return_type(self, NOAA_heat_index):
        """Ensure function returns an xr.DataArray object"""
        assert type(NOAA_heat_index) == xr.core.dataarray.DataArray

    def test_units(self, NOAA_heat_index):
        """Ensure output data has the proper units attribute"""
        assert NOAA_heat_index.attrs["units"] == "degF"


class TestEffectiveTemperature:
    @pytest.fixture
    def eft(self, test_dataset_Jan2015_LAcounty_45km_daily):
        """Derive effective temp from daily air temperature"""
        T = test_dataset_Jan2015_LAcounty_45km_daily["Air Temperature at 2m"]
        da = effective_temp(T)
        return da

    def test_expected_return_type(self, eft):
        """Ensure function returns an xr.DataArray object"""
        assert type(eft) == xr.core.dataarray.DataArray

    def test_units(self, eft, test_dataset_Jan2015_LAcounty_45km_daily):
        """Ensure output data has the proper units attribute"""
        assert (
            eft.attrs["units"]
            == test_dataset_Jan2015_LAcounty_45km_daily["Air Temperature at 2m"].attrs[
                "units"
            ]
        )


class TestFosbergFireIndex:
    @pytest.fixture
    def ffi(self, test_dataset_01Jan2015_LAcounty_45km_hourly):
        """Derive fosberg fire index from hourly data"""

        # Parse out variables from the Dataset
        p_Pa = test_dataset_01Jan2015_LAcounty_45km_hourly["Surface Pressure"]
        t_K = test_dataset_01Jan2015_LAcounty_45km_hourly["Air Temperature at 2m"]
        q2 = test_dataset_01Jan2015_LAcounty_45km_hourly[
            "Water Vapor Mixing Ratio at 2m"
        ]
        u10 = test_dataset_01Jan2015_LAcounty_45km_hourly[
            "West-East component of Wind at 10m"
        ]
        v10 = test_dataset_01Jan2015_LAcounty_45km_hourly[
            "North-South component of Wind at 10m"
        ]

        # Derive relative humidity
        # Returned in units of [0-100]
        rh_da = compute_relative_humidity(
            pressure=p_Pa, temperature=t_K, mixing_ratio=q2
        )

        # Derive windspeed
        # Returned in units of m/s
        windspeed_da_ms = compute_wind_mag(u10=u10, v10=v10)

        # Convert units to proper units for fosberg index
        t2_da_F = convert_units(t_K, "degF")
        windspeed_da_mph = convert_units(windspeed_da_ms, "mph")

        # Compute the index
        da = fosberg_fire_index(
            t2_F=t2_da_F, rh_percent=rh_da, windspeed_mph=windspeed_da_mph
        )
        return da

    def test_expected_return_type(self, ffi):
        """Ensure function returns an xr.DataArray object"""
        assert type(ffi) == xr.core.dataarray.DataArray

    def test_units(self, ffi):
        """Ensure output data has the proper units attribute"""
        assert ffi.attrs["units"] == "[0 to 100]"
