"""This script tests that the functions used to compute derived variables perform as expected."""

import pytest
import xarray as xr

from climakitae.tools.derived_variables import (
    compute_dewpointtemp,
    compute_relative_humidity,
    compute_specific_humidity,
    compute_wind_dir,
    compute_wind_mag,
)


@pytest.fixture
def rel_humidity(test_data_2022_monthly_45km):
    """Compute relative humidity and return data"""
    da = compute_relative_humidity(
        pressure=test_data_2022_monthly_45km["PSFC"],
        temperature=test_data_2022_monthly_45km["T2"],
        mixing_ratio=test_data_2022_monthly_45km["Q2"],
    )
    return da


@pytest.fixture
def wind_mag(test_data_2022_monthly_45km):
    """Compute wind magnitude and return data"""
    da = compute_wind_mag(
        u10=test_data_2022_monthly_45km["U10"], v10=test_data_2022_monthly_45km["V10"]
    )
    return da


@pytest.fixture
def wind_dir(test_data_2022_monthly_45km):
    """Compute wind direction and return data"""
    da = xr.apply_ufunc(
        compute_wind_dir,
        u10=test_data_2022_monthly_45km["U10"],
        v10=test_data_2022_monthly_45km["V10"],
    )
    return da


@pytest.fixture
def dew_pnt(rel_humidity, test_data_2022_monthly_45km):
    """Compute dew point temp and return data"""
    da = compute_dewpointtemp(
        temperature=test_data_2022_monthly_45km["T2"],
        rel_hum=rel_humidity,
    )
    return da


@pytest.fixture
def spec_humidity(dew_pnt, test_data_2022_monthly_45km):
    """Compute specific humidity and return data"""
    da = compute_specific_humidity(
        tdps=dew_pnt,
        pressure=test_data_2022_monthly_45km["PSFC"],
    )
    return da


def test_expected_return_type(rel_humidity, wind_mag, dew_pnt, spec_humidity):
    """Ensure function returns an xr.DataArray object"""
    assert type(rel_humidity) == xr.core.dataarray.DataArray
    assert type(wind_mag) == xr.core.dataarray.DataArray
    assert type(dew_pnt) == xr.core.dataarray.DataArray
    assert type(spec_humidity) == xr.core.dataarray.DataArray
