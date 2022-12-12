"""This script tests that the functions used to compute derived variables perform as expected. """

import xarray as xr
import numpy as np
import pytest
import os
from climakitae.derive_variables import (
    _compute_relative_humidity,
    _compute_wind_mag,
    _compute_dewpointtemp
)


@pytest.fixture
def rel_humidity(test_data_2022_monthly_45km):
    """Compute relative humidity and return data"""
    da = _compute_relative_humidity(
        pressure = test_data_2022_monthly_45km["PSFC"],
        temperature = test_data_2022_monthly_45km["T2"],
        mixing_ratio = test_data_2022_monthly_45km["Q2"],
    )
    return da


@pytest.fixture
def wind_mag(test_data_2022_monthly_45km):
    """Compute relative humidity and return data"""
    da = _compute_wind_mag(
        u10 = test_data_2022_monthly_45km["U10"], 
        v10 = test_data_2022_monthly_45km["V10"]
    )
    return da

@pytest.fixture
def dew_pnt(rel_humidity, test_data_2022_monthly_45km):
    """Compute relative humidity and return data""" 
    da = _compute_dewpointtemp(
        temperature = test_data_2022_monthly_45km["T2"],
        rel_hum = rel_humidity,
    )
    return da

def test_expected_data_name(rel_humidity, wind_mag, dew_pnt):
    """Ensure that xr.DataArray has the correct assigned name"""
    assert rel_humidity.name == "rh_derived"
    assert wind_mag.name == "wind_speed_derived"
    assert dew_pnt.name == "dew_point_derived"

def test_expected_return_type(rel_humidity, wind_mag, dew_pnt):
    """Ensure function returns an xr.DataArray object"""
    assert type(rel_humidity) == xr.core.dataarray.DataArray
    assert type(wind_mag) == xr.core.dataarray.DataArray
    assert type(dew_pnt) == xr.core.dataarray.DataArray
