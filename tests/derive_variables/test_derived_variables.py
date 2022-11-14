"""This script tests that the functions used to compute derived variables perform as expected. """

import xarray as xr
import numpy as np
import pytest
import os
from climakitae.derive_variables import (
    _compute_total_precip,
    _compute_relative_humidity,
    _compute_wind_mag,
)


@pytest.fixture
def total_precip(test_data_2022_monthly_45km):
    """Compute total precipitation and return data"""
    da = _compute_total_precip(
        cumulus_precip=test_data_2022_monthly_45km["RAINC"],
        gridcell_precip=test_data_2022_monthly_45km["RAINNC"],
    )
    return da


@pytest.fixture
def rel_humidity(test_data_2022_monthly_45km):
    """Compute relative humidity and return data"""
    da = _compute_relative_humidity(
        pressure=test_data_2022_monthly_45km["PSFC"],
        temperature=test_data_2022_monthly_45km["T2"],
        mixing_ratio=test_data_2022_monthly_45km["Q2"],
    )
    return da


@pytest.fixture
def wind_mag(test_data_2022_monthly_45km):
    """Compute relative humidity and return data"""
    da = _compute_wind_mag(
        u10=test_data_2022_monthly_45km["U10"], v10=test_data_2022_monthly_45km["V10"]
    )
    return da


def test_expected_data_name(total_precip, rel_humidity, wind_mag):
    """Ensure that xr.DataArray has the correct assigned name"""
    assert total_precip.name == "TOT_PRECIP"
    assert rel_humidity.name == "REL_HUMIDITY"
    assert wind_mag.name == "WIND_MAG"


def test_expected_return_type(total_precip, rel_humidity, wind_mag):
    """Ensure function returns an xr.DataArray object"""
    assert type(total_precip) == xr.core.dataarray.DataArray
    assert type(rel_humidity) == xr.core.dataarray.DataArray
    assert type(wind_mag) == xr.core.dataarray.DataArray


def test_expected_attributes(total_precip, wind_mag):
    """Ensure that function output contains expected descriptive attributes"""
    """Relative humidity does not return attributes at the moment"""
    expected_attributes = [
        "units",
        "description",
    ]  # Attributes we expect the output of each function to contain
    assert all([attr in total_precip.attrs.keys() for attr in expected_attributes])
    # assert all([attr in rel_humidity.attrs.keys() for attr in expected_attributes])
    assert all([attr in wind_mag.attrs.keys() for attr in expected_attributes])


def test_precip_check_expected_value(test_data_2022_monthly_45km):
    """Ensure _compute_total_precip function returns expected value"""
    test_data_subset = test_data_2022_monthly_45km.sel(
        simulation="cesm2",
        scenario="SSP 2-4.5 -- Middle of the Road",
        time="2022-01-01",
    )  # Get small subset of testing data
    total_precip = _compute_total_precip(
        cumulus_precip=test_data_subset["RAINC"],
        gridcell_precip=test_data_subset["RAINNC"],
    )
    expected_value = (
        test_data_subset["RAINC"] + test_data_subset["RAINNC"]
    )  # Value you expect the function to return
    assert total_precip.equals(expected_value)
