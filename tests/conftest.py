"""Shared data and paths between multiple unit tests. """

import os
import pytest
import xarray as xr


@pytest.fixture
def rootdir():
    """Add path to test data as fixture."""
    return os.path.dirname(os.path.abspath("tests/test_data"))


@pytest.fixture
def test_data_2022_monthly_45km(rootdir):
    """Read in test dataset using xarray."""
    filename = "test_data/test_dataset_2022_2022_monthly_45km.nc"
    filepath = os.path.join(rootdir, filename)
    ds = xr.open_dataset(filepath)
    return ds


@pytest.fixture
def T2_hourly(rootdir):
    """Small hourly temperature data set"""
    test_filename = "test_data/threshold_data_T2_2050_2051_hourly_45km.nc"
    test_filepath = os.path.join(rootdir, test_filename)
    da = xr.open_dataset(test_filepath)["Air Temperature at 2m"]
    da.attrs["frequency"] = "hourly"
    return da


@pytest.fixture
def test_dataset_Jan2015_LAcounty_45km_daily(rootdir):
    """Read in test dataset using xarray."""
    filename = "test_data/test_dataset_Jan2015_LAcounty_45km_daily.nc"
    filepath = os.path.join(rootdir, filename)
    ds = xr.open_dataset(filepath)
    return ds


@pytest.fixture
def test_dataset_01Jan2015_LAcounty_45km_hourly(rootdir):
    """Read in test dataset using xarray."""
    filename = "test_data/test_dataset_01Jan2015_LAcounty_45km_hourly.nc"
    filepath = os.path.join(rootdir, filename)
    ds = xr.open_dataset(filepath)
    return ds
