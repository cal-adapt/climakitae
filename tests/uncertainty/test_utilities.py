"""
Tests for the pure utility functions in the uncertainty module.
"""

import datetime

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.core.data_interface import DataParameters
from climakitae.explore.uncertainty import (  # get_ks_pval_df, # TODO I think this function is broken
    _calendar_align,
    _cf_to_dt,
    get_warm_level,
)


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


def test_get_warm_level():
    """Test get_warm_level function."""

    # catch exception for bad warm_level type
    with pytest.raises(ValueError) as excinfo:
        get_warm_level("bad", None, ipcc=False)
    assert str(excinfo.value) == "Please specify warming level as an integer or float."

    # catch exception for warm_level not accepted value
    with pytest.raises(ValueError) as excinfo:
        get_warm_level(5.0, None, ipcc=False)
    assert (
        str(excinfo.value)
        == "Specified warming level is not valid. Options are: 1.5, 2.0, 3.0, 4.0"
    )

    # catch exception for bad dataset type


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
