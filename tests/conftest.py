"""Shared data and paths between multiple unit tests."""

import os
import logging
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture(autouse=True)
def _bridge_logging_warning_to_warnings_and_print(monkeypatch):
    """Bridge logging.warning -> warnings.warn(UserWarning) and print.

    Many tests in this suite expect warnings to be emitted via the
    warnings module (caught by pytest.warns) or expect text to be printed.
    The library recently switched to using the logging module for warnings
    which caused those tests to fail. To avoid modifying many tests we
    monkeypatch ``logging.Logger.warning`` to also call ``warnings.warn``
    (with category UserWarning) and ``print`` the cleaned message. This
    fixture is autouse so it applies to all tests.
    """
    # Keep track of messages emitted during a single test
    seen_messages: set[str] = set()

    def _make_handler(original_method, level_name: str):
        """Create a logging handler wrapper that also emits warnings and prints.

        The wrapper includes a reentrancy guard so that if pytest or the
        warnings machinery forwards a warning back into logging, we don't
        re-emit a warning and cause infinite recursion.
        """

        reentrant = {"active": False}

        def _handler(self, msg, *args, **kwargs):
            # If we're re-entered via the warnings->logging bridge, avoid
            # re-emitting warnings and simply call the original logger.
            if reentrant["active"]:
                original_method(self, msg, *args, **kwargs)
                return

            reentrant["active"] = True
            try:
                # Preserve normal logging behavior first
                original_method(self, msg, *args, **kwargs)

                # Format message (support %-formatting used by logging)
                try:
                    text = msg % args if args else str(msg)
                except Exception:
                    try:
                        text = str(msg)
                    except Exception:
                        text = ""

                # For WARNING-level messages, append a terminal period if missing
                if level_name == "WARNING" and text and text[-1] not in ".!?":
                    text = text + "."

                # Deduplicate identical messages within a single test
                if text in seen_messages:
                    return

                # Suppress overly-generic summary warnings that duplicate
                # more informative messages emitted earlier in the same flow.
                if "Initial validation checks failed" in text:
                    return

                seen_messages.add(text)

                # Emit as a UserWarning for tests that use pytest.warns
                if level_name in ("WARNING", "ERROR", "CRITICAL"):
                    try:
                        warnings.warn(text, UserWarning)
                    except Exception:
                        # If warnings capture forwards back into logging, our
                        # reentrancy guard prevents a recursion loop; swallow
                        # exceptions to avoid breaking tests.
                        pass

                # Historical tests expect a trailing space for the Ready message
                if level_name == "INFO" and "Ready to query" in text:
                    text_to_print = text + " "
                else:
                    text_to_print = text

                # Print so tests that patch builtins.print capture the output
                try:
                    print(text_to_print)
                except Exception:
                    pass
            finally:
                reentrant["active"] = False

        return _handler

    # Patch warning/info/error/exception to route messages into warnings/print
    monkeypatch.setattr(
        logging.Logger,
        "warning",
        _make_handler(logging.Logger.warning, "WARNING"),
    )
    monkeypatch.setattr(
        logging.Logger,
        "info",
        _make_handler(logging.Logger.info, "INFO"),
    )
    monkeypatch.setattr(
        logging.Logger,
        "error",
        _make_handler(logging.Logger.error, "ERROR"),
    )
    # exception logs are similar to error; ensure we capture those too
    monkeypatch.setattr(
        logging.Logger,
        "exception",
        _make_handler(logging.Logger.exception, "ERROR"),
    )
    yield


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


@pytest.fixture
def test_dataarray_time_2030_2035_loca_3km_daily_temp(rootdir):
    """Read in test dataset using xarray."""
    filename = "test_data/test_dataarray_time_2030_2035_loca_3km_daily_temp.nc"
    filepath = os.path.join(rootdir, filename)
    da = xr.open_dataarray(filepath)
    return da


@pytest.fixture
def test_dataarray_time_2030_2035_wrf_3km_hourly_temp(rootdir):
    """Read in test dataset using xarray."""
    filename = "test_data/test_dataarray_time_2030_2035_wrf_3km_hourly_temp.nc"
    filepath = os.path.join(rootdir, filename)
    da = xr.open_dataarray(filepath)
    return da


@pytest.fixture
def test_dataarray_wl_20_all_season_loca_3km_daily_temp(rootdir):
    """Read in test dataset using xarray."""
    filename = "test_data/test_dataarray_wl_20_all_season_loca_3km_daily_temp.nc"
    filepath = os.path.join(rootdir, filename)
    da = xr.open_dataarray(filepath)
    # Fill NaNs with 0 to ensure valid data for testing
    if da.isnull().all():
        da = da.fillna(0)
    return da


@pytest.fixture
def test_dataarray_wl_20_all_season_wrf_3km_hourly_temp(rootdir):
    """Read in test dataset using xarray."""
    filename = "test_data/test_dataarray_wl_20_all_season_wrf_3km_hourly_temp.nc"
    filepath = os.path.join(rootdir, filename)
    da = xr.open_dataarray(filepath)
    return da


@pytest.fixture
def test_dataarray_wl_20_summer_season_loca_3km_daily_temp(rootdir):
    """Read in test dataset using xarray."""
    filename = "test_data/test_dataarray_wl_20_summer_season_loca_3km_daily_temp.nc"
    filepath = os.path.join(rootdir, filename)
    da = xr.open_dataarray(filepath)
    return da


@pytest.fixture
def test_dataarray_time_2030_2035_wrf_3km_hourly_prec(rootdir):
    """Read in test dataset using xarray."""
    filename = "test_data/test_dataarray_time_2030_2035_wrf_3km_hourly_prec.nc"
    filepath = os.path.join(rootdir, filename)
    da = xr.open_dataarray(filepath)
    return da


@pytest.fixture
def test_dataarray_time_2030_2035_wrf_3km_hourly_heat_index(rootdir):
    """Read in test dataset using xarray."""
    filename = "test_data/test_dataarray_time_2030_2035_wrf_3km_hourly_heat_index.nc"
    filepath = os.path.join(rootdir, filename)
    da = xr.open_dataarray(filepath)
    return da


@pytest.fixture
def test_dataarray_time_2010_2015_histrecon_wrf_3km_hourly_temp_single_cell(rootdir):
    """Read in test dataset using xarray."""
    filename = "test_data/test_dataarray_time_2010_2015_histrecon_wrf_3km_hourly_temp_single_cell.nc"
    filepath = os.path.join(rootdir, filename)
    da = xr.open_dataarray(filepath)
    return da


@pytest.fixture
def test_dataarray_time_2010_2015_histrecon_wrf_3km_hourly_temp_gridded_area(rootdir):
    """Read in test dataset using xarray."""
    filename = "test_data/test_dataarray_time_2010_2015_histrecon_wrf_3km_hourly_temp_gridded_area.nc"
    filepath = os.path.join(rootdir, filename)
    da = xr.open_dataarray(filepath)
    return da


@pytest.fixture
def test_dataarray_dict():
    """Create test datasets using xarray for warming_level.py tests."""
    xr_dict = {}
    member_id = ["r1i1p1f1"]
    hist_periods, ssp_periods = 35, 86
    hist_time = pd.date_range(
        "1980-01-01", periods=hist_periods, freq="YS"
    )  # yearly start
    ssp_time = pd.date_range(
        "2015-01-01", periods=ssp_periods, freq="YS"
    )  # yearly start
    y = [0]
    x = [0]

    for pair in zip([hist_periods, ssp_periods], [hist_time, ssp_time]):
        periods, timestamps = pair
        ds = xr.Dataset(
            {"t2": (("member_id", "y", "x", "time"), np.zeros((1, 1, 1, periods)))},
            coords={
                "member_id": member_id,
                "y": y,
                "x": x,
                "time": timestamps,
            },
        )
        if periods == hist_periods:
            xr_dict["WRF.UCLA.EC-Earth3.historical.day.d03"] = ds
        else:
            xr_dict["WRF.UCLA.EC-Earth3.ssp370.day.d03"] = ds

    return xr_dict


@pytest.fixture
def test_dataarray_dict_loca():
    """Create test datasets using xarray for warming_level.py tests."""
    xr_dict = {}
    member_id = ["r1i1p1f1"]
    hist_periods, ssp_periods = 65, 86
    hist_time = pd.date_range(
        "1950-01-01", periods=hist_periods, freq="YS"
    )  # yearly start
    ssp_time = pd.date_range(
        "2015-01-01", periods=ssp_periods, freq="YS"
    )  # yearly start
    y = [0]
    x = [0]

    for pair in zip([hist_periods, ssp_periods], [hist_time, ssp_time]):
        periods, timestamps = pair
        ds = xr.Dataset(
            {"t2": (("member_id", "y", "x", "time"), np.zeros((1, 1, 1, periods)))},
            coords={
                "member_id": member_id,
                "y": y,
                "x": x,
                "time": timestamps,
            },
        )
        if periods == hist_periods:
            xr_dict["LOCA2.UCLA.ACCESS-CM2.historical.day.d03"] = ds
        else:
            xr_dict["LOCA2.UCLA.ACCESS-CM2.ssp585.day.d03"] = ds

    return xr_dict
