import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def wrf_dataset():
    """Synthetic WRF-like dataset with t2, t2max, t2min, u10, v10, rh."""
    time = pd.date_range("2020-01-01", periods=3)
    shape = (3, 2, 2)

    t2 = 290.0 + np.arange(shape[0]).reshape(shape[0], 1, 1) * 2.0
    t2 = t2 + np.zeros(shape)
    t2max = t2 + 5.0
    t2min = t2 - 5.0

    u10 = np.ones(shape) * 2.0
    v10 = np.ones(shape) * 1.0
    rh = np.ones(shape) * 50.0

    ds = xr.Dataset(
        {
            "t2": (("time", "y", "x"), t2),
            "t2max": (("time", "y", "x"), t2max),
            "t2min": (("time", "y", "x"), t2min),
            "u10": (("time", "y", "x"), u10),
            "v10": (("time", "y", "x"), v10),
            "rh": (("time", "y", "x"), rh),
        },
        coords={"time": time, "y": np.arange(2), "x": np.arange(2)},
    )
    return ds


@pytest.fixture
def loca_dataset():
    """Synthetic LOCA-like dataset with tasmax and tasmin."""
    time = pd.date_range("2020-01-01", periods=3)
    shape = (3, 2, 2)

    tasmax = 295.0 + np.arange(shape[0]).reshape(shape[0], 1, 1) * 1.5
    tasmax = tasmax + np.zeros(shape)
    tasmin = tasmax - 10.0

    ds = xr.Dataset(
        {
            "tasmax": (("time", "lat", "lon"), tasmax),
            "tasmin": (("time", "lat", "lon"), tasmin),
        },
        coords={"time": time, "lat": np.linspace(32, 42, 2), "lon": np.linspace(-125, -114, 2)},
    )
    return ds


@pytest.fixture
def wind_dataset():
    time = pd.date_range("2020-01-01", periods=3)
    shape = (3, 2, 2)
    u10 = np.linspace(-2.0, 2.0, num=np.prod(shape)).reshape(shape)
    v10 = np.linspace(1.0, -1.0, num=np.prod(shape)).reshape(shape)

    ds = xr.Dataset({"u10": (("time", "y", "x"), u10), "v10": (("time", "y", "x"), v10)}, coords={"time": time, "y": np.arange(2), "x": np.arange(2)})
    return ds


@pytest.fixture
def humidity_dataset():
    time = pd.date_range("2020-01-01", periods=3)
    shape = (3, 2, 2)

    t2 = 295.0 + np.zeros(shape)
    q2 = np.ones(shape) * 0.008
    psfc = np.ones(shape) * 100000.0
    rh = np.ones(shape) * 50.0

    ds = xr.Dataset(
        {
            "t2": (("time", "y", "x"), t2),
            "q2": (("time", "y", "x"), q2),
            "psfc": (("time", "y", "x"), psfc),
            "rh": (("time", "y", "x"), rh),
        },
        coords={"time": time, "y": np.arange(2), "x": np.arange(2)},
    )
    return ds
