import pytest
import os
import datetime as dt
import numpy as np
import xarray as xr
import climakitae.explore.timeseries as tst

# -------- Read in the test dataset and return a TimeSeriesParams object -------


@pytest.fixture
def test_TSP(rootdir: str) -> tst.TimeSeriesParameters:
    # This data is generated in "create_timeseries_test_data.py"
    test_filename = "test_data/timeseries_data_T2_2014_2016_monthly_45km.nc"
    test_filepath = os.path.join(rootdir, test_filename)
    test_data = xr.open_dataset(test_filepath).T2
    # Resample to daily frequency for single simulation
    test_data = (
        test_data.sel({"simulation": "cesm2"}).resample(time="1D").interpolate("linear")
    )
    test_data.attrs["frequency"] = "daily"

    # Compute area average
    weights = np.cos(np.deg2rad(test_data.lat))
    test_data = test_data.weighted(weights).mean("x").mean("y")

    # Adding in some daily variability
    rng = np.random.default_rng()
    n = rng.uniform(-5, 5, (len(test_data.time),))
    test_data = test_data + n

    ts = tst.TimeSeries(test_data)  # make Timeseries object
    return ts.choices  # return the underlying TimeSeriesParams object for testing


def test_daily_smoothing(test_TSP: tst.TimeSeriesParameters):
    # Specify Params options
    test_TSP.smoothing = "Running Mean"
    test_TSP.num_timesteps = 3
    test_TSP.anomaly = False

    # Transform data and test
    result = test_TSP.transform_data()  # transform_data calls _running_mean()
    assert (result == test_TSP.data).sum().values.item() == 0


# ------------- Test anomaly and smoothing together ----------------------------


def test_daily_anomaly_and_smoothing(test_TSP: tst.TimeSeriesParameters):
    # Specify Params options
    test_TSP.smoothing = "Running Mean"
    test_TSP.num_timesteps = 3
    test_TSP.anomaly = True
    test_TSP.reference_range = (dt.datetime(2014, 1, 1), dt.datetime(2014, 12, 31))

    # Transform data and test
    result = test_TSP.transform_data()
    assert (result == test_TSP.data).sum().values.item() == 0
