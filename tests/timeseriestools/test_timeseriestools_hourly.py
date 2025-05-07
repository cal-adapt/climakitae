"""
This script contains tests on various Timeseries Tools options using hourly
data. For now, the tests only test that the various parameter combinations can
run through `transform_data` without error, and that the data has been transformed
(not equal to original values), but does not test for exact expected values.
"""

import os

import numpy as np
import pytest
import xarray as xr

import climakitae.explore.timeseries as tst


@pytest.fixture
def test_TSP(rootdir: str) -> tst.TimeSeriesParameters:
    """Read in the test dataset and return a TimeSeriesParams object."""
    # This data is generated in "create_timeseries_test_data.py"
    test_filename = "test_data/timeseries_data_T2_2014_2016_monthly_45km.nc"
    test_filepath = os.path.join(rootdir, test_filename)
    test_data = xr.open_dataset(test_filepath).T2
    # Resample to hourly frequency for single simulation
    test_data = (
        test_data.sel({"simulation": "cesm2"})
        .sel(time=slice("2014-01-01", "2014-01-31"))
        .resample(time="1H")
        .interpolate("linear")
    )
    test_data.attrs["frequency"] = "hourly"

    # Compute area average
    weights = np.cos(np.deg2rad(test_data.lat))
    test_data = test_data.weighted(weights).mean("x").mean("y")

    # Adding in some hourly variability
    rng = np.random.default_rng()
    n = rng.uniform(-0.5, 0.5, (len(test_data.time),))
    test_data = test_data + n

    ts = tst.TimeSeries(test_data)  # make Timeseries object
    return ts.choices  # return the underlying TimeSeriesParams object for testing


@pytest.mark.advanced
class TestTimeseriesHourly:

    def test_hourly_seasonal(self, test_TSP: tst.TimeSeriesParameters):
        """Testing a basic transformation with hourly data."""
        # Specify Params options
        test_TSP.remove_seasonal_cycle = True

        # Transform data and test
        result = test_TSP.transform_data()
        assert (result == test_TSP.data).sum().values.item() == 0
