"""
This script contains tests on various Timeseries Tools options using monthly
data. For now, the tests only test that the various parameter combinations can
run through `transform_data` without error, and that the data has been transformed
(not equal to original values), but does not test for exact expected values.
"""

import datetime as dt
import os

import numpy as np
import pytest
import xarray as xr

import climakitae.explore.timeseries as tst


@pytest.fixture
def test_TSP(rootdir) -> tst.TimeSeriesParameters:
    """Read in the test dataset and return a TimeSeriesParams object."""
    # This data is generated in "create_timeseries_test_data.py"
    test_filename = "test_data/timeseries_data_T2_2014_2016_monthly_45km.nc"
    test_filepath = os.path.join(rootdir, test_filename)
    test_data = xr.open_dataset(test_filepath).T2

    # Compute area average
    weights = np.cos(np.deg2rad(test_data.lat))
    test_data = test_data.weighted(weights).mean(["x", "y"])

    ts = tst.TimeSeries(test_data)  # make Timeseries object
    return ts.choices  # return the underlying TimeSeriesParams object for testing


@pytest.mark.advanced
class TestTimeseriesMonthlyTransform:

    def test_monthly_smoothing(self, test_TSP: tst.TimeSeriesParameters):
        """Test monthly running mean."""
        # Specify Params options
        test_TSP.smoothing = "Running Mean"
        test_TSP.num_timesteps = 3
        test_TSP.anomaly = False

        # Transform data and test
        result = test_TSP.transform_data()  # transform_data calls _running_mean()
        assert (result == test_TSP.data).sum().values.item() == 0

    def test_monthly_anomaly(self, test_TSP: tst.TimeSeriesParameters):
        """Test monthly weighted anomaly."""
        # Specify Params options
        test_TSP.anomaly = True
        test_TSP.reference_range = (dt.datetime(2014, 1, 1), dt.datetime(2014, 12, 31))

        # Transform data and test
        result = test_TSP.transform_data()
        assert (result == test_TSP.data).sum().values.item() == 0

    def test_monthly_anomaly_separate_seasons(self, test_TSP: tst.TimeSeriesParameters):
        """Test monthly weighted anomaly with separate seasons option."""
        # Specify Params options
        test_TSP.anomaly = True
        test_TSP.separate_seasons = True
        test_TSP.reference_range = (dt.datetime(2014, 1, 1), dt.datetime(2014, 12, 31))

        # Transform data and test
        result = test_TSP.transform_data()
        assert (result == test_TSP.data).sum().values.item() == 0

    def test_monthly_anomaly_and_smoothing(self, test_TSP: tst.TimeSeriesParameters):
        """Test anomaly and smoothing together."""
        # Specify Params options
        test_TSP.smoothing = "Running Mean"
        test_TSP.num_timesteps = 3
        test_TSP.anomaly = True
        test_TSP.reference_range = (dt.datetime(2014, 1, 1), dt.datetime(2014, 12, 31))

        # Transform data and test
        result = test_TSP.transform_data()
        assert (result == test_TSP.data).sum().values.item() == 0

    def test_monthly_anomaly_and_smoothing_separate_seasons(
        self,
        test_TSP: tst.TimeSeriesParameters,
    ):
        """Test anomaly, smoothing, and separate seasons options together."""
        # Specify Params options
        test_TSP.smoothing = "Running Mean"
        test_TSP.num_timesteps = 3
        test_TSP.anomaly = True
        test_TSP.separate_seasons = True
        test_TSP.reference_range = (dt.datetime(2014, 1, 1), dt.datetime(2014, 12, 31))

        # Transform data and test
        result = test_TSP.transform_data()
        assert (result == test_TSP.data).sum().values.item() == 0

    def test_seasonal(self, test_TSP: tst.TimeSeriesParameters):
        """Test seasonal cycle removal without smoothing."""
        # Specify Params options
        test_TSP.anomaly = False
        test_TSP.remove_seasonal_cycle = True

        # Transform data and test
        result = test_TSP.transform_data()
        assert (result == test_TSP.data).sum().values.item() == 0

    def test_seasonal_and_smoothing(self, test_TSP: tst.TimeSeriesParameters):
        """Test seasonal cycle removal with smoothing."""
        # Specify Params options
        test_TSP.smoothing = "Running Mean"
        test_TSP.num_timesteps = 3
        test_TSP.anomaly = False
        test_TSP.remove_seasonal_cycle = True

        # Transform data and test
        result = test_TSP.transform_data()
        assert (result == test_TSP.data).sum().values.item() == 0


@pytest.mark.advanced
class TestTimeseriesMonthlyExtremes:

    def test_extremes_smoothing(self, test_TSP: tst.TimeSeriesParameters):
        """Test extremes min with smoothing."""
        # Specify Params options
        test_TSP.anomaly = False
        test_TSP.smoothing = "Running Mean"
        test_TSP.num_timesteps = 3
        test_TSP.extremes = ["Min"]
        test_TSP.resample_window = 2

        # Transform data and test
        result = test_TSP.transform_data()
        assert (result == test_TSP.data).sum().values.item() == 0

    def test_extremes_min(self, test_TSP: tst.TimeSeriesParameters):
        """Test extremes min without smoothing."""
        # Specify Params options
        test_TSP.anomaly = False
        test_TSP.extremes = ["Min"]
        test_TSP.resample_window = 2

        # Transform data and test
        result = test_TSP.transform_data()
        assert (result == test_TSP.data).sum().values.item() == 0

    def test_extremes_max(self, test_TSP: tst.TimeSeriesParameters):
        """Test extremes max without smoothing."""
        # Specify Params options
        test_TSP.extremes = ["Max"]
        test_TSP.resample_window = 2

        # Transform data and test
        result = test_TSP.transform_data()
        assert (result == test_TSP.data).sum().values.item() == 0

    def test_extremes_percentile(self, test_TSP: tst.TimeSeriesParameters):
        """Test extremes percentile without smoothing."""
        # Specify Params options
        test_TSP.anomaly = False
        test_TSP.extremes = ["Percentile"]
        test_TSP.resample_window = 2
        test_TSP.percentile = 0.95

        # Transform data and test
        result = test_TSP.transform_data()
        assert (result == test_TSP.data).sum().values.item() == 0


class TestTimeseriesMonthlyErrors:

    def test_timeseries_no_data_array(
        self,
    ):
        """Provide empty dataset (not data array) to raise error."""
        with pytest.raises(ValueError):
            ts = tst.TimeSeries(xr.Dataset())

    def test_timeseries_lat_error(self, rootdir: str):
        """Provide lat/lon data to TimeSeries to raise error."""
        # Also changing the scenario to test multiple error case
        test_filename = "test_data/timeseries_data_T2_2014_2016_monthly_45km.nc"
        test_filepath = os.path.join(rootdir, test_filename)
        test_data = xr.open_dataset(test_filepath).T2
        test_data["scenario"] = np.array(["SSP 2-4.5"], dtype="<U44")

        with pytest.raises(ValueError):
            ts = tst.TimeSeries(test_data)

    def test_timeseries_scenario_error(self, test_TSP: tst.TimeSeriesParameters):
        """Remove 'Historical' text from scenario list in data to raise error."""
        test_data = test_TSP.data
        test_data["scenario"] = np.array(["SSP 2-4.5"], dtype="<U44")

        with pytest.raises(ValueError):
            ts = tst.TimeSeries(test_data)


class TestTimeseriesObject:

    def test_output_current(self, rootdir: str):
        test_filename = "test_data/timeseries_data_T2_2014_2016_monthly_45km.nc"
        test_filepath = os.path.join(rootdir, test_filename)
        test_data = xr.open_dataset(test_filepath).T2

        # Compute area average
        weights = np.cos(np.deg2rad(test_data.lat))
        test_data = test_data.weighted(weights).mean("x").mean("y")

        ts = tst.TimeSeries(test_data)  # make Timeseries object
        current = ts.output_current()

        assert isinstance(current, xr.core.dataarray.DataArray)
        for item in [
            "anomaly",
            "extremes",
            "reference_range",
            "remove_seasonal_cycle",
            "resample_period",
            "resample_window",
            "smoothing",
        ]:
            assert "timeseries: " + item in current.attrs
