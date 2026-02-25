"""
Units tests for the ConvertToLocalTime processor.

This module contains tests to verify the functionality of the ConvertToLocalTime
processor in the climakitae.new_core.processors.convert_to_local_time module. The tests
cover various scenarios including using different data types and gridded versus
station data.
"""

import pandas as pd
import pytest
import xarray as xr
import numpy as np

from climakitae.new_core.processors.convert_to_local_time import ConvertToLocalTime


@pytest.fixture
def processor():
    """Fixture to create a ConvertToLocalTime processor instance."""
    yield ConvertToLocalTime(value={"convert": "yes", "repair_time_axis": "no"})


@pytest.fixture
def processor_no_convert():
    """Fixture to create a ConvertToLocalTime processor instance."""
    yield ConvertToLocalTime(value={"convert": "no"})


@pytest.fixture
def processor_repair_time_axis():
    """Fixture to create a ConvertToLocalTime processor instance."""
    yield ConvertToLocalTime(value={"convert": "yes", "repair_time_axis": "yes"})


@pytest.fixture
def test_dataarray_daylight_savings():
    """Fixture to create a sample xarray.DataArray for testing with a full year."""
    dataarray = xr.DataArray(
        data=np.ones((8784, 1, 1)),
        dims=["time", "lat", "lon"],
        coords={
            "time": pd.date_range("2016-01-01 00", "2016-12-31 23", freq="1h"),
            "lat": [35],
            "lon": [-119],
        },
    )
    yield dataarray


@pytest.fixture
def test_dataset_daylight_savings():
    """Fixture to create a sample xarray.Dataset for testing with a full year."""
    dataset = xr.Dataset(
        {
            "var1": (("time", "lat", "lon"), np.ones((8784, 1, 1))),
            "var2": (("time", "lat", "lon"), np.ones((8784, 1, 1))),
        },
        coords={
            "time": pd.date_range("2016-01-01 00", "2016-12-31 23", freq="1h"),
            "lat": [35],
            "lon": [-119],
        },
    )
    yield dataset


@pytest.fixture
def test_dataarray():
    """Fixture to create a sample xarray.DataArray for testing."""
    dataarray = xr.DataArray(
        data=np.ones((24, 1, 1)),
        dims=["time", "lat", "lon"],
        coords={
            "time": pd.date_range("2000-01-01 00", "2000-01-01 23", freq="1h"),
            "lat": [35],
            "lon": [-119],
        },
    )
    yield dataarray


@pytest.fixture
def test_dataset():
    """Fixture to create a sample xarray.Dataset for testing."""
    dataset = xr.Dataset(
        {
            "var1": (("time", "lat", "lon"), np.ones((24, 1, 1))),
            "var2": (("time", "lat", "lon"), np.ones((24, 1, 1))),
        },
        coords={
            "time": pd.date_range("2000-01-01 00", "2000-01-01 23", freq="1h"),
            "lat": [35],
            "lon": [-119],
        },
    )
    yield dataset


@pytest.fixture
def test_daily():
    """Fixture to create a sample xarray.DataArray for testing."""
    dataarray = xr.DataArray(
        data=np.ones((10, 1, 1)),
        dims=["time", "lat", "lon"],
        coords={
            "time": pd.date_range("2000-01-01", "2000-01-10", freq="1d"),
            "lat": [35],
            "lon": [-119],
        },
    )
    yield dataarray


@pytest.fixture
def test_hdp_station():
    """Fixture to create a sample xarray.DataArray for testing."""
    dataset = xr.Dataset(
        {
            "var1": (("time", "lat", "lon"), np.ones((24, 1, 1))),
            "var2": (("time", "lat", "lon"), np.ones((24, 1, 1))),
        },
        coords={
            "time": pd.date_range("2000-01-01 00", "2000-01-01 23", freq="1h"),
            "lat": (("time", "lat"), np.ones((24, 1)) * 35),
            "lon": (("time", "lon"), np.ones((24, 1)) * -119),
        },
    )
    yield dataset


class TestConvertToLocalTimeInit:
    """Tests for the initialization of ConvertToLocalTime processor."""

    def test_init(self):
        """Test initialization of ConvertToLocalTime processor."""
        processor = ConvertToLocalTime(value={"convert": "yes"})
        assert processor.value[0] == "y"
        assert processor.name == "convert_to_local_time"


class TestConvertToLocalTimeExecute:
    """Tests for the execute method of ConvertToLocalTime processor."""

    def test_convert_to_local_time_station(
        self,
        processor: ConvertToLocalTime,
        test_hdp_station: xr.DataArray,
    ) -> None:
        """Test converting an xarray.DataArray."""
        result = processor.execute(test_hdp_station, context={"_catalog_key": "hdp"})
        assert result["var1"].attrs["timezone"] == "America/Los_Angeles"
        assert result["var2"].attrs["timezone"] == "America/Los_Angeles"
        assert result.time[0] == pd.Timestamp("1999-12-31 16:00:00")

    def test_convert_to_local_time_dataarray(
        self,
        processor: ConvertToLocalTime,
        test_dataarray: xr.DataArray,
    ) -> None:
        """Test converting an xarray.DataArray."""
        result = processor.execute(test_dataarray, context={})
        assert result.attrs["timezone"] == "America/Los_Angeles"
        assert result.time[0] == pd.Timestamp("1999-12-31 16:00:00")

    def test_convert_to_local_time_dataset(
        self,
        processor: ConvertToLocalTime,
        test_dataset: xr.Dataset,
    ) -> None:
        """Test converting an xarray.Dataset."""
        result = processor.execute(test_dataset, context={})
        assert result.time.size == 24
        assert result["var1"].attrs["timezone"] == "America/Los_Angeles"
        assert result["var2"].attrs["timezone"] == "America/Los_Angeles"
        assert result.time[0] == pd.Timestamp("1999-12-31 16:00:00")

    @pytest.mark.parametrize("container_type", [list, tuple, dict])
    def test_convert_to_local_time_data_types(
        self,
        container_type: type,
        processor: ConvertToLocalTime,
        test_dataarray: xr.DataArray,
        test_dataset: xr.Dataset,
    ) -> None:
        """Test conversion with different container types."""
        if container_type is dict:
            data = {
                "dataarray": test_dataarray,
                "dataset": test_dataset,
            }
        elif container_type is list:
            data = [test_dataarray, test_dataset]
        elif container_type is tuple:
            data = (test_dataarray, test_dataset)

        result = processor.execute(data, context={})

        # Check DataArray result
        if container_type is dict:
            da_result = result["dataarray"]
            ds_result = result["dataset"]
        else:
            da_result = result[0]
            ds_result = result[1]

        assert da_result.attrs["timezone"] == "America/Los_Angeles"

        # Check Dataset result
        assert ds_result["var1"].attrs["timezone"] == "America/Los_Angeles"
        assert ds_result["var2"].attrs["timezone"] == "America/Los_Angeles"
        assert ds_result.time[0] == pd.Timestamp("1999-12-31 16:00:00")

    def test_convert_to_local_time_invalid_type(
        self,
        processor: ConvertToLocalTime,
    ) -> None:
        """Test that an invalid type raises a TypeError."""
        with pytest.warns(UserWarning, match="Invalid data type for subsetting."):
            processor.execute(42, context={})

    def test_convert_to_local_time_invalid_daily_freq(
        self,
        processor: ConvertToLocalTime,
        test_daily: xr.DataArray,
    ) -> None:
        """Test that no conversion happens for daily data."""
        result = processor.execute(test_daily, context={})
        assert "timezone" not in result.attrs
        assert (result.time == test_daily.time).all()

    def test_convert_to_local_time_repair_time_axis_dataarray(
        self,
        processor_repair_time_axis: ConvertToLocalTime,
        test_dataarray_daylight_savings: xr.DataArray,
    ) -> None:
        """Test repair_time_axis option with a data array."""
        result = processor_repair_time_axis.execute(
            test_dataarray_daylight_savings, context={}
        )
        assert result.attrs["timezone"] == "America/Los_Angeles"

        # No leap day timestamps
        assert pd.Timestamp("2016-02-29 23") not in result.time

        # Missing hour at end of daylight savings copied
        assert pd.Timestamp("2016-03-13 02") in result.time
        assert result.sel(time=pd.Timestamp("2016-03-13 02")) == result.sel(
            time=pd.Timestamp("2016-03-13 01")
        )

        # extra daylight savings timestamp should be dropped by this processor
        # so only 1 value present
        assert result.time.sel(time=pd.Timestamp("2016-11-06 01")).shape == ()

    def test_convert_to_local_time_repair_time_axis_dataset(
        self,
        processor_repair_time_axis: ConvertToLocalTime,
        test_dataset_daylight_savings: xr.Dataset,
    ) -> None:
        """Test repair_time_axis option with a dataset."""
        result = processor_repair_time_axis.execute(
            test_dataset_daylight_savings, context={}
        )
        assert result["var1"].attrs["timezone"] == "America/Los_Angeles"

        # No leap day timestamps
        assert pd.Timestamp("2016-02-29 23") not in result.time

        # Missing hour at end of daylight savings copied
        assert pd.Timestamp("2016-03-13 02") in result.time
        assert result.sel(time=pd.Timestamp("2016-03-13 02")) == result.sel(
            time=pd.Timestamp("2016-03-13 01")
        )

        # extra daylight savings timestamp should be dropped by this processor
        # so only 1 value present
        assert result.time.sel(time=pd.Timestamp("2016-11-06 01")).shape == ()

    def test_convert_to_local_time_no_conversion(
        self,
        processor_no_convert: ConvertToLocalTime,
        test_dataarray: xr.DataArray,
    ) -> None:
        """Test that no time conversion happens when convert='no'."""
        result = processor_no_convert.execute(test_dataarray, context={})
        assert "timezone" not in result.attrs
        assert result.time[0] == test_dataarray.time[0]
