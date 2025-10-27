"""
Units tests for the TimeSlice processor.

This module contains tests to verify the functionality of the TimeSlice
processor in the climakitae.new_core.processors.time_slice module. The tests
cover various scenarios including slicing data arrays and datasets based on
specified time ranges.
"""

from typing import Iterable, Union

import pandas as pd
import pytest
import xarray as xr

from climakitae.new_core.processors.time_slice import TimeSlice


@pytest.fixture
def processor() -> TimeSlice:
    """Fixture to create a TimeSlice processor instance."""
    yield TimeSlice(value=("2000-01-01", "2000-12-31"))


@pytest.fixture
def test_dataarray() -> xr.DataArray:
    """Fixture to create a sample xarray.DataArray for testing."""
    data = xr.DataArray(
        data=[[1, 2], [3, 4], [5, 6]],
        dims=["time", "space"],
        coords={
            "time": pd.date_range("2000-01-01", periods=3, freq="2QS-JAN"),
            "space": ["A", "B"],
        },
    )
    return data


class TestTimeSliceInit:
    """Tests for the initialization of TimeSlice processor."""

    def test_init(self):
        """Test initialization of TimeSlice processor."""
        processor = TimeSlice(value=("1990-01-01", "1990-12-31"))
        assert processor.value[0] == pd.Timestamp("1990-01-01")
        assert processor.value[1] == pd.Timestamp("1990-12-31")
        assert processor.name == "time_slice"


class TestTimeSliceExecute:
    """Tests for the execute method of TimeSlice processor."""

    def test_time_slice_dataarray_in_dict(
        self,
        processor: TimeSlice,
        test_dataarray: xr.DataArray,
    ) -> None:
        """Test slicing an xarray.DataArray within a dict."""
        data = {"example_key": test_dataarray}
        result = processor.execute(data, context={})
        sliced_data = result["example_key"]
        assert sliced_data.time.size == 2
        assert sliced_data.space.size == 2
        assert sliced_data.values.tolist() == [[1, 2], [3, 4]]

    def test_time_slice_dataarray(
        self,
        processor: TimeSlice,
        test_dataarray: xr.DataArray,
    ) -> None:
        """Test slicing an xarray.DataArray."""
        result = processor.execute(test_dataarray, context={})
        assert result.time.size == 2
        assert result.space.size == 2
        assert result.values.tolist() == [[1, 2], [3, 4]]

    def test_time_slice_dataset(
        self,
        processor: TimeSlice,
    ) -> None:
        """Test slicing an xarray.Dataset."""
        data = xr.Dataset(
            {
                "var1": (("time", "space"), [[1, 2], [3, 4], [5, 6]]),
                "var2": (("time", "space"), [[10, 20], [30, 40], [50, 60]]),
            },
            coords={
                "time": pd.date_range("2000-01-01", periods=3, freq="2QS-JAN"),
                "space": ["A", "B"],
            },
        )
        result = processor.execute(data, context={})
        assert result.time.size == 2
        assert result.space.size == 2
        assert result["var1"].values.tolist() == [[1, 2], [3, 4]]
        assert result["var2"].values.tolist() == [[10, 20], [30, 40]]

    def test_time_slice_list(
        self,
        processor: TimeSlice,
    ) -> None:
        """Test slicing a list of xarray.DataArray and xarray.Dataset."""
        data_array = xr.DataArray(
            data=[[1, 2], [3, 4], [5, 6]],
            dims=["time", "space"],
            coords={
                "time": pd.date_range("2000-01-01", periods=3, freq="6M"),
                "space": ["A", "B"],
            },
        )
        dataset = xr.Dataset(
            {
                "var1": (("time", "space"), [[1, 2], [3, 4], [5, 6]]),
                "var2": (("time", "space"), [[10, 20], [30, 40], [50, 60]]),
            },
            coords={
                "time": pd.date_range("2000-01-01", periods=3, freq="6M"),
                "space": ["A", "B"],
            },
        )
        data_list: Iterable[Union[xr.DataArray, xr.Dataset]] = [data_array, dataset]
        result = processor.execute(data_list, context={})

        # Check DataArray result
        da_result = result[0]
        assert da_result.time.size == 2
        assert da_result.space.size == 2
        assert da_result.values.tolist() == [[1, 2], [3, 4]]

        # Check Dataset result
        ds_result = result[1]
        assert ds_result.time.size == 2
        assert ds_result.space.size == 2
        assert ds_result["var1"].values.tolist() == [[1, 2], [3, 4]]
        assert ds_result["var2"].values.tolist() == [[10, 20], [30, 40]]

    def test_time_slice_invalid_type(
        self,
        processor: TimeSlice,
    ) -> None:
        """Test that an invalid type raises a TypeError."""
        with pytest.raises(TypeError):
            processor.execute(42, context={})
