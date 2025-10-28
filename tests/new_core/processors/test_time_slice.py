"""
Units tests for the TimeSlice processor.

This module contains tests to verify the functionality of the TimeSlice
processor in the climakitae.new_core.processors.time_slice module. The tests
cover various scenarios including slicing data arrays and datasets based on
specified time ranges.
"""

import pandas as pd
import pytest
import xarray as xr

from climakitae.new_core.processors.time_slice import TimeSlice


@pytest.fixture
def processor():
    """Fixture to create a TimeSlice processor instance."""
    yield TimeSlice(value=("2000-01-01", "2000-12-31"))


@pytest.fixture
def test_dataarray():
    """Fixture to create a sample xarray.DataArray for testing."""
    dataarray = xr.DataArray(
        data=[[1, 2], [3, 4], [5, 6]],
        dims=["time", "space"],
        coords={
            "time": pd.date_range("2000-01-01", periods=3, freq="2QS-JAN"),
            "space": ["A", "B"],
        },
    )
    yield dataarray


@pytest.fixture
def test_dataset():
    """Fixture to create a sample xarray.Dataset for testing."""
    dataset = xr.Dataset(
        {
            "var1": (("time", "space"), [[1, 2], [3, 4], [5, 6]]),
            "var2": (("time", "space"), [[10, 20], [30, 40], [50, 60]]),
        },
        coords={
            "time": pd.date_range("2000-01-01", periods=3, freq="2QS-JAN"),
            "space": ["A", "B"],
        },
    )
    yield dataset


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
        test_dataset: xr.Dataset,
    ) -> None:
        """Test slicing an xarray.Dataset."""
        result = processor.execute(test_dataset, context={})
        assert result.time.size == 2
        assert result.space.size == 2
        assert result["var1"].values.tolist() == [[1, 2], [3, 4]]
        assert result["var2"].values.tolist() == [[10, 20], [30, 40]]

    @pytest.mark.parametrize("container_type", [list, tuple, dict])
    def test_time_slice_data_types(
        self,
        container_type: type,
        processor: TimeSlice,
        test_dataarray: xr.DataArray,
        test_dataset: xr.Dataset,
    ) -> None:
        """Test slicing with different container types."""
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

        assert da_result.time.size == 2
        assert da_result.space.size == 2
        assert da_result.values.tolist() == [[1, 2], [3, 4]]

        # Check Dataset result
        assert ds_result.time.size == 2
        assert ds_result.space.size == 2
        assert ds_result["var1"].values.tolist() == [[1, 2], [3, 4]]
        assert ds_result["var2"].values.tolist() == [[10, 20], [30, 40]]

    def test_time_slice_invalid_type(
        self,
        processor: TimeSlice,
    ) -> None:
        """Test that an invalid type raises a TypeError."""
        with pytest.warns(UserWarning, match="Invalid data type for subsetting."):
            processor.execute(42, context={})
