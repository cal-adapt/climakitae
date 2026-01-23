"""
Unit tests for the DropLeapDays processor.

This module contains tests to verify the functionality of the DropLeapDays
processor in the climakitae.new_core.processors.drop_leap_days module. The tests
cover various scenarios including dropping leap days from data arrays and datasets.
"""

import pandas as pd
import pytest
import xarray as xr

from climakitae.new_core.processors.drop_leap_days import DropLeapDays


@pytest.fixture
def processor():
    """Fixture to create a DropLeapDays processor instance."""
    yield DropLeapDays(value=True)


@pytest.fixture
def processor_disabled():
    """Fixture to create a disabled DropLeapDays processor instance."""
    yield DropLeapDays(value=False)


@pytest.fixture
def test_dataarray_with_leap_day():
    """Fixture to create a sample xarray.DataArray with a leap day."""
    # Create dates including Feb 29, 2000 (leap year)
    dates = pd.to_datetime(["2000-02-28", "2000-02-29", "2000-03-01", "2000-03-02"])
    dataarray = xr.DataArray(
        data=[[1, 2], [3, 4], [5, 6], [7, 8]],
        dims=["time", "space"],
        coords={
            "time": dates,
            "space": ["A", "B"],
        },
    )
    yield dataarray


@pytest.fixture
def test_dataset_with_leap_day():
    """Fixture to create a sample xarray.Dataset with a leap day."""
    dates = pd.to_datetime(["2000-02-28", "2000-02-29", "2000-03-01", "2000-03-02"])
    dataset = xr.Dataset(
        {
            "var1": (("time", "space"), [[1, 2], [3, 4], [5, 6], [7, 8]]),
            "var2": (("time", "space"), [[10, 20], [30, 40], [50, 60], [70, 80]]),
        },
        coords={
            "time": dates,
            "space": ["A", "B"],
        },
    )
    yield dataset


@pytest.fixture
def test_dataarray_no_leap_day():
    """Fixture to create a sample xarray.DataArray without leap days."""
    dates = pd.to_datetime(["2001-02-27", "2001-02-28", "2001-03-01", "2001-03-02"])
    dataarray = xr.DataArray(
        data=[[1, 2], [3, 4], [5, 6], [7, 8]],
        dims=["time", "space"],
        coords={
            "time": dates,
            "space": ["A", "B"],
        },
    )
    yield dataarray


@pytest.fixture
def test_dataarray_no_time_dim():
    """Fixture to create a sample xarray.DataArray without time dimension."""
    dataarray = xr.DataArray(
        data=[1, 2, 3, 4],
        dims=["space"],
        coords={
            "space": ["A", "B", "C", "D"],
        },
    )
    yield dataarray


class TestDropLeapDaysInit:
    """Tests for the initialization of DropLeapDays processor."""

    def test_init_default(self):
        """Test default initialization of DropLeapDays processor."""
        processor = DropLeapDays()
        assert processor.value is True
        assert processor.name == "drop_leap_days"

    def test_init_enabled(self):
        """Test initialization with value=True."""
        processor = DropLeapDays(value=True)
        assert processor.value is True
        assert processor.name == "drop_leap_days"

    def test_init_disabled(self):
        """Test initialization with value=False."""
        processor = DropLeapDays(value=False)
        assert processor.value is False
        assert processor.name == "drop_leap_days"


class TestDropLeapDaysExecute:
    """Tests for the execute method of DropLeapDays processor."""

    def test_drop_leap_days_dataarray(
        self,
        processor: DropLeapDays,
        test_dataarray_with_leap_day: xr.DataArray,
    ) -> None:
        """Test dropping leap days from an xarray.DataArray."""
        result = processor.execute(test_dataarray_with_leap_day, context={})
        assert result.time.size == 3
        assert result.space.size == 2
        # Feb 29 should be removed
        assert result.values.tolist() == [[1, 2], [5, 6], [7, 8]]
        # Verify Feb 29 is not in the result
        assert not any((result.time.dt.month == 2) & (result.time.dt.day == 29))

    def test_drop_leap_days_dataset(
        self,
        processor: DropLeapDays,
        test_dataset_with_leap_day: xr.Dataset,
    ) -> None:
        """Test dropping leap days from an xarray.Dataset."""
        result = processor.execute(test_dataset_with_leap_day, context={})
        assert result.time.size == 3
        assert result.space.size == 2
        assert result["var1"].values.tolist() == [[1, 2], [5, 6], [7, 8]]
        assert result["var2"].values.tolist() == [[10, 20], [50, 60], [70, 80]]

    def test_disabled_processor_passes_through(
        self,
        processor_disabled: DropLeapDays,
        test_dataarray_with_leap_day: xr.DataArray,
    ) -> None:
        """Test that disabled processor passes data through unchanged."""
        result = processor_disabled.execute(test_dataarray_with_leap_day, context={})
        assert result.time.size == 4
        xr.testing.assert_equal(result, test_dataarray_with_leap_day)

    def test_no_leap_day_unchanged(
        self,
        processor: DropLeapDays,
        test_dataarray_no_leap_day: xr.DataArray,
    ) -> None:
        """Test that data without leap days remains unchanged."""
        result = processor.execute(test_dataarray_no_leap_day, context={})
        assert result.time.size == 4
        xr.testing.assert_equal(result, test_dataarray_no_leap_day)

    def test_no_time_dimension(
        self,
        processor: DropLeapDays,
        test_dataarray_no_time_dim: xr.DataArray,
    ) -> None:
        """Test that data without time dimension is returned unchanged."""
        result = processor.execute(test_dataarray_no_time_dim, context={})
        xr.testing.assert_equal(result, test_dataarray_no_time_dim)

    @pytest.mark.parametrize("container_type", [list, tuple, dict])
    def test_drop_leap_days_data_types(
        self,
        container_type: type,
        processor: DropLeapDays,
        test_dataarray_with_leap_day: xr.DataArray,
        test_dataset_with_leap_day: xr.Dataset,
    ) -> None:
        """Test dropping leap days with different container types."""
        if container_type is dict:
            data = {
                "dataarray": test_dataarray_with_leap_day,
                "dataset": test_dataset_with_leap_day,
            }
        elif container_type is list:
            data = [test_dataarray_with_leap_day, test_dataset_with_leap_day]
        elif container_type is tuple:
            data = (test_dataarray_with_leap_day, test_dataset_with_leap_day)

        result = processor.execute(data, context={})

        # Check DataArray result
        if container_type is dict:
            da_result = result["dataarray"]
            ds_result = result["dataset"]
        else:
            da_result = result[0]
            ds_result = result[1]

        assert da_result.time.size == 3
        assert da_result.values.tolist() == [[1, 2], [5, 6], [7, 8]]

        # Check Dataset result
        assert ds_result.time.size == 3
        assert ds_result["var1"].values.tolist() == [[1, 2], [5, 6], [7, 8]]

    def test_invalid_type_warning(
        self,
        processor: DropLeapDays,
    ) -> None:
        """Test that an invalid type logs a warning and returns the input."""
        with pytest.warns(UserWarning, match="Invalid data type for DropLeapDays"):
            result = processor.execute(42, context={})
        assert result == 42


class TestDropLeapDaysUpdateContext:
    """Tests for the update_context method of DropLeapDays processor."""

    def test_update_context(
        self,
        processor: DropLeapDays,
        test_dataarray_with_leap_day: xr.DataArray,
    ) -> None:
        """Test that context is updated after processing."""
        context = {}
        processor.execute(test_dataarray_with_leap_day, context=context)

        from climakitae.core.constants import _NEW_ATTRS_KEY

        assert _NEW_ATTRS_KEY in context
        assert "drop_leap_days" in context[_NEW_ATTRS_KEY]
        assert "February 29" in context[_NEW_ATTRS_KEY]["drop_leap_days"]


class TestDropLeapDaysMultipleLeapYears:
    """Tests for handling multiple leap years."""

    def test_multiple_leap_years(
        self,
        processor: DropLeapDays,
    ) -> None:
        """Test dropping leap days across multiple leap years."""
        # Create dates spanning multiple leap years
        dates = pd.to_datetime([
            "2000-02-28", "2000-02-29", "2000-03-01",  # 2000 is a leap year
            "2004-02-28", "2004-02-29", "2004-03-01",  # 2004 is a leap year
            "2001-02-28", "2001-03-01",  # 2001 is not a leap year
        ])
        dataarray = xr.DataArray(
            data=list(range(1, 9)),  # 8 values to match 8 dates
            dims=["time"],
            coords={"time": dates},
        )

        result = processor.execute(dataarray, context={})

        # Should have 6 timestamps (8 original - 2 leap days)
        assert result.time.size == 6
        # Verify no Feb 29 dates remain
        assert not any((result.time.dt.month == 2) & (result.time.dt.day == 29))
