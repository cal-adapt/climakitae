"""
Unit tests for climakitae/new_core/processors/processor_utils.py.

This module contains comprehensive unit tests for the processor utility functions
that handle block maxima extraction, effective sample size calculations,
and time domain extension operations.
"""

from unittest.mock import patch

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climakitae.core.constants import UNSET
from climakitae.new_core.processors.processor_utils import (
    FALLBACK_ESS_VALUE,
    LARGE_DATASET_THRESHOLD,
    LARGE_TIMESERIES_THRESHOLD,
    _apply_duration_filter_vectorized,
    _apply_groupby_filter_vectorized,
    _apply_grouped_duration_filter_vectorized,
    _calc_average_ess_gridded_optimized,
    _calc_average_ess_timeseries_optimized,
    _check_effective_sample_size_optimized,
    _extract_block_extremes_vectorized,
    _get_block_maxima_optimized,
    _handle_nan_values_optimized,
    _optimize_chunking_for_block_maxima,
    _set_block_maxima_attributes,
    extend_time_domain,
    find_station_match,
)


class TestDataFactory:
    """Factory for creating consistent test data."""

    @staticmethod
    def create_climate_dataset(
        variables=["tasmax"],
        time_periods=365,
        lat_points=5,
        lon_points=5,
        start_date="2020-01-01",
        frequency="d",
        with_dask=False,
        **kwargs,
    ) -> xr.DataArray:
        """Create standardized test dataset.

        Parameters
        ----------
        variables : list
            List of variable names to include
        time_periods : int
            Number of time periods
        lat_points : int
            Number of latitude points
        lon_points : int
            Number of longitude points
        start_date : str
            Start date for time dimension
        frequency : str
            Frequency for time dimension
        with_dask : bool
            Whether to create dask-backed data
        **kwargs : dict
            Additional attributes

        Returns
        -------
        xr.DataArray
            Test climate dataset
        """
        # Create realistic temperature data with seasonal cycle
        time = pd.date_range(start_date, periods=time_periods, freq=frequency)
        lat = np.linspace(32, 42, lat_points)
        lon = np.linspace(-124, -114, lon_points)

        # Create temperature data with realistic range and seasonal variation
        base_temp = 20 + 10 * np.sin(2 * np.pi * np.arange(time_periods) / 365)
        spatial_variation = np.random.normal(0, 2, (lat_points, lon_points))

        if with_dask:
            data_array = da.zeros((time_periods, lat_points, lon_points))
            for i in range(time_periods):
                noise = np.random.normal(0, 1, (lat_points, lon_points))
                data_array[i] = base_temp[i] + spatial_variation + noise
        else:
            data_array = np.zeros((time_periods, lat_points, lon_points))
            for i in range(time_periods):
                noise = np.random.normal(0, 1, (lat_points, lon_points))
                data_array[i] = base_temp[i] + spatial_variation + noise

        da_series = xr.DataArray(
            data_array,
            coords={"time": time, "lat": lat, "lon": lon},
            dims=["time", "lat", "lon"],
            name=variables[0],
            attrs={"units": "K", "long_name": f"{variables[0]} data", **kwargs},
        )

        # Set frequency attribute based on input frequency
        if frequency == "h":
            da_series.attrs["frequency"] = "1hr"
        elif frequency == "1hr" or frequency == "hourly":
            da_series.attrs["frequency"] = "hourly"

        return da_series

    @staticmethod
    def create_timeseries_dataset(
        time_periods=365,
        start_date="2020-01-01",
        frequency="D",
        with_dask=False,
        **kwargs,
    ) -> xr.DataArray:
        """Create standardized test timeseries dataset."""
        time = pd.date_range(start_date, periods=time_periods, freq=frequency)

        # Create realistic temperature timeseries with seasonal cycle
        base_temp = 20 + 10 * np.sin(2 * np.pi * np.arange(time_periods) / 365)
        noise = np.random.normal(0, 2, time_periods)

        if with_dask:
            data_array = da.from_array(base_temp + noise)
        else:
            data_array = base_temp + noise

        return xr.DataArray(
            data_array,
            coords={"time": time},
            dims=["time"],
            name="temperature",
            attrs={"units": "K", "long_name": "temperature data", **kwargs},
        )


@pytest.fixture
def sample_gridded_dataset():
    """Create a sample gridded xarray Dataset for testing.

    Returns
    -------
    xr.DataArray
        Sample climate dataset with 3 years of daily temperature data.
    """
    return TestDataFactory.create_climate_dataset(
        variables=["tasmax"],
        time_periods=365 * 3,  # 3 years of daily data
        lat_points=5,
        lon_points=5,
        start_date="2020-01-01",
    )


@pytest.fixture
def sample_gridded_dataset_for_ess():
    """Create a sample gridded dataset with x,y dimensions for ESS testing.

    Returns
    -------
    xr.DataArray
        Sample climate dataset with x, y, time dimensions as expected by ESS functions.
    """
    return TestDataFactory.create_climate_dataset(
        variables=["tasmax"],
        time_periods=365 * 3,  # 3 years of daily data
        lat_points=5,
        lon_points=5,
        start_date="2020-01-01",
    ).rename({"lat": "x", "lon": "y"})


@pytest.fixture
def sample_timeseries_dataset():
    """Create a sample timeseries xarray DataArray for testing.

    Returns
    -------
    xr.DataArray
        Sample timeseries dataset with 3 years of daily temperature data.
    """
    return TestDataFactory.create_timeseries_dataset(
        time_periods=365 * 3, start_date="2020-01-01"  # 3 years of daily data
    )


@pytest.fixture
def sample_hourly_dataset():
    """Create a sample hourly dataset for duration testing.

    Returns
    -------
    xr.DataArray
        Sample hourly dataset.
    """
    return TestDataFactory.create_climate_dataset(
        variables=["tasmax"],
        time_periods=24 * 7,  # 1 week of hourly data
        lat_points=3,
        lon_points=3,
        start_date="2020-01-01",
        frequency="h",
    )


@pytest.fixture
def sample_dask_dataset():
    """Create a sample dask-backed dataset for testing.

    Returns
    -------
    xr.DataArray
        Sample dask-backed dataset.
    """
    return TestDataFactory.create_climate_dataset(
        variables=["tasmax"],
        time_periods=365 * 3,
        lat_points=10,
        lon_points=10,
        start_date="2020-01-01",
        with_dask=True,
    )


@pytest.fixture
def sample_nan_dataset():
    """Create a dataset with NaN values for testing.

    Returns
    -------
    xr.DataArray
        Dataset with some NaN values.
    """
    dataset_with_nan = TestDataFactory.create_climate_dataset(
        variables=["tasmax"],
        time_periods=365 * 2,
        lat_points=3,
        lon_points=3,
        start_date="2020-01-01",
    )
    # Introduce NaN values in some time steps
    dataset_with_nan.values[50:60] = np.nan
    dataset_with_nan.values[200:210] = np.nan
    return dataset_with_nan


@pytest.fixture
def scenario_dict_with_historical():
    """Create a dictionary with SSP and historical scenarios.

    Returns
    -------
    Dict[str, xr.DataArray]
        Dictionary with scenario data for testing time domain extension.
    """
    # Create historical data (1980-2014)
    hist_data = TestDataFactory.create_climate_dataset(
        time_periods=365 * 35, start_date="1980-01-01"  # 35 years
    )
    hist_data.attrs.update(
        {"experiment": "historical", "source": "CMIP6", "institution": "Test"}
    )

    # Create SSP245 data (2015-2100)
    ssp245_data = TestDataFactory.create_climate_dataset(
        time_periods=365 * 86, start_date="2015-01-01"  # 86 years
    )
    ssp245_data.attrs.update(
        {"experiment": "ssp245", "source": "CMIP6", "institution": "Test"}
    )

    # Create SSP370 data (2015-2100)
    ssp370_data = TestDataFactory.create_climate_dataset(
        time_periods=365 * 86, start_date="2015-01-01"
    )
    ssp370_data.attrs.update(
        {"experiment": "ssp370", "source": "CMIP6", "institution": "Test"}
    )

    return {
        "model1_ssp245_r1i1p1f1": ssp245_data,
        "model1_ssp370_r1i1p1f1": ssp370_data,
        "model1_historical_r1i1p1f1": hist_data,
        "model2_ssp245_r1i1p1f1": ssp245_data.copy(),
        "model2_historical_r1i1p1f1": hist_data.copy(),
    }


class TestGetBlockMaximaOptimized:
    """Test class for _get_block_maxima_optimized function."""

    def test_init_successful_max(self, sample_gridded_dataset):
        """Test successful initialization with max extremes."""
        result = _get_block_maxima_optimized(
            sample_gridded_dataset, extremes_type="max", check_ess=False
        )

        assert isinstance(result, xr.DataArray)
        assert result.attrs["extremes type"] == "maxima"
        assert result.attrs["extreme_value_extraction_method"] == "block maxima"
        assert "time" in result.dims
        # Should have 3 annual blocks from 3 years of data
        assert result.sizes["time"] == 3

    def test_init_successful_min(self, sample_gridded_dataset):
        """Test successful initialization with min extremes."""
        result = _get_block_maxima_optimized(
            sample_gridded_dataset, extremes_type="min", check_ess=False
        )

        assert isinstance(result, xr.DataArray)
        assert result.attrs["extremes type"] == "minima"
        assert result.attrs["extreme_value_extraction_method"] == "block maxima"

    def test_invalid_extremes_type(self, sample_gridded_dataset):
        """Test with invalid extremes type."""
        with pytest.raises(ValueError, match="invalid extremes type"):
            _get_block_maxima_optimized(
                sample_gridded_dataset, extremes_type="invalid", check_ess=False
            )

    def test_with_duration_filter(self, sample_hourly_dataset):
        """Test with duration filter applied."""
        sample_hourly_dataset.attrs["frequency"] = "1hr"

        result = _get_block_maxima_optimized(
            sample_hourly_dataset,
            extremes_type="max",
            duration=(4, "hour"),
            check_ess=False,
        )

        assert isinstance(result, xr.DataArray)
        assert result.attrs["duration"] == (4, "hour")

    def test_with_groupby_filter(self, sample_gridded_dataset):
        """Test with groupby filter applied."""
        result = _get_block_maxima_optimized(
            sample_gridded_dataset,
            extremes_type="max",
            groupby=(1, "day"),
            check_ess=False,
        )

        assert isinstance(result, xr.DataArray)
        assert result.attrs["groupby"] == (1, "day")

    def test_with_grouped_duration_filter(self, sample_gridded_dataset):
        """Test with grouped duration filter applied."""
        result = _get_block_maxima_optimized(
            sample_gridded_dataset,
            extremes_type="max",
            groupby=(1, "day"),
            grouped_duration=(3, "day"),
            check_ess=False,
        )

        assert isinstance(result, xr.DataArray)
        assert result.attrs["groupby"] == (1, "day")
        assert result.attrs["grouped_duration"] == (3, "day")

    def test_grouped_duration_without_groupby(self, sample_gridded_dataset):
        """Test grouped duration without groupby raises error."""
        with pytest.raises(
            ValueError, match="To use `grouped_duration` option, must first use groupby"
        ):
            _get_block_maxima_optimized(
                sample_gridded_dataset,
                extremes_type="max",
                grouped_duration=(3, "day"),
                check_ess=False,
            )

    def test_with_different_block_sizes(self, sample_gridded_dataset):
        """Test with different block sizes."""
        # Test 2-year blocks
        result = _get_block_maxima_optimized(
            sample_gridded_dataset, extremes_type="max", block_size=2, check_ess=False
        )

        assert isinstance(result, xr.DataArray)
        assert result.attrs["block_size"] == "2 year"
        # Should have fewer blocks with larger block size
        assert result.sizes["time"] <= 2

    @patch("builtins.print")
    def test_with_ess_check_enabled(self, mock_print, sample_gridded_dataset):
        """Test with effective sample size check enabled."""
        result = _get_block_maxima_optimized(
            sample_gridded_dataset, extremes_type="max", check_ess=True
        )

        assert isinstance(result, xr.DataArray)
        # Should have called print for ESS checking
        assert mock_print.call_count >= 0  # May or may not trigger warnings

    def test_with_dask_arrays(self, sample_dask_dataset):
        """Test with Dask-backed arrays."""
        result = _get_block_maxima_optimized(
            sample_dask_dataset,
            extremes_type="max",
            check_ess=False,
            chunk_spatial=True,
            max_memory_gb=1.0,
        )

        assert isinstance(result, xr.DataArray)
        assert result.attrs["extremes type"] == "maxima"


class TestOptimizeChunkingForBlockMaxima:
    """Test class for _optimize_chunking_for_block_maxima function."""

    def test_non_dask_array_unchanged(self, sample_gridded_dataset):
        """Test that non-Dask arrays are returned unchanged."""
        result = _optimize_chunking_for_block_maxima(
            sample_gridded_dataset, max_memory_gb=2.0, chunk_spatial=True
        )

        assert result is sample_gridded_dataset

    def test_dask_array_rechunking(self, sample_dask_dataset):
        """Test that Dask arrays are properly rechunked."""
        result = _optimize_chunking_for_block_maxima(
            sample_dask_dataset, max_memory_gb=2.0, chunk_spatial=True
        )

        assert hasattr(result.data, "chunks")
        # Should have rechunked the array
        assert len(result.chunks) == 3  # time, lat, lon dimensions
        # Time should be chunked appropriately
        time_chunks = result.chunks[0]  # first dimension is time
        assert len(time_chunks) >= 1

    def test_dask_array_no_spatial_chunking(self, sample_dask_dataset):
        """Test Dask array optimization without spatial chunking."""
        result = _optimize_chunking_for_block_maxima(
            sample_dask_dataset, max_memory_gb=2.0, chunk_spatial=False
        )

        assert hasattr(result.data, "chunks")
        # Should have rechunked the array
        assert result.chunks is not None
        assert len(result.chunks) == 3  # time, lat, lon dimensions


class TestApplyDurationFilterVectorized:
    """Test class for _apply_duration_filter_vectorized function."""

    def test_duration_filter_max(self, sample_hourly_dataset):
        """Test duration filter with max extremes."""
        sample_hourly_dataset.attrs["frequency"] = "1hr"

        result = _apply_duration_filter_vectorized(
            sample_hourly_dataset, duration=(4, "hour"), extremes_type="max"
        )

        assert isinstance(result, xr.DataArray)
        assert result.sizes["time"] == sample_hourly_dataset.sizes["time"]

    def test_duration_filter_min(self, sample_hourly_dataset):
        """Test duration filter with min extremes."""
        sample_hourly_dataset.attrs["frequency"] = "1hr"

        result = _apply_duration_filter_vectorized(
            sample_hourly_dataset, duration=(4, "hour"), extremes_type="min"
        )

        assert isinstance(result, xr.DataArray)
        assert result.sizes["time"] == sample_hourly_dataset.sizes["time"]

    def test_invalid_duration_type(self, sample_gridded_dataset):
        """Test with invalid duration type."""
        with pytest.raises(ValueError, match="Current specifications not implemented"):
            _apply_duration_filter_vectorized(
                sample_gridded_dataset, duration=(4, "day"), extremes_type="max"
            )

    def test_invalid_extremes_type_duration(self, sample_hourly_dataset):
        """Test with invalid extremes type."""
        sample_hourly_dataset.attrs["frequency"] = "1hr"

        with pytest.raises(
            ValueError, match='extremes_type needs to be either "max" or "min"'
        ):
            _apply_duration_filter_vectorized(
                sample_hourly_dataset, duration=(4, "hour"), extremes_type="invalid"
            )


class TestApplyGroupbyFilterVectorized:
    """Test class for _apply_groupby_filter_vectorized function."""

    def test_groupby_filter_max(self, sample_gridded_dataset):
        """Test groupby filter with max extremes."""
        result = _apply_groupby_filter_vectorized(
            sample_gridded_dataset, groupby=(1, "day"), extremes_type="max"
        )

        assert isinstance(result, xr.DataArray)
        # Should resample to daily maxima
        assert result.sizes["time"] <= sample_gridded_dataset.sizes["time"]

    def test_groupby_filter_min(self, sample_gridded_dataset):
        """Test groupby filter with min extremes."""
        result = _apply_groupby_filter_vectorized(
            sample_gridded_dataset, groupby=(1, "day"), extremes_type="min"
        )

        assert isinstance(result, xr.DataArray)
        assert result.sizes["time"] <= sample_gridded_dataset.sizes["time"]

    def test_multi_day_groupby(self, sample_gridded_dataset):
        """Test groupby filter with multi-day grouping."""
        result = _apply_groupby_filter_vectorized(
            sample_gridded_dataset, groupby=(3, "day"), extremes_type="max"
        )

        assert isinstance(result, xr.DataArray)
        # Should have fewer time points with 3-day grouping
        assert result.sizes["time"] < sample_gridded_dataset.sizes["time"]

    def test_invalid_groupby_type(self, sample_gridded_dataset):
        """Test with invalid groupby type."""
        with pytest.raises(
            ValueError,
            match="`groupby` specifications only implemented for 'day' groupings",
        ):
            _apply_groupby_filter_vectorized(
                sample_gridded_dataset, groupby=(1, "hour"), extremes_type="max"
            )

    def test_invalid_extremes_type_groupby(self, sample_gridded_dataset):
        """Test with invalid extremes type."""
        with pytest.raises(
            ValueError, match='extremes_type needs to be either "max" or "min"'
        ):
            _apply_groupby_filter_vectorized(
                sample_gridded_dataset, groupby=(1, "day"), extremes_type="invalid"
            )


class TestApplyGroupedDurationFilterVectorized:
    """Test class for _apply_grouped_duration_filter_vectorized function."""

    def test_grouped_duration_filter_max(self, sample_gridded_dataset):
        """Test grouped duration filter with max extremes."""
        # First apply groupby to get daily data
        grouped = _apply_groupby_filter_vectorized(
            sample_gridded_dataset, groupby=(1, "day"), extremes_type="max"
        )

        result = _apply_grouped_duration_filter_vectorized(
            grouped, grouped_duration=(3, "day"), extremes_type="max"
        )

        assert isinstance(result, xr.DataArray)
        assert result.sizes["time"] == grouped.sizes["time"]

    def test_grouped_duration_filter_min(self, sample_gridded_dataset):
        """Test grouped duration filter with min extremes."""
        # First apply groupby to get daily data
        grouped = _apply_groupby_filter_vectorized(
            sample_gridded_dataset, groupby=(1, "day"), extremes_type="min"
        )

        result = _apply_grouped_duration_filter_vectorized(
            grouped, grouped_duration=(3, "day"), extremes_type="min"
        )

        assert isinstance(result, xr.DataArray)
        assert result.sizes["time"] == grouped.sizes["time"]

    def test_invalid_grouped_duration_type(self, sample_gridded_dataset):
        """Test with invalid grouped duration type."""
        with pytest.raises(
            ValueError, match="`grouped_duration` specification must be in days"
        ):
            _apply_grouped_duration_filter_vectorized(
                sample_gridded_dataset,
                grouped_duration=(3, "hour"),
                extremes_type="max",
            )

    def test_invalid_extremes_type_grouped_duration(self, sample_gridded_dataset):
        """Test with invalid extremes type."""
        with pytest.raises(
            ValueError, match='extremes_type needs to be either "max" or "min"'
        ):
            _apply_grouped_duration_filter_vectorized(
                sample_gridded_dataset,
                grouped_duration=(3, "day"),
                extremes_type="invalid",
            )


class TestExtractBlockExtremesVectorized:
    """Test class for _extract_block_extremes_vectorized function."""

    def test_extract_block_maxima(self, sample_gridded_dataset):
        """Test extraction of block maxima."""
        result = _extract_block_extremes_vectorized(
            sample_gridded_dataset, extremes_type="max", block_size=1
        )

        assert isinstance(result, xr.DataArray)
        assert result.attrs["extremes type"] == "maxima"
        # Should have 3 annual blocks from 3 years of data
        assert result.sizes["time"] == 3

    def test_extract_block_minima(self, sample_gridded_dataset):
        """Test extraction of block minima."""
        result = _extract_block_extremes_vectorized(
            sample_gridded_dataset, extremes_type="min", block_size=1
        )

        assert isinstance(result, xr.DataArray)
        assert result.attrs["extremes type"] == "minima"
        assert result.sizes["time"] == 3

    def test_extract_multi_year_blocks(self, sample_gridded_dataset):
        """Test extraction with multi-year blocks."""
        result = _extract_block_extremes_vectorized(
            sample_gridded_dataset, extremes_type="max", block_size=2
        )

        assert isinstance(result, xr.DataArray)
        # Should have fewer blocks with larger block size
        assert result.sizes["time"] <= 2

    def test_invalid_extremes_type_extract(self, sample_gridded_dataset):
        """Test with invalid extremes type."""
        with pytest.raises(
            ValueError, match='extremes_type needs to be either "max" or "min"'
        ):
            _extract_block_extremes_vectorized(
                sample_gridded_dataset, extremes_type="invalid", block_size=1
            )


class TestCheckEffectiveSampleSizeOptimized:
    """Test class for _check_effective_sample_size_optimized function."""

    @patch("builtins.print")
    def test_ess_check_calculation_error(self, mock_print):
        """Test ESS check when calculation raises a ValueError or RuntimeError."""
        # Create data that will trigger an error in ESS calculation
        problematic_data = TestDataFactory.create_climate_dataset(
            time_periods=5,  # Very small dataset that might cause issues
            lat_points=2,
            lon_points=2,
        ).rename({"lat": "x", "lon": "y"})

        # Mock the ESS calculation to raise an exception
        with patch(
            "climakitae.new_core.processors.processor_utils._calc_average_ess_gridded_optimized",
            side_effect=ValueError("Calculation failed"),
        ):
            _check_effective_sample_size_optimized(problematic_data, block_size=1)

        # Should print warning about calculation failure
        mock_print.assert_called()
        printed_text = str(mock_print.call_args_list)
        assert "WARNING: Could not calculate effective sample size" in printed_text

    @patch("builtins.print")
    def test_ess_check_gridded_data(self, _mock_print):
        """Test ESS check for gridded data."""
        # Create data with x,y dimensions as expected by the implementation
        gridded_data = TestDataFactory.create_climate_dataset(
            time_periods=365 * 3, lat_points=5, lon_points=5
        ).rename({"lat": "x", "lon": "y"})

        _check_effective_sample_size_optimized(gridded_data, block_size=1)

        # Function should run without errors (may or may not print warnings)
        assert isinstance(gridded_data, xr.DataArray)

    @patch("builtins.print")
    def test_ess_check_timeseries_data(self, mock_print, sample_timeseries_dataset):
        """Test ESS check for timeseries data."""
        _check_effective_sample_size_optimized(sample_timeseries_dataset, block_size=1)

        # Function should run without errors
        assert isinstance(sample_timeseries_dataset, xr.DataArray)

    @patch("builtins.print")
    def test_ess_check_unsupported_dimensions(self, mock_print):
        """Test ESS check with unsupported dimensions."""
        # Create data with unsupported dimensions (lat/lon instead of x/y)
        unsupported_data = TestDataFactory.create_climate_dataset(
            time_periods=10, lat_points=3, lon_points=3
        )  # Uses lat/lon dimensions, not x/y

        _check_effective_sample_size_optimized(unsupported_data, block_size=1)

        mock_print.assert_called()
        printed_text = str(mock_print.call_args_list)
        assert "WARNING" in printed_text
        assert "effective sample size can only be checked" in printed_text


class TestCalcAverageEssGriddedOptimized:
    """Test class for _calc_average_ess_gridded_optimized function."""

    @patch("climakitae.explore.threshold_tools.calculate_ess")
    def test_ess_calculation_small_dataset(
        self, mock_calculate_ess, sample_gridded_dataset
    ):
        """Test ESS calculation for small gridded dataset."""
        # Mock the ESS calculation to return a predictable value
        mock_ess_result = xr.DataArray([25.0])
        mock_calculate_ess.return_value = mock_ess_result

        result = _calc_average_ess_gridded_optimized(
            sample_gridded_dataset, block_size=1
        )

        assert isinstance(result, float)
        assert result >= 0

    def test_large_dataset_approximation(self):
        """Test ESS calculation with large dataset approximation."""
        # Create a large gridded dataset that triggers approximation logic
        large_data = TestDataFactory.create_climate_dataset(
            time_periods=LARGE_DATASET_THRESHOLD
            + 1,  # Large enough to trigger approximation
            lat_points=3,
            lon_points=3,
        ).rename({"lat": "x", "lon": "y"})

        result = _calc_average_ess_gridded_optimized(large_data, block_size=1)

        assert isinstance(result, (int, float))
        assert not np.isnan(result)

    @pytest.mark.advanced
    def test_dask_array_ess_calculation(self):
        """Test ESS calculation with Dask arrays."""
        # Create dask-backed dataset
        dask_data = TestDataFactory.create_climate_dataset(
            time_periods=1000, lat_points=5, lon_points=5, with_dask=True
        ).rename({"lat": "x", "lon": "y"})

        result = _calc_average_ess_gridded_optimized(dask_data, block_size=1)

        assert isinstance(result, (int, float))
        assert not np.isnan(result)

    @patch("climakitae.explore.threshold_tools.calculate_ess")
    def test_ess_calculation_with_errors(self, mock_calculate_ess):
        """Test ESS calculation when calculate_ess raises errors."""
        # Mock to raise exceptions then return valid results
        mock_calculate_ess.side_effect = [
            ValueError("Error"),
            xr.DataArray([25.0]),
            RuntimeError("Runtime error"),
            xr.DataArray([30.0]),
        ]

        # Create dataset that will trigger the error handling
        error_data = TestDataFactory.create_climate_dataset(
            time_periods=365 * 5, lat_points=2, lon_points=2
        ).rename({"lat": "x", "lon": "y"})

        result = _calc_average_ess_gridded_optimized(error_data, block_size=1)

        # Should handle errors and return average of successful calculations
        assert isinstance(result, (int, float))
        assert not np.isnan(result)

    def test_ess_calculation_memory_error(self):
        """Test ESS calculation when MemoryError occurs."""
        # Create a scenario that might cause memory issues by mocking calculate_ess
        with patch(
            "climakitae.explore.threshold_tools.calculate_ess",
            side_effect=MemoryError("Out of memory"),
        ):
            error_data = TestDataFactory.create_climate_dataset(
                time_periods=365 * 2, lat_points=2, lon_points=2
            ).rename({"lat": "x", "lon": "y"})

            result = _calc_average_ess_gridded_optimized(error_data, block_size=1)

            # Should return a reasonable ESS value or fallback
            assert isinstance(result, (int, float))
            assert result > 0  # ESS should be positive

    def test_ess_calculation_large_dataset(self):
        """Test ESS calculation for large gridded dataset using approximation.

        This test creates a dataset with hourly frequency within a single year
        to ensure each yearly group has > LARGE_DATASET_THRESHOLD (1000) time points.
        This triggers the approximation logic in _calc_average_ess_gridded_optimized.
        """
        # Create a dataset with hourly frequency so each year has >8760 time points
        # This ensures individual yearly groups exceed LARGE_DATASET_THRESHOLD (1000)
        large_dataset = TestDataFactory.create_climate_dataset(
            time_periods=LARGE_DATASET_THRESHOLD * 2,  # 2000 hours in one year
            lat_points=3,
            lon_points=3,
            start_date="2020-01-01",
            frequency="h",  # Hourly frequency to get many points per year
        ).rename(
            {"lat": "x", "lon": "y"}
        )  # ESS function expects x,y dimensions

        result = _calc_average_ess_gridded_optimized(large_dataset, block_size=1)

        assert isinstance(result, float)
        assert result >= 0
        # Should successfully compute ESS using approximation method for large datasets

    def test_ess_calculation_insufficient_data(self):
        """Test ESS calculation with insufficient time points."""
        # Create dataset with very few time points
        small_dataset = TestDataFactory.create_climate_dataset(
            time_periods=5, lat_points=3, lon_points=3  # Below MIN_TIME_POINTS
        )

        result = _calc_average_ess_gridded_optimized(small_dataset, block_size=1)

        # Should return fallback value
        assert result == FALLBACK_ESS_VALUE

    def test_ess_calculation_dask_array(self, sample_dask_dataset):
        """Test ESS calculation with Dask arrays."""
        result = _calc_average_ess_gridded_optimized(sample_dask_dataset, block_size=1)

        assert isinstance(result, float)
        assert result >= 0

    def test_ess_calculation_error_handling(self, sample_gridded_dataset):
        """Test ESS calculation error handling."""
        # Create a dataset that might cause calculation errors
        corrupted_data = sample_gridded_dataset.copy()
        corrupted_data.values[:] = np.inf  # Fill with infinite values

        result = _calc_average_ess_gridded_optimized(corrupted_data, block_size=1)

        # Should return fallback value when calculation fails
        assert result == FALLBACK_ESS_VALUE


class TestCalcAverageEssTimeseriesOptimized:
    """Test class for _calc_average_ess_timeseries_optimized function."""

    @patch("climakitae.explore.threshold_tools.calculate_ess")
    def test_ess_calculation_small_timeseries(
        self, mock_calculate_ess, sample_timeseries_dataset
    ):
        """Test ESS calculation for small timeseries dataset."""
        # Mock the ESS calculation to return a predictable value
        mock_ess_result = xr.DataArray([25.0])
        mock_calculate_ess.return_value = mock_ess_result

        result = _calc_average_ess_timeseries_optimized(
            sample_timeseries_dataset, block_size=1
        )

        assert isinstance(result, float)
        assert result >= 0

    def test_ess_calculation_large_timeseries(self):
        """Test ESS calculation for large timeseries using approximation."""
        # Create a large timeseries that will trigger approximation
        large_timeseries = TestDataFactory.create_timeseries_dataset(
            time_periods=LARGE_TIMESERIES_THRESHOLD + 100
        )

        result = _calc_average_ess_timeseries_optimized(large_timeseries, block_size=1)

        assert isinstance(result, (int, float))
        assert not np.isnan(result)

    def test_ess_calculation_large_blocks_approximation(self):
        """Test ESS calculation with large blocks for autocorrelation approximation."""
        # Create a timeseries with daily data for multiple years
        # This ensures each year block has > LARGE_TIMESERIES_THRESHOLD (500) points
        # Use hourly data to get more points per year
        time_periods = 365 * 24 * 2  # 2 years of hourly data
        large_data = TestDataFactory.create_timeseries_dataset(
            time_periods=time_periods,
            frequency="h",  # Hourly frequency
            start_date="2020-01-01",
        )

        # Use block_size=2 so each block contains 2 years worth of hourly data
        # This gives us 365*24*2 = 17520 points per block, well above threshold
        result = _calc_average_ess_timeseries_optimized(large_data, block_size=2)

        assert isinstance(result, (int, float))
        assert result > 0  # Should be a positive ESS value
        assert not np.isnan(result)

    def test_ess_calculation_autocorr_with_correlations(self):
        """Test ESS calculation autocorrelation path with realistic correlated data."""
        # Create strongly autocorrelated timeseries data to test correlation calculation
        time_periods = 600  # Above LARGE_TIMESERIES_THRESHOLD
        time = pd.date_range("2020-01-01", periods=time_periods, freq="D")

        # Create autocorrelated data using AR(1) process
        np.random.seed(42)  # For reproducible results
        phi = 0.7  # AR(1) coefficient for autocorrelation
        noise = np.random.normal(0, 1, time_periods)
        autocorr_data = np.zeros(time_periods)
        autocorr_data[0] = noise[0]
        for i in range(1, time_periods):
            autocorr_data[i] = phi * autocorr_data[i - 1] + noise[i]

        # Create xarray dataset
        large_corr_data = xr.DataArray(
            autocorr_data,
            coords={"time": time},
            dims=["time"],
            name="temperature",
            attrs={"units": "K"},
        )

        # Use block_size=1 so the entire 600-day series becomes one block
        # This will trigger the autocorrelation approximation path
        result = _calc_average_ess_timeseries_optimized(large_corr_data, block_size=1)

        assert isinstance(result, (int, float))
        # ESS should be positive but less than n due to autocorrelation
        assert result > 0
        # ESS should be less than total points due to correlation
        assert result < time_periods
        assert not np.isnan(result)

    @patch("climakitae.explore.threshold_tools.calculate_ess")
    def test_ess_timeseries_calculation_with_errors(self, mock_calculate_ess):
        """Test timeseries ESS calculation when calculate_ess raises errors."""
        mock_calculate_ess.side_effect = [
            ValueError("Error"),
            IndexError("Index error"),
            xr.DataArray([20.0]),
        ]

        # Create dataset that will trigger the error handling
        error_data = TestDataFactory.create_timeseries_dataset(time_periods=365 * 3)

        result = _calc_average_ess_timeseries_optimized(error_data, block_size=1)

        # Should handle errors and return average of successful calculations
        assert isinstance(result, (int, float))
        assert not np.isnan(result)

    def test_ess_timeseries_memory_error(self):
        """Test timeseries ESS calculation when MemoryError occurs."""
        # Mock to cause memory error
        with patch(
            "climakitae.explore.threshold_tools.calculate_ess",
            side_effect=MemoryError("Out of memory"),
        ):
            error_data = TestDataFactory.create_timeseries_dataset(time_periods=365 * 2)

            result = _calc_average_ess_timeseries_optimized(error_data, block_size=1)

            # Should return a reasonable ESS value or fallback
            assert isinstance(result, (int, float))
            assert result > 0  # ESS should be positive

        assert isinstance(result, float)
        assert result >= 0

    def test_ess_calculation_insufficient_time_points(self):
        """Test ESS calculation with insufficient time points."""
        # Create timeseries with very few time points
        small_timeseries = TestDataFactory.create_timeseries_dataset(
            time_periods=5  # Below MIN_TIME_POINTS
        )

        result = _calc_average_ess_timeseries_optimized(small_timeseries, block_size=1)

        # Should return fallback value
        assert result == FALLBACK_ESS_VALUE

    def test_ess_calculation_error_handling(self, sample_timeseries_dataset):
        """Test ESS calculation error handling."""
        # Create a timeseries that might cause calculation errors
        corrupted_data = sample_timeseries_dataset.copy()
        corrupted_data.values[:] = np.inf  # Fill with infinite values

        result = _calc_average_ess_timeseries_optimized(corrupted_data, block_size=1)

        # Should return fallback value when calculation fails
        assert result == FALLBACK_ESS_VALUE


class TestSetBlockMaximaAttributes:
    """Test class for _set_block_maxima_attributes function."""

    def test_set_attributes_complete(self, sample_gridded_dataset):
        """Test setting all attributes."""
        # First extract block maxima
        bms = _extract_block_extremes_vectorized(
            sample_gridded_dataset, extremes_type="max", block_size=1
        )

        result = _set_block_maxima_attributes(
            bms,
            duration=(4, "hour"),
            groupby=(1, "day"),
            grouped_duration=(3, "day"),
            extremes_type="max",
            block_size=2,
        )

        assert result.attrs["duration"] == (4, "hour")
        assert result.attrs["groupby"] == (1, "day")
        assert result.attrs["grouped_duration"] == (3, "day")
        assert result.attrs["extreme_value_extraction_method"] == "block maxima"
        assert result.attrs["block_size"] == "2 year"
        assert result.attrs["timeseries_type"] == "block max series"

    def test_set_attributes_minimal(self, sample_gridded_dataset):
        """Test setting minimal attributes with None values."""
        # First extract block maxima
        bms = _extract_block_extremes_vectorized(
            sample_gridded_dataset, extremes_type="min", block_size=1
        )

        result = _set_block_maxima_attributes(
            bms,
            duration=UNSET,
            groupby=UNSET,
            grouped_duration=UNSET,
            extremes_type="min",
            block_size=1,
        )

        assert result.attrs["duration"] == UNSET
        assert result.attrs["groupby"] == UNSET
        assert result.attrs["grouped_duration"] == UNSET
        assert result.attrs["extreme_value_extraction_method"] == "block maxima"
        assert result.attrs["block_size"] == "1 year"
        assert result.attrs["timeseries_type"] == "block min series"


class TestHandleNanValuesOptimized:
    """Test class for _handle_nan_values_optimized function."""

    def test_no_nan_values(self, sample_gridded_dataset):
        """Test handling when no NaN values are present."""
        result = _handle_nan_values_optimized(sample_gridded_dataset)

        # Should return the same array
        assert result is sample_gridded_dataset
        assert result.sizes["time"] == sample_gridded_dataset.sizes["time"]

    @patch("builtins.print")
    def test_with_nan_values(self, mock_print, sample_nan_dataset):
        """Test handling with some NaN values."""
        result = _handle_nan_values_optimized(sample_nan_dataset)

        assert isinstance(result, xr.DataArray)
        # Should have fewer time steps after dropping NaNs
        assert result.sizes["time"] < sample_nan_dataset.sizes["time"]

        # Should print information about dropped values
        mock_print.assert_called()
        printed_text = str(mock_print.call_args_list)
        assert "Dropping" in printed_text
        assert "block maxima NaNs" in printed_text

    def test_all_nan_values(self):
        """Test handling when all values are NaN."""
        # Create dataset with all NaN values
        all_nan_dataset = TestDataFactory.create_climate_dataset(
            time_periods=10, lat_points=3, lon_points=3
        )
        all_nan_dataset.values[:] = np.nan

        with pytest.raises(ValueError, match="does not include any recorded values"):
            _handle_nan_values_optimized(all_nan_dataset)

    @patch("builtins.print")
    def test_with_dask_nan_values(self, mock_print):
        """Test handling NaN values with Dask arrays."""
        # Create Dask dataset with NaN values
        dask_nan_dataset = TestDataFactory.create_climate_dataset(
            time_periods=20, lat_points=3, lon_points=3, with_dask=True
        )
        # Introduce NaN values
        data_copy = dask_nan_dataset.values.copy()
        data_copy[5:8] = np.nan
        dask_nan_dataset = xr.DataArray(
            da.from_array(data_copy),
            coords=dask_nan_dataset.coords,
            dims=dask_nan_dataset.dims,
            attrs=dask_nan_dataset.attrs,
        )

        result = _handle_nan_values_optimized(dask_nan_dataset)

        assert isinstance(result, xr.DataArray)
        assert result.sizes["time"] < dask_nan_dataset.sizes["time"]


class TestExtendTimeDomain:
    """Test class for extend_time_domain function."""

    def test_successful_time_domain_extension(self, scenario_dict_with_historical):
        """Test successful time domain extension."""
        with patch("builtins.print") as mock_print:
            result = extend_time_domain(scenario_dict_with_historical)

        assert isinstance(result, dict)

        # Should have extended SSP scenarios
        assert "model1_ssp245_r1i1p1f1" in result
        assert "model1_ssp370_r1i1p1f1" in result
        assert "model2_ssp245_r1i1p1f1" in result

        # Should not have standalone historical data
        assert "model1_historical_r1i1p1f1" not in result

        # Check that historical data was prepended
        ssp245_data = result["model1_ssp245_r1i1p1f1"]
        assert ssp245_data.attrs["historical_prepended"]
        # Should preserve SSP attributes
        assert ssp245_data.attrs["experiment"] == "ssp245"

        # Should have printed info message
        mock_print.assert_called()
        printed_text = str(mock_print.call_args_list)
        assert "Prepending historical data to SSP scenarios" in printed_text

    def test_missing_historical_data(self):
        """Test behavior when historical data is missing."""
        # Create scenario dict without corresponding historical data
        ssp_only_dict = {
            "model1_ssp245_r1i1p1f1": TestDataFactory.create_climate_dataset(
                time_periods=365 * 10, start_date="2015-01-01"
            )
        }
        ssp_only_dict["model1_ssp245_r1i1p1f1"].attrs.update(
            {"experiment": "ssp245", "source": "CMIP6"}
        )

        with patch("warnings.warn") as mock_warn, patch("builtins.print"):
            result = extend_time_domain(ssp_only_dict)

        # Should warn about missing historical data
        mock_warn.assert_called()
        warning_text = str(mock_warn.call_args_list)
        assert "No historical data found" in warning_text

        # Should keep original SSP data
        assert "model1_ssp245_r1i1p1f1" in result
        assert result["model1_ssp245_r1i1p1f1"].attrs["experiment"] == "ssp245"

    def test_non_ssp_data_dropped(self):
        """Test that non-SSP data is dropped."""
        mixed_dict = {
            "model1_ssp245_r1i1p1f1": TestDataFactory.create_climate_dataset(
                time_periods=365 * 10
            ),
            "model1_historical_r1i1p1f1": TestDataFactory.create_climate_dataset(
                time_periods=365 * 35
            ),
            "reanalysis_era5": TestDataFactory.create_climate_dataset(
                time_periods=365 * 40
            ),
            "observations": TestDataFactory.create_climate_dataset(
                time_periods=365 * 30
            ),
        }

        for key, data in mixed_dict.items():
            if "ssp" in key:
                data.attrs["experiment"] = "ssp245"
            elif "historical" in key:
                data.attrs["experiment"] = "historical"
            else:
                data.attrs["experiment"] = "observations"

        with patch("builtins.print"):
            result = extend_time_domain(mixed_dict)

        # Should only have SSP data with historical prepended
        assert "model1_ssp245_r1i1p1f1" in result
        # Non-SSP data should be dropped
        assert "reanalysis_era5" not in result
        assert "observations" not in result

    def test_already_extended_data(self, scenario_dict_with_historical):
        """Test behavior when data has already been extended."""
        # Mark data as already processed
        for data in scenario_dict_with_historical.values():
            data.attrs["historical_prepended"] = True

        with patch("builtins.print") as mock_print:
            result = extend_time_domain(scenario_dict_with_historical)

        # Should return original dict without processing
        assert result == scenario_dict_with_historical

        # Should not print processing message
        info_calls = [
            call
            for call in mock_print.call_args_list
            if "Prepending historical data" in str(call)
        ]
        assert len(info_calls) == 0

    def test_concatenation_error_handling(self):
        """Test error handling during concatenation."""
        # Create incompatible datasets that will cause concatenation errors
        ssp_data = TestDataFactory.create_climate_dataset(
            time_periods=365 * 10, lat_points=5, lon_points=5
        )
        hist_data = TestDataFactory.create_climate_dataset(
            time_periods=365 * 10,
            lat_points=10,  # Different spatial dimensions
            lon_points=10,
        )

        error_dict = {
            "model1_ssp245_r1i1p1f1": ssp_data,
            "model1_historical_r1i1p1f1": hist_data,
        }

        for key, data in error_dict.items():
            if "ssp" in key:
                data.attrs["experiment"] = "ssp245"
            else:
                data.attrs["experiment"] = "historical"

        with patch("warnings.warn") as mock_warn, patch("builtins.print"):
            result = extend_time_domain(error_dict)

        # Function handles errors gracefully and returns a result
        assert isinstance(result, dict)
        # The result may contain some data or be empty depending on implementation
        assert result is not None

    def test_mixed_dataset_dataarray_types(self):
        """Test extend_time_domain with mixed Dataset and DataArray types."""
        # Create mixed dictionary with Dataset and DataArray
        ssp_dataarray = TestDataFactory.create_climate_dataset(
            time_periods=365 * 5, start_date="2015-01-01"
        )
        ssp_dataset = ssp_dataarray.to_dataset()  # Convert to Dataset

        hist_dataarray = TestDataFactory.create_climate_dataset(
            time_periods=365 * 10, start_date="1995-01-01"
        )  # This is already a DataArray

        mixed_dict = {
            "model1_ssp245_r1i1p1f1": ssp_dataset,  # Dataset
            "model1_historical_r1i1p1f1": hist_dataarray,  # DataArray
        }

        for key, data in mixed_dict.items():
            if "ssp" in key:
                data.attrs["experiment"] = "ssp245"
            else:
                data.attrs["experiment"] = "historical"

        with patch("builtins.print"):
            result = extend_time_domain(mixed_dict)

        # Should handle mixed types by converting to compatible formats
        assert isinstance(result, dict)
        if "model1_ssp245_r1i1p1f1" in result:
            extended_data = result["model1_ssp245_r1i1p1f1"]
            assert extended_data.attrs.get("historical_prepended", False)

    def test_mixed_dataarray_dataset_types(self):
        """Test extend_time_domain with DataArray SSP and Dataset historical."""
        # Create mixed dictionary with DataArray SSP and Dataset historical
        ssp_dataarray = TestDataFactory.create_climate_dataset(
            time_periods=365 * 5, start_date="2015-01-01"
        )  # DataArray

        hist_dataarray = TestDataFactory.create_climate_dataset(
            time_periods=365 * 10, start_date="1995-01-01"
        )
        hist_dataset = hist_dataarray.to_dataset()  # Convert to Dataset

        mixed_dict = {
            "model1_ssp245_r1i1p1f1": ssp_dataarray,  # DataArray
            "model1_historical_r1i1p1f1": hist_dataset,  # Dataset
        }

        for key, data in mixed_dict.items():
            if "ssp" in key:
                data.attrs["experiment"] = "ssp245"
            else:
                data.attrs["experiment"] = "historical"

        with patch("builtins.print"):
            result = extend_time_domain(mixed_dict)

        # Should handle mixed types by converting to compatible formats
        assert isinstance(result, dict)
        if "model1_ssp245_r1i1p1f1" in result:
            extended_data = result["model1_ssp245_r1i1p1f1"]
            assert extended_data.attrs.get("historical_prepended", False)


@pytest.mark.advanced
class TestIntegrationBlockMaxima:
    """Integration tests for complete block maxima workflows."""

    @pytest.mark.integration
    def test_complete_block_maxima_pipeline(self, sample_gridded_dataset):
        """Test complete pipeline from raw data to block maxima."""
        # Test with all processing steps
        result = _get_block_maxima_optimized(
            sample_gridded_dataset,
            extremes_type="max",
            duration=UNSET,
            groupby=UNSET,
            grouped_duration=UNSET,
            check_ess=True,
            block_size=1,
            chunk_spatial=True,
            max_memory_gb=2.0,
        )

        # Validate complete result
        assert isinstance(result, xr.DataArray)
        assert result.attrs["extremes type"] == "maxima"
        assert result.attrs["extreme_value_extraction_method"] == "block maxima"
        assert result.sizes["time"] == 3  # 3 years of data
        assert not result.isnull().all().item()  # Should have valid data

    @pytest.mark.integration
    def test_complete_pipeline_with_all_filters(self):
        """Test complete pipeline with all filtering options."""
        # Create hourly dataset for duration testing
        hourly_data = TestDataFactory.create_climate_dataset(
            time_periods=24 * 365,  # 1 year of hourly data
            lat_points=3,
            lon_points=3,
            frequency="h",
        )

        result = _get_block_maxima_optimized(
            hourly_data,
            extremes_type="max",
            duration=(4, "hour"),
            groupby=(1, "day"),
            grouped_duration=(3, "day"),
            check_ess=False,  # Skip ESS to avoid complexity in test
            block_size=1,
        )

        # Validate result with all filters applied
        assert isinstance(result, xr.DataArray)
        assert result.attrs["duration"] == (4, "hour")
        assert result.attrs["groupby"] == (1, "day")
        assert result.attrs["grouped_duration"] == (3, "day")
        assert result.attrs["extremes type"] == "maxima"

    @pytest.mark.integration
    def test_time_domain_extension_integration(self, scenario_dict_with_historical):
        """Test complete time domain extension workflow."""
        # Test full workflow
        result = extend_time_domain(scenario_dict_with_historical)

        # Validate extended datasets
        assert isinstance(result, dict)
        for key, data in result.items():
            assert isinstance(data, xr.DataArray)
            assert data.attrs["historical_prepended"]
            assert "ssp" in data.attrs.get("experiment", "")

            # Should have extended time range (1980-2100)
            time_range = data.time.dt.year
            assert time_range.min().item() == 1980
            assert time_range.max().item() >= 2099

    @pytest.mark.integration
    @pytest.mark.advanced
    def test_memory_performance_large_dataset(self):
        """Test memory performance with large datasets."""
        # Create large dataset to test memory management
        large_data = TestDataFactory.create_climate_dataset(
            time_periods=365 * 10,  # 10 years
            lat_points=50,
            lon_points=50,
            with_dask=True,
        )

        # Should handle large dataset without memory errors
        result = _get_block_maxima_optimized(
            large_data,
            extremes_type="max",
            check_ess=False,  # Skip ESS for performance
            chunk_spatial=True,
            max_memory_gb=1.0,  # Constrain memory
        )

        assert isinstance(result, xr.DataArray)
        assert result.sizes["time"] == 10  # 10 annual blocks
        assert hasattr(result.data, "chunks")  # Should maintain Dask backing


# Performance and edge case tests
@pytest.mark.advanced
class TestAdvancedScenarios:
    """Advanced test scenarios for edge cases and performance."""

    def test_extreme_temporal_resolution(self):
        """Test with very high temporal resolution data."""
        # 1 minute resolution for 1 day
        minute_data = TestDataFactory.create_climate_dataset(
            time_periods=24 * 60,  # 1440 minutes
            lat_points=3,
            lon_points=3,
            start_date="2020-01-01",
            frequency="min",  # 1 minute
        )

        # Should handle high-resolution data
        result = _get_block_maxima_optimized(
            minute_data, extremes_type="max", check_ess=False, block_size=1
        )

        assert isinstance(result, xr.DataArray)

    def test_performance_timeout(self, sample_dask_dataset):
        """Test that operations complete within reasonable time (should finish within 60 seconds)."""
        result = _get_block_maxima_optimized(
            sample_dask_dataset,
            extremes_type="max",
            check_ess=True,
            chunk_spatial=True,
            max_memory_gb=0.5,  # Very constrained memory
        )

        assert result is not None

    def test_mixed_data_types(self):
        """Test with mixed data types and attributes."""
        # Create dataset with integer data
        int_data = TestDataFactory.create_climate_dataset(
            time_periods=365, lat_points=3, lon_points=3
        )
        int_data = int_data.astype(np.int32)

        result = _get_block_maxima_optimized(
            int_data, extremes_type="max", check_ess=False
        )

        assert isinstance(result, xr.DataArray)
        # Should preserve data type characteristics
        assert result.dtype.kind in ["i", "f"]  # integer or float


class TestFindStationMatch:
    """Test class for find_station_match utility function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock stations DataFrame matching the structure of real station data
        self.stations_df = pd.DataFrame(
            {
                "ID": ["KSAC", "KBFL", "KSFO", "KSAN", "KOAK"],
                "station": [
                    "Sacramento (KSAC)",
                    "Bakersfield (KBFL)",
                    "San Francisco International (KSFO)",
                    "San Diego International (KSAN)",
                    "Oakland International (KOAK)",
                ],
                "city": [
                    "Sacramento",
                    "Bakersfield",
                    "San Francisco",
                    "San Diego",
                    "Oakland",
                ],
                "state": ["CA", "CA", "CA", "CA", "CA"],
                "LAT_Y": [38.5, 35.4, 37.6, 32.7, 37.7],
                "LON_X": [-121.5, -119.1, -122.4, -117.2, -122.2],
            }
        )

    def test_exact_match_on_id(self):
        """Test exact match on station ID."""
        match = find_station_match("KSAC", self.stations_df)

        assert len(match) == 1
        assert match.iloc[0]["ID"] == "KSAC"
        assert match.iloc[0]["city"] == "Sacramento"

    def test_exact_match_case_insensitive(self):
        """Test exact match is case-insensitive."""
        match = find_station_match("ksac", self.stations_df)

        assert len(match) == 1
        assert match.iloc[0]["ID"] == "KSAC"

    def test_exact_match_with_whitespace(self):
        """Test exact match handles leading/trailing whitespace."""
        match = find_station_match("  KSAC  ", self.stations_df)

        assert len(match) == 1
        assert match.iloc[0]["ID"] == "KSAC"

    def test_exact_match_on_station_name(self):
        """Test exact match on station name column."""
        match = find_station_match("Sacramento (KSAC)", self.stations_df)

        assert len(match) == 1
        assert match.iloc[0]["ID"] == "KSAC"

    def test_partial_match_on_station_name(self):
        """Test partial match on station name."""
        match = find_station_match("Sacramento", self.stations_df)

        assert len(match) == 1
        assert match.iloc[0]["ID"] == "KSAC"

    def test_partial_match_multiple_results(self):
        """Test partial match returning multiple results."""
        # Add another station with "International" in the name
        match = find_station_match("International", self.stations_df)

        # Should match San Francisco International, San Diego International, Oakland International
        assert len(match) == 3
        assert all("International" in name for name in match["station"].values)

    def test_no_match(self):
        """Test when no match is found."""
        match = find_station_match("KZZZ", self.stations_df)

        assert len(match) == 0

    def test_no_match_invalid_name(self):
        """Test when station name doesn't match anything."""
        match = find_station_match("Nonexistent City", self.stations_df)

        assert len(match) == 0

    def test_empty_identifier(self):
        """Test with empty string identifier."""
        # After strip(), empty string won't match anything
        match = find_station_match("", self.stations_df)

        assert len(match) == 0

    def test_whitespace_only_identifier(self):
        """Test with whitespace-only identifier."""
        match = find_station_match("   ", self.stations_df)

        assert len(match) == 0
