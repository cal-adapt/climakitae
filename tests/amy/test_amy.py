"""
Unit tests for climakitae/explore/amy.py

This module contains comprehensive unit tests for the AMY (Average Meteorological Year)
and climate profile computation functions that provide climate profile analysis.
"""

import pytest
import pandas as pd
import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch

from climakitae.explore.amy import (
    retrieve_profile_data,
    get_climate_profile,
    compute_profile,
    get_profile_units,
    get_profile_metadata,
    set_profile_metadata,
    _compute_difference_profile,
    _compute_multiindex_difference,
    _compute_simulation_paired_difference,
    _compute_warming_level_difference,
    _compute_mixed_index_difference,
    _compute_simple_difference,
    _find_matching_historic_column,
    _get_historic_hour_mean,
    _find_matching_historic_value,
    _format_based_on_structure,
    _construct_profile_dataframe,
    _create_simple_dataframe,
    _create_single_wl_multi_sim_dataframe,
    _create_multi_wl_single_sim_dataframe,
    _create_multi_wl_multi_sim_dataframe,
    _stack_profile_data,
    _format_meteo_yr_df,
)


class TestRetrieveProfileData:
    """Test class for retrieve_profile_data function.

    Tests the function's ability to retrieve climate profile data based on
    various parameter configurations and location specifications.

    Attributes
    ----------
    mock_get_data : MagicMock
        Mock for get_data function.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_get_data_patcher = patch("climakitae.explore.amy.get_data")
        self.mock_get_data = self.mock_get_data_patcher.start()

    def teardown_method(self):
        """Clean up test fixtures."""
        self.mock_get_data_patcher.stop()

    def test_retrieve_profile_data_returns_tuple(self):
        """Test that retrieve_profile_data returns a tuple of two datasets."""
        # Setup mock return values
        mock_historic = MagicMock(spec=xr.Dataset)
        mock_future = MagicMock(spec=xr.Dataset)
        self.mock_get_data.side_effect = [mock_historic, mock_future]

        # Execute function
        result = retrieve_profile_data(warming_level=[2.0])

        # Verify outcome: returns tuple of two datasets
        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 2, "Tuple should contain exactly 2 elements"
        historic_data, future_data = result
        assert historic_data == mock_historic, "First element should be historic data"
        assert future_data == mock_future, "Second element should be future data"

    def test_retrieve_profile_data_with_invalid_parameters_raises_error(self):
        """Test that retrieve_profile_data raises error for invalid parameter keys."""
        # Execute and verify outcome: should raise ValueError for invalid keys
        with pytest.raises(ValueError, match="Invalid input"):
            retrieve_profile_data(invalid_param="test", another_invalid=123)

    def test_retrieve_profile_data_with_no_delta_returns_none_historic(self):
        """Test that retrieve_profile_data returns None for historic when no_delta=True."""
        # Setup mock return value
        mock_future = MagicMock(spec=xr.Dataset)
        self.mock_get_data.return_value = mock_future

        # Execute function with no_delta=True
        result = retrieve_profile_data(no_delta=True, warming_level=[2.0])

        # Verify outcome: historic data should be None when no_delta=True
        assert isinstance(result, tuple), "Should return a tuple"
        historic_data, future_data = result
        assert historic_data is None, "Historic data should be None when no_delta=True"
        assert future_data == mock_future, "Future data should be returned"

    def test_retrieve_profile_data_accepts_valid_parameters(self):
        """Test that retrieve_profile_data accepts valid parameter combinations."""
        # Setup mock return values
        mock_historic = MagicMock(spec=xr.Dataset)
        mock_future = MagicMock(spec=xr.Dataset)
        self.mock_get_data.side_effect = [mock_historic, mock_future]

        # Execute function with basic valid parameters
        result = retrieve_profile_data(warming_level=[1.5, 2.0])

        # Verify outcome: function should complete successfully with valid parameters
        assert isinstance(result, tuple), "Should return a tuple with valid parameters"
        assert len(result) == 2, "Should return tuple of two elements"

        # Verify get_data was called twice (once for historic, once for future)
        assert self.mock_get_data.call_count == 2, "Should call get_data twice"

    def test_retrieve_profile_data_with_complex_parameters(self):
        """Test that retrieve_profile_data handles complex parameter combinations."""
        # Setup mock return values
        mock_historic = MagicMock(spec=xr.Dataset)
        mock_future = MagicMock(spec=xr.Dataset)
        self.mock_get_data.side_effect = [mock_historic, mock_future]

        # Execute function with complex valid parameters that previously caused the bug
        result = retrieve_profile_data(
            variable="Air Temperature at 2m",
            resolution="45 km",
            warming_level=[1.5, 2.0],
            cached_area="bay area",
            units="degC",
        )

        # Verify outcome: function should complete successfully
        assert isinstance(result, tuple), "Should handle complex parameters"
        assert len(result) == 2, "Should return two datasets"
        historic_data, future_data = result
        assert historic_data == mock_historic, "Historic data should be returned"
        assert future_data == mock_future, "Future data should be returned"


class TestGetClimateProfile:
    """Test class for get_climate_profile function.

    Tests the high-level function that combines data retrieval and profile
    computation to produce climate profiles as DataFrames.

    Attributes
    ----------
    mock_retrieve_profile_data : MagicMock
        Mock for retrieve_profile_data function.
    mock_compute_profile : MagicMock
        Mock for compute_profile function.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_retrieve_patcher = patch(
            "climakitae.explore.amy.retrieve_profile_data"
        )
        self.mock_compute_patcher = patch("climakitae.explore.amy.compute_profile")
        self.mock_print_patcher = patch("builtins.print")

        self.mock_retrieve_profile_data = self.mock_retrieve_patcher.start()
        self.mock_compute_profile = self.mock_compute_patcher.start()
        self.mock_print = self.mock_print_patcher.start()

    def teardown_method(self):
        """Clean up test fixtures."""
        self.mock_retrieve_patcher.stop()
        self.mock_compute_patcher.stop()
        self.mock_print_patcher.stop()

    def test_get_climate_profile_returns_dataframe(self):
        """Test that get_climate_profile returns a pandas DataFrame."""
        # Setup mock datasets with proper data_vars
        mock_historic_data = MagicMock(spec=xr.Dataset)
        mock_historic_data.data_vars = {"tasmax": MagicMock()}
        mock_future_data = MagicMock(spec=xr.Dataset)
        mock_future_data.data_vars = {"tasmax": MagicMock()}

        self.mock_retrieve_profile_data.return_value = (
            mock_historic_data,
            mock_future_data,
        )

        # Create mock profile DataFrames
        mock_future_profile = pd.DataFrame(np.random.rand(365, 24))
        mock_historic_profile = pd.DataFrame(np.random.rand(365, 24))
        self.mock_compute_profile.side_effect = [
            mock_future_profile,
            mock_historic_profile,
        ]

        # Execute function
        result = get_climate_profile(warming_level=[2.0])

        # Verify outcome: returns a DataFrame
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape[0] > 0, "DataFrame should have rows"
        assert result.shape[1] > 0, "DataFrame should have columns"

    def test_get_climate_profile_with_no_delta_returns_raw_future(self):
        """Test that get_climate_profile returns raw future profile when no_delta=True."""
        # Setup mock datasets with proper data_vars
        mock_future_data = MagicMock(spec=xr.Dataset)
        mock_future_data.data_vars = {"tasmax": MagicMock()}

        # When no_delta=True, only future data is returned, historic is None
        self.mock_retrieve_profile_data.return_value = (None, mock_future_data)

        # Create mock future profile
        mock_future_profile = pd.DataFrame(np.random.rand(365, 24))
        self.mock_compute_profile.return_value = mock_future_profile

        # Execute function with no_delta=True
        result = get_climate_profile(warming_level=[2.0], no_delta=True)

        # Verify outcome: returns the raw future profile (no difference calculation)
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert (
            result.shape == mock_future_profile.shape
        ), "Should return the original future profile shape"
        # Verify compute_profile was called only once (for future data only)
        assert (
            self.mock_compute_profile.call_count == 1
        ), "Should call compute_profile only once for future data"

    def test_get_climate_profile_raises_error_when_no_data_returned(self):
        """Test that get_climate_profile raises ValueError when no data is retrieved."""
        # Setup scenario where both datasets are None
        self.mock_retrieve_profile_data.return_value = (None, None)

        # Execute and verify outcome: should raise ValueError
        with pytest.raises(
            ValueError,
            match="No data returned for either historical or future datasets",
        ):
            get_climate_profile(warming_level=[2.0])


class TestComputeProfile:
    """Test class for compute_profile function.

    Tests the core function that computes climate profiles from xarray DataArrays
    using quantile-based analysis across multiple years of data.

    Attributes
    ----------
    sample_data : xr.DataArray
        Sample climate data for testing.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create smaller sample for faster testing (just enough for the algorithm)
        time_delta = pd.date_range(
            "2020-01-01", periods=8760, freq="h"
        )  # 1 year of hourly data
        warming_levels = [1.5]
        simulations = ["sim1"]

        # Create test data with proper dimensions
        data = np.random.rand(len(warming_levels), len(time_delta), len(simulations))

        self.sample_data = xr.DataArray(
            data,
            dims=["warming_level", "time_delta", "simulation"],
            coords={
                "warming_level": warming_levels,
                "time_delta": time_delta,
                "simulation": simulations,
            },
            attrs={"units": "degC", "variable_id": "tasmax"},
        )

    def test_compute_profile_returns_dataframe_with_correct_shape(self):
        """Test that compute_profile returns DataFrame with expected dimensions."""
        # Execute function
        result = compute_profile(self.sample_data, days_in_year=365, q=0.5)

        # Verify outcome: returns DataFrame with correct shape
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape[0] == 365, "Should have 365 rows (days)"
        assert result.shape[1] > 0, "Should have columns for hours and other dimensions"

        # Verify the DataFrame has proper index and column structure
        assert result.index.dtype == int or isinstance(
            result.index, pd.Index
        ), "Should have proper index"
        assert hasattr(result, "attrs"), "Should preserve metadata in attrs"

    def test_compute_profile_respects_days_in_year_parameter(self):
        """Test that compute_profile creates DataFrame with specified number of days."""
        # Execute function with regular year (365 days) - this should work with 8760 hours
        result_365 = compute_profile(self.sample_data, days_in_year=365, q=0.5)

        # Verify outcome: correct number of rows based on days_in_year
        assert result_365.shape[0] == 365, "Should have 365 rows for regular year"
        assert result_365.shape[1] == 24, "Should have 24 columns for hours"

    def test_compute_profile_preserves_metadata_from_input(self):
        """Test that compute_profile preserves important metadata from input DataArray."""
        # Execute function
        result = compute_profile(self.sample_data, days_in_year=365, q=0.75)

        # Verify outcome: metadata is preserved and enhanced
        assert "units" in result.attrs, "Should preserve units from input data"
        assert "quantile" in result.attrs, "Should include quantile information"
        assert "method" in result.attrs, "Should include method description"
        assert (
            result.attrs["quantile"] == 0.75
        ), "Should record the correct quantile used"
        assert result.attrs["units"] == "degC", "Should preserve original units"


class TestProfileUtilityFunctions:
    """Test class for profile utility functions.

    Tests the utility functions that handle metadata extraction, units,
    and profile DataFrame formatting operations.

    Attributes
    ----------
    sample_profile : pd.DataFrame
        Sample profile DataFrame for testing.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create sample profile DataFrame with metadata
        self.sample_profile = pd.DataFrame(
            np.random.rand(365, 24), index=range(1, 366), columns=range(1, 25)
        )
        # Add metadata attributes
        self.sample_profile.attrs = {
            "units": "degF",
            "variable_id": "tasmax",
            "quantile": 0.5,
            "method": "8760 analysis",
            "description": "Test climate profile",
        }

    def test_get_profile_units_returns_correct_units(self):
        """Test that get_profile_units extracts units from DataFrame metadata."""
        # Execute function
        units = get_profile_units(self.sample_profile)

        # Verify outcome: returns the correct units
        assert units == "degF", "Should return the units from DataFrame metadata"

    def test_get_profile_units_returns_unknown_when_missing(self):
        """Test that get_profile_units returns 'Unknown' when units are not present."""
        # Create DataFrame without units
        profile_no_units = pd.DataFrame(np.random.rand(10, 5))

        # Execute function
        units = get_profile_units(profile_no_units)

        # Verify outcome: returns 'Unknown' for missing units
        assert (
            units == "Unknown"
        ), "Should return 'Unknown' when units metadata is missing"

    def test_get_profile_metadata_returns_all_attributes(self):
        """Test that get_profile_metadata returns complete metadata dictionary."""
        # Execute function
        metadata = get_profile_metadata(self.sample_profile)

        # Verify outcome: returns dictionary with all metadata
        assert isinstance(metadata, dict), "Should return a dictionary"
        assert metadata["units"] == "degF", "Should include units"
        assert metadata["variable_id"] == "tasmax", "Should include variable_id"
        assert metadata["quantile"] == 0.5, "Should include quantile"
        assert metadata["method"] == "8760 analysis", "Should include method"
        assert len(metadata) == 5, "Should return all metadata attributes"

    def test_set_profile_metadata_updates_dataframe_attrs(self):
        """Test that set_profile_metadata properly updates DataFrame attributes."""
        # Setup new metadata to add
        new_metadata = {
            "author": "Test User",
            "created_date": "2023-01-01",
            "notes": "Test profile data",
        }

        # Execute function
        set_profile_metadata(self.sample_profile, new_metadata)

        # Verify outcome: DataFrame attrs are updated
        assert "author" in self.sample_profile.attrs, "Should add new author attribute"
        assert (
            "created_date" in self.sample_profile.attrs
        ), "Should add new created_date attribute"
        assert "notes" in self.sample_profile.attrs, "Should add new notes attribute"
        assert (
            self.sample_profile.attrs["author"] == "Test User"
        ), "Should set correct author value"

        # Original attributes should still exist
        assert (
            self.sample_profile.attrs["units"] == "degF"
        ), "Should preserve original units"

    def test_set_profile_metadata_raises_error_for_non_dict_input(self):
        """Test that set_profile_metadata raises ValueError for non-dictionary input."""
        # Execute and verify outcome: should raise ValueError for non-dict input
        with pytest.raises(
            ValueError, match="Metadata must be provided as a dictionary"
        ):
            set_profile_metadata(self.sample_profile, "not_a_dict")


class TestComputeDifferenceProfile:
    """Test class for _compute_difference_profile function.

    Tests the function that computes differences between future and historic
    climate profiles, handling various DataFrame column structures including
    simple indexes and MultiIndex columns.

    Attributes
    ----------
    simple_future_profile : pd.DataFrame
        Simple future profile with single-level columns.
    simple_historic_profile : pd.DataFrame
        Simple historic profile with single-level columns.
    multi_future_profile : pd.DataFrame
        Future profile with MultiIndex columns.
    multi_historic_profile : pd.DataFrame
        Historic profile with MultiIndex columns.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create simple profiles (single-level columns)
        self.simple_future_profile = pd.DataFrame(
            np.random.rand(365, 24) + 20.0,  # Future is warmer
            index=range(1, 366),
            columns=range(1, 25),
        )
        self.simple_historic_profile = pd.DataFrame(
            np.random.rand(365, 24) + 15.0,  # Historic is cooler
            index=range(1, 366),
            columns=range(1, 25),
        )

        # Create MultiIndex profiles
        hours = list(range(1, 25))
        wl_levels = [1.5, 2.0]
        simulations = ["sim1", "sim2"]

        # Future with MultiIndex (Hour, Warming_Level, Simulation)
        multi_cols_future = pd.MultiIndex.from_product(
            [hours, wl_levels, simulations],
            names=["Hour", "Warming_Level", "Simulation"],
        )
        self.multi_future_profile = pd.DataFrame(
            np.random.rand(365, len(multi_cols_future)) + 20.0,
            index=range(1, 366),
            columns=multi_cols_future,
        )

        # Historic with MultiIndex (Hour, Simulation)
        multi_cols_historic = pd.MultiIndex.from_product(
            [hours, simulations], names=["Hour", "Simulation"]
        )
        self.multi_historic_profile = pd.DataFrame(
            np.random.rand(365, len(multi_cols_historic)) + 15.0,
            index=range(1, 366),
            columns=multi_cols_historic,
        )

    def test_compute_difference_profile_with_simple_columns(self):
        """Test _compute_difference_profile with simple single-level columns."""
        # Execute function
        result = _compute_difference_profile(
            self.simple_future_profile, self.simple_historic_profile
        )

        # Verify outcome: returns DataFrame with difference values
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert (
            result.shape == self.simple_future_profile.shape
        ), "Shape should match future profile"

        # Check that differences are computed correctly (future - historic)
        # Since future values are ~20 and historic are ~15, differences should be ~5
        assert (
            result.mean().mean() > 0
        ), "Future should be warmer than historic on average"
        assert (
            result.mean().mean() < 10
        ), "Difference should be reasonable (< 10 degrees)"

    def test_compute_difference_profile_with_multiindex_columns(self):
        """Test _compute_difference_profile with MultiIndex columns."""
        # Create matched MultiIndex profiles for testing
        hours = list(range(1, 25))
        simulations = ["sim1", "sim2"]

        # Both profiles have (Hour, Simulation) structure
        multi_cols = pd.MultiIndex.from_product(
            [hours, simulations], names=["Hour", "Simulation"]
        )

        future_multi = pd.DataFrame(
            np.random.rand(365, len(multi_cols)) + 20.0,
            index=range(1, 366),
            columns=multi_cols,
        )
        historic_multi = pd.DataFrame(
            np.random.rand(365, len(multi_cols)) + 15.0,
            index=range(1, 366),
            columns=multi_cols,
        )

        # Execute function
        result = _compute_difference_profile(future_multi, historic_multi)

        # Verify outcome: returns DataFrame with MultiIndex structure preserved
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert isinstance(
            result.columns, pd.MultiIndex
        ), "Should preserve MultiIndex structure"
        assert result.columns.names == [
            "Hour",
            "Simulation",
        ], "Should preserve column level names"
        assert result.shape == future_multi.shape, "Shape should match future profile"

    def test_compute_difference_profile_with_mixed_index_types(self):
        """Test _compute_difference_profile when future has MultiIndex and historic has simple columns."""
        # Execute function with mixed index types
        result = _compute_difference_profile(
            self.multi_future_profile, self.simple_historic_profile
        )

        # Verify outcome: returns DataFrame preserving future structure
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert isinstance(
            result.columns, pd.MultiIndex
        ), "Should preserve future MultiIndex structure"
        assert (
            result.shape == self.multi_future_profile.shape
        ), "Shape should match future profile"
        assert (
            result.columns.names == self.multi_future_profile.columns.names
        ), "Should preserve column level names"

    def test_compute_difference_profile_handles_empty_dataframes(self):
        """Test _compute_difference_profile with empty DataFrames."""
        # Create empty DataFrames
        empty_future = pd.DataFrame()
        empty_historic = pd.DataFrame()

        # Execute function
        result = _compute_difference_profile(empty_future, empty_historic)

        # Verify outcome: returns empty DataFrame
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.empty, "Should return empty DataFrame for empty inputs"

    def test_compute_difference_profile_preserves_metadata(self):
        """Test that _compute_difference_profile preserves metadata from future profile."""
        # Add metadata to future profile
        self.simple_future_profile.attrs = {
            "units": "degC",
            "variable_id": "tasmax",
            "description": "Future temperature profile",
        }

        # Execute function
        result = _compute_difference_profile(
            self.simple_future_profile, self.simple_historic_profile
        )

        # Verify outcome: metadata is preserved
        assert hasattr(result, "attrs"), "Result should have attrs attribute"
        assert result.attrs["units"] == "degC", "Should preserve units metadata"
        assert (
            result.attrs["variable_id"] == "tasmax"
        ), "Should preserve variable_id metadata"


class TestComputeMultiindexDifference:
    """Test class for _compute_multiindex_difference function.

    Tests the function that computes differences when both future and historic
    profiles have MultiIndex columns, handling various combinations of warming
    levels and simulations.

    Attributes
    ----------
    future_profile : pd.DataFrame
        Future profile with MultiIndex columns.
    historic_profile : pd.DataFrame
        Historic profile with MultiIndex columns.
    """

    def setup_method(self):
        """Set up test fixtures."""
        hours = list(range(1, 25))
        simulations = ["sim1", "sim2"]
        warming_levels = [1.5, 2.0]

        # Create future profile with (Hour, Warming_Level, Simulation)
        future_cols = pd.MultiIndex.from_product(
            [hours, warming_levels, simulations],
            names=["Hour", "Warming_Level", "Simulation"],
        )
        self.future_profile = pd.DataFrame(
            np.random.rand(365, len(future_cols)) + 20.0,
            index=range(1, 366),
            columns=future_cols,
        )

        # Create historic profile with (Hour, Simulation)
        historic_cols = pd.MultiIndex.from_product(
            [hours, simulations], names=["Hour", "Simulation"]
        )
        self.historic_profile = pd.DataFrame(
            np.random.rand(365, len(historic_cols)) + 15.0,
            index=range(1, 366),
            columns=historic_cols,
        )

    def test_compute_multiindex_difference_with_simulation_levels(self):
        """Test _compute_multiindex_difference when both profiles have Simulation levels."""
        # Create profiles where both have Simulation levels
        hours = list(range(1, 25))
        simulations = ["sim1", "sim2"]

        # Both profiles have (Hour, Simulation) structure
        cols = pd.MultiIndex.from_product(
            [hours, simulations], names=["Hour", "Simulation"]
        )

        future_sim = pd.DataFrame(
            np.random.rand(365, len(cols)) + 20.0, index=range(1, 366), columns=cols
        )
        historic_sim = pd.DataFrame(
            np.random.rand(365, len(cols)) + 15.0, index=range(1, 366), columns=cols
        )

        # Execute function
        result = _compute_multiindex_difference(future_sim, historic_sim)

        # Verify outcome: returns DataFrame with proper structure
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert isinstance(
            result.columns, pd.MultiIndex
        ), "Should maintain MultiIndex structure"
        assert result.columns.names == [
            "Hour",
            "Simulation",
        ], "Should preserve column level names"
        assert result.shape == future_sim.shape, "Shape should match future profile"

    def test_compute_multiindex_difference_with_warming_level_only(self):
        """Test _compute_multiindex_difference when future has Warming_Level but no Simulation."""
        # Create future profile with (Hour, Warming_Level) but no Simulation
        hours = list(range(1, 25))
        warming_levels = [1.5, 2.0]

        future_cols = pd.MultiIndex.from_product(
            [hours, warming_levels], names=["Hour", "Warming_Level"]
        )
        future_wl = pd.DataFrame(
            np.random.rand(365, len(future_cols)) + 20.0,
            index=range(1, 366),
            columns=future_cols,
        )

        # Create simple historic profile
        historic_simple = pd.DataFrame(
            np.random.rand(365, len(hours)) + 15.0, index=range(1, 366), columns=hours
        )

        # Execute function
        result = _compute_multiindex_difference(future_wl, historic_simple)

        # Verify outcome: returns DataFrame preserving future structure
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert isinstance(
            result.columns, pd.MultiIndex
        ), "Should maintain future MultiIndex structure"
        assert result.columns.names == [
            "Hour",
            "Warming_Level",
        ], "Should preserve future column level names"
        assert result.shape == future_wl.shape, "Shape should match future profile"

    def test_compute_multiindex_difference_handles_duplicate_columns(self):
        """Test _compute_multiindex_difference handles DataFrames with duplicate columns."""
        # Create DataFrame with duplicate columns to test deduplication
        hours = [1, 1, 2, 2]  # Duplicate hours
        simulations = ["sim1", "sim1"]  # Duplicate simulations

        duplicate_cols = pd.MultiIndex.from_product(
            [hours, simulations], names=["Hour", "Simulation"]
        )

        future_dup = pd.DataFrame(
            np.random.rand(10, len(duplicate_cols)) + 20.0,
            index=range(1, 11),
            columns=duplicate_cols,
        )
        historic_dup = pd.DataFrame(
            np.random.rand(10, len(duplicate_cols)) + 15.0,
            index=range(1, 11),
            columns=duplicate_cols,
        )

        # Execute function - should handle duplicates gracefully
        result = _compute_multiindex_difference(future_dup, historic_dup)

        # Verify outcome: returns valid DataFrame (internal handling of duplicates)
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape[0] == future_dup.shape[0], "Should preserve number of rows"


class TestComputeSimulationPairedDifference:
    """Test class for _compute_simulation_paired_difference function.

    Tests the function that computes paired differences when both profiles
    have simulation dimensions, matching simulations between future and historic
    profiles for accurate comparison.

    Attributes
    ----------
    future_profile : pd.DataFrame
        Future profile with simulation columns.
    historic_profile : pd.DataFrame
        Historic profile with simulation columns.
    """

    def setup_method(self):
        """Set up test fixtures."""
        hours = list(range(1, 25))
        simulations = ["sim1", "sim2", "sim3"]

        # Create future profile with (Hour, Simulation)
        future_cols = pd.MultiIndex.from_product(
            [hours, simulations], names=["Hour", "Simulation"]
        )
        self.future_profile = pd.DataFrame(
            np.random.rand(365, len(future_cols)) + 20.0,
            index=range(1, 366),
            columns=future_cols,
        )

        # Create historic profile with (Hour, Simulation) - same simulations
        self.historic_profile = pd.DataFrame(
            np.random.rand(365, len(future_cols)) + 15.0,
            index=range(1, 366),
            columns=future_cols,
        )

    def test_compute_simulation_paired_difference_matches_common_simulations(self):
        """Test _compute_simulation_paired_difference matches common simulations correctly."""
        # Execute function with matching simulation sets
        future_levels = ["Hour", "Simulation"]
        historic_levels = ["Hour", "Simulation"]

        result = _compute_simulation_paired_difference(
            self.future_profile, self.historic_profile, future_levels, historic_levels
        )

        # Verify outcome: returns DataFrame with paired differences
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert isinstance(
            result.columns, pd.MultiIndex
        ), "Should maintain MultiIndex structure"
        assert result.columns.names == [
            "Hour",
            "Simulation",
        ], "Should preserve column level names"
        assert (
            result.shape == self.future_profile.shape
        ), "Shape should match future profile"

        # Check that differences are computed (future - historic)
        # Values should be positive since future (20+) > historic (15+)
        assert (
            result.mean().mean() > 0
        ), "Differences should be positive (future > historic)"

    def test_compute_simulation_paired_difference_with_no_common_simulations(self):
        """Test _compute_simulation_paired_difference when no simulations match."""
        # Create historic profile with different simulations
        hours = list(range(1, 25))
        different_sims = ["sim4", "sim5", "sim6"]  # No overlap with future sims

        historic_cols = pd.MultiIndex.from_product(
            [hours, different_sims], names=["Hour", "Simulation"]
        )
        historic_different = pd.DataFrame(
            np.random.rand(365, len(historic_cols)) + 15.0,
            index=range(1, 366),
            columns=historic_cols,
        )

        # Execute function with no matching simulations
        future_levels = ["Hour", "Simulation"]
        historic_levels = ["Hour", "Simulation"]

        result = _compute_simulation_paired_difference(
            self.future_profile, historic_different, future_levels, historic_levels
        )

        # Verify outcome: returns DataFrame when no matches (computes average difference)
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert (
            result.shape == self.future_profile.shape
        ), "Shape should match future profile"
        # When no common simulations, function computes difference using averages
        # Results should still be meaningful (positive since future > historic)
        assert (
            result.mean().mean() > 0
        ), "Should compute positive differences even without matched sims"

    def test_compute_simulation_paired_difference_with_duplicate_columns(self):
        """Test _compute_simulation_paired_difference handles duplicate columns correctly."""
        # Create future profile with duplicate columns
        hours = [1, 1, 2, 2]  # Duplicate hours
        sims = ["sim1", "sim1"]  # Duplicate simulations

        dup_cols = pd.MultiIndex.from_product(
            [hours, sims], names=["Hour", "Simulation"]
        )

        future_dup = pd.DataFrame(
            np.random.rand(10, len(dup_cols)) + 20.0,
            index=range(1, 11),
            columns=dup_cols,
        )
        historic_dup = pd.DataFrame(
            np.random.rand(10, len(dup_cols)) + 15.0,
            index=range(1, 11),
            columns=dup_cols,
        )

        # Execute function
        future_levels = ["Hour", "Simulation"]
        historic_levels = ["Hour", "Simulation"]

        result = _compute_simulation_paired_difference(
            future_dup, historic_dup, future_levels, historic_levels
        )

        # Verify outcome: handles duplicates gracefully
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape[0] == future_dup.shape[0], "Should preserve row count"


class TestComputeWarmingLevelDifference:
    """Test class for _compute_warming_level_difference function.

    Tests the function that computes differences for profiles with warming
    levels but no simulations, handling various column structures and
    matching logic between future and historic profiles.

    Attributes
    ----------
    future_profile : pd.DataFrame
        Future profile with warming level columns.
    historic_profile : pd.DataFrame
        Historic profile for comparison.
    """

    def setup_method(self):
        """Set up test fixtures."""
        hours = list(range(1, 25))
        warming_levels = [1.5, 2.0, 3.0]

        # Create future profile with (Hour, Warming_Level) MultiIndex
        future_cols = pd.MultiIndex.from_product(
            [hours, warming_levels], names=["Hour", "Warming_Level"]
        )
        self.future_profile = pd.DataFrame(
            np.random.rand(365, len(future_cols)) + 20.0,
            index=range(1, 366),
            columns=future_cols,
        )

        # Create simple historic profile with hour columns
        self.historic_profile = pd.DataFrame(
            np.random.rand(365, 24) + 15.0,
            index=range(1, 366),
            columns=hours,
        )

    def test_compute_warming_level_difference_returns_difference_dataframe(self):
        """Test _compute_warming_level_difference returns DataFrame with computed differences."""
        # Execute function
        future_levels = ["Hour", "Warming_Level"]
        historic_levels = []

        result = _compute_warming_level_difference(
            self.future_profile, self.historic_profile, future_levels, historic_levels
        )

        # Verify outcome: returns DataFrame with differences computed
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert (
            result.shape == self.future_profile.shape
        ), "Result shape should match future profile"
        assert isinstance(
            result.columns, pd.MultiIndex
        ), "Should preserve MultiIndex structure"
        assert result.columns.names == [
            "Hour",
            "Warming_Level",
        ], "Should preserve column level names"

        # Verify differences are computed (future - historic should be positive on average)
        assert (
            result.mean().mean() > 0
        ), "Future should be warmer than historic on average"


class TestFormatBasedOnStructure:
    """Test class for _format_based_on_structure function.

    Tests the function that formats DataFrames based on whether they have
    single-level or multi-level columns, ensuring proper formatting for
    climate profile display and processing.

    Attributes
    ----------
    simple_df : pd.DataFrame
        DataFrame with simple single-level columns.
    multi_df : pd.DataFrame
        DataFrame with MultiIndex columns.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create DataFrame with simple columns
        self.simple_df = pd.DataFrame(
            np.random.rand(365, 24), index=range(1, 366), columns=range(1, 25)
        )

        # Create DataFrame with MultiIndex columns
        hours = list(range(1, 25))
        simulations = ["sim1", "sim2"]
        multi_cols = pd.MultiIndex.from_product(
            [hours, simulations], names=["Hour", "Simulation"]
        )
        self.multi_df = pd.DataFrame(
            np.random.rand(365, len(multi_cols)),
            index=range(1, 366),
            columns=multi_cols,
        )

    def test_format_based_on_structure_with_simple_columns(self):
        """Test _format_based_on_structure with single-level columns."""
        # Store original state for comparison
        original_columns = self.simple_df.columns.copy()

        # Execute function
        _format_based_on_structure(self.simple_df)

        # Verify outcome: simple columns should be formatted appropriately
        assert not isinstance(
            self.simple_df.columns, pd.MultiIndex
        ), "Should maintain simple column structure"
        # The function modifies the DataFrame in-place, check it's still valid
        assert isinstance(
            self.simple_df, pd.DataFrame
        ), "Should remain a valid DataFrame"
        assert len(self.simple_df.columns) == len(
            original_columns
        ), "Should preserve column count"

    def test_format_based_on_structure_with_multiindex_columns(self):
        """Test _format_based_on_structure with MultiIndex columns."""
        # Store original MultiIndex info
        original_shape = self.multi_df.shape

        # Execute function
        _format_based_on_structure(self.multi_df)

        # Verify outcome: MultiIndex should be handled appropriately
        assert isinstance(
            self.multi_df, pd.DataFrame
        ), "Should remain a valid DataFrame"
        assert self.multi_df.shape == original_shape, "Should preserve DataFrame shape"
        # Function should preserve or appropriately modify MultiIndex structure
        if isinstance(self.multi_df.columns, pd.MultiIndex):
            assert (
                self.multi_df.columns.names is not None
            ), "Should have meaningful column level names"

    def test_format_based_on_structure_handles_empty_dataframe(self):
        """Test _format_based_on_structure with empty DataFrame."""
        # Create empty DataFrame
        empty_df = pd.DataFrame()

        # Execute function - expect it to raise an error for empty DataFrame
        with pytest.raises(ValueError, match="Length mismatch"):
            _format_based_on_structure(empty_df)


class TestConstructProfileDataframe:
    """Test class for _construct_profile_dataframe function.

    Tests the function that constructs climate profile DataFrames with appropriate
    column structures based on warming level and simulation dimensions.

    Attributes
    ----------
    profile_data : dict
        Sample profile data dictionary for testing.
    warming_levels : np.ndarray
        Array of warming level values.
    simulations : list
        List of simulation identifiers.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.warming_levels = np.array([1.5, 2.0])
        self.simulations = ["sim1", "sim2"]

        # Create sample profile data dictionary
        self.profile_data = {}
        for wl in self.warming_levels:
            for i, sim in enumerate(self.simulations):
                wl_key = f"WL_{wl}"
                sim_key = f"Sim{i+1}"  # Simple simulation labels
                # Create 365x24 profile matrix
                self.profile_data[(wl_key, sim_key)] = np.random.rand(365, 24) + 20.0

        # Simple function to get simulation labels
        def sim_label_func(sim, sim_idx):
            return f"Sim{sim_idx + 1}"

        self.sim_label_func = sim_label_func

    def test_construct_profile_dataframe_single_wl_single_sim(self):
        """Test _construct_profile_dataframe with single warming level and single simulation."""
        # Use only first warming level and simulation
        single_wl = np.array([1.5])
        single_sim = ["sim1"]

        # Create appropriate profile data
        profile_data = {("WL_1.5", "Sim1"): np.random.rand(365, 24) + 20.0}

        # Execute function
        result = _construct_profile_dataframe(
            profile_data=profile_data,
            warming_levels=single_wl,
            simulations=single_sim,
            sim_label_func=self.sim_label_func,
            days_in_year=365,
            hours_per_day=24,
        )

        # Verify outcome: simple DataFrame structure
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape == (365, 24), "Should have 365 rows and 24 columns"
        assert not isinstance(
            result.columns, pd.MultiIndex
        ), "Should have simple column structure"
        assert list(result.columns) == list(
            range(1, 25)
        ), "Columns should be hours 1-24"

    def test_construct_profile_dataframe_single_wl_multi_sim(self):
        """Test _construct_profile_dataframe with single warming level and multiple simulations."""
        # Use single warming level but multiple simulations
        single_wl = np.array([1.5])
        multi_sim = ["sim1", "sim2"]

        # Create appropriate profile data
        profile_data = {
            ("WL_1.5", "Sim1"): np.random.rand(365, 24) + 20.0,
            ("WL_1.5", "Sim2"): np.random.rand(365, 24) + 20.0,
        }

        # Execute function
        result = _construct_profile_dataframe(
            profile_data=profile_data,
            warming_levels=single_wl,
            simulations=multi_sim,
            sim_label_func=self.sim_label_func,
            days_in_year=365,
            hours_per_day=24,
        )

        # Verify outcome: MultiIndex structure with (Hour, Simulation)
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape == (
            365,
            48,
        ), "Should have 365 rows and 48 columns (24*2 sims)"
        assert isinstance(
            result.columns, pd.MultiIndex
        ), "Should have MultiIndex column structure"
        assert result.columns.names == [
            "Hour",
            "Simulation",
        ], "Should have Hour and Simulation levels"

    def test_construct_profile_dataframe_multi_wl_single_sim(self):
        """Test _construct_profile_dataframe with multiple warming levels and single simulation."""
        # Use multiple warming levels but single simulation
        multi_wl = np.array([1.5, 2.0])
        single_sim = ["sim1"]

        # Create appropriate profile data
        profile_data = {
            ("WL_1.5", "Sim1"): np.random.rand(365, 24) + 20.0,
            ("WL_2.0", "Sim1"): np.random.rand(365, 24) + 22.0,
        }

        # Execute function
        result = _construct_profile_dataframe(
            profile_data=profile_data,
            warming_levels=multi_wl,
            simulations=single_sim,
            sim_label_func=self.sim_label_func,
            days_in_year=365,
            hours_per_day=24,
        )

        # Verify outcome: MultiIndex structure with (Hour, Warming_Level)
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape == (
            365,
            48,
        ), "Should have 365 rows and 48 columns (24*2 WLs)"
        assert isinstance(
            result.columns, pd.MultiIndex
        ), "Should have MultiIndex column structure"
        assert result.columns.names == [
            "Hour",
            "Warming_Level",
        ], "Should have Hour and Warming_Level levels"

    def test_construct_profile_dataframe_multi_wl_multi_sim(self):
        """Test _construct_profile_dataframe with multiple warming levels and multiple simulations."""
        # Use multiple warming levels and multiple simulations
        multi_wl = np.array([1.5, 2.0])
        multi_sim = ["sim1", "sim2"]

        # Create appropriate profile data for all combinations
        profile_data = {
            ("WL_1.5", "Sim1"): np.random.rand(365, 24) + 20.0,
            ("WL_1.5", "Sim2"): np.random.rand(365, 24) + 20.0,
            ("WL_2.0", "Sim1"): np.random.rand(365, 24) + 22.0,
            ("WL_2.0", "Sim2"): np.random.rand(365, 24) + 22.0,
        }

        # Execute function
        result = _construct_profile_dataframe(
            profile_data=profile_data,
            warming_levels=multi_wl,
            simulations=multi_sim,
            sim_label_func=self.sim_label_func,
            days_in_year=365,
            hours_per_day=24,
        )

        # Verify outcome: MultiIndex structure with (Hour, Warming_Level, Simulation)
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape == (
            365,
            96,
        ), "Should have 365 rows and 96 columns (24*2WL*2sim)"
        assert isinstance(
            result.columns, pd.MultiIndex
        ), "Should have MultiIndex column structure"
        # The function creates a 3-level MultiIndex for this scenario
        assert len(result.columns.levels) >= 2, "Should have at least 2 column levels"


class TestStackProfileData:
    """Test class for _stack_profile_data function.

    Tests the function that stacks profile data arrays into the appropriate
    format for DataFrame creation, handling different ordering and dimensions.

    Attributes
    ----------
    profile_data : dict
        Sample profile data dictionary for testing.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create sample profile data - 365 days x 24 hours
        self.profile_data = {
            ("WL_1.5", "Sim1"): np.random.rand(365, 24) + 20.0,
            ("WL_1.5", "Sim2"): np.random.rand(365, 24) + 20.0,
            ("WL_2.0", "Sim1"): np.random.rand(365, 24) + 22.0,
            ("WL_2.0", "Sim2"): np.random.rand(365, 24) + 22.0,
        }

    def test_stack_profile_data_hour_first_two_level(self):
        """Test _stack_profile_data with hour_first=True and two levels."""
        # Execute function with hour-first ordering
        result = _stack_profile_data(
            profile_data=self.profile_data,
            hours_per_day=24,
            wl_names=["WL_1.5", "WL_2.0"],
            sim_names=["Sim1", "Sim2"],
            hour_first=True,
            three_level=False,
        )

        # Verify outcome: returns 2D array with correct shape
        assert isinstance(result, np.ndarray), "Should return a numpy array"
        assert result.shape == (
            365,
            96,
        ), "Should have 365 rows and 96 columns (24*2WL*2sim)"
        assert not np.isnan(result).any(), "Should not contain NaN values"

        # Check that all values are positive (since input data is 20+ and 22+)
        assert np.all(result > 0), "All values should be positive"

    def test_stack_profile_data_three_level_structure(self):
        """Test _stack_profile_data with three_level=True."""
        # Execute function with three-level structure
        result = _stack_profile_data(
            profile_data=self.profile_data,
            hours_per_day=24,
            wl_names=["WL_1.5", "WL_2.0"],
            sim_names=["Sim1", "Sim2"],
            hour_first=True,
            three_level=True,
        )

        # Verify outcome: returns 2D array with proper dimensions
        assert isinstance(result, np.ndarray), "Should return a numpy array"
        assert result.shape == (365, 96), "Should have correct total column count"
        assert result.dtype.kind == "f", "Should contain floating point data"

    def test_stack_profile_data_handles_single_simulation(self):
        """Test _stack_profile_data with single simulation."""
        # Create single simulation profile data
        single_sim_data = {
            ("WL_1.5", "Sim1"): np.random.rand(365, 24) + 20.0,
            ("WL_2.0", "Sim1"): np.random.rand(365, 24) + 22.0,
        }

        # Execute function
        result = _stack_profile_data(
            profile_data=single_sim_data,
            hours_per_day=24,
            wl_names=["WL_1.5", "WL_2.0"],
            sim_names=["Sim1"],
            hour_first=True,
            three_level=False,
        )

        # Verify outcome: correct dimensions for single simulation
        assert isinstance(result, np.ndarray), "Should return a numpy array"
        assert result.shape == (365, 48), "Should have 48 columns (24*2WL*1sim)"
        assert np.all(result >= 20), "Values should be in expected range"

    def test_stack_profile_data_preserves_data_values(self):
        """Test _stack_profile_data preserves original data values correctly."""
        # Create simple test data to verify preservation
        simple_data = {
            ("WL_1.5", "Sim1"): np.ones((365, 24)) * 25.0,  # All values = 25
            ("WL_2.0", "Sim1"): np.ones((365, 24)) * 30.0,  # All values = 30
        }

        # Execute function
        result = _stack_profile_data(
            profile_data=simple_data,
            hours_per_day=24,
            wl_names=["WL_1.5", "WL_2.0"],
            sim_names=["Sim1"],
            hour_first=True,
            three_level=False,
        )

        # Verify outcome: data values are preserved
        assert isinstance(result, np.ndarray), "Should return a numpy array"
        # Check that we have both 25.0 and 30.0 values in the result
        unique_values = np.unique(result)
        assert 25.0 in unique_values, "Should preserve 25.0 values from WL_1.5"
        assert 30.0 in unique_values, "Should preserve 30.0 values from WL_2.0"
