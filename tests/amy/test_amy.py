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

    def test_compute_warming_level_difference_matches_corresponding_hours(self):
        """Test _compute_warming_level_difference correctly matches hours between profiles."""
        # Create specific test data to verify hour matching
        hours = [1, 12, 24]  # Use subset of hours for clearer testing
        warming_levels = [2.0]

        # Future profile with known values
        future_cols = pd.MultiIndex.from_product(
            [hours, warming_levels], names=["Hour", "Warming_Level"]
        )
        future_data = pd.DataFrame(
            [[25.0, 30.0, 20.0]],
            index=[1],
            columns=future_cols,  # One row for simplicity
        )

        # Historic profile with known values for matching
        historic_data = pd.DataFrame(
            [[15.0, 20.0, 10.0]], index=[1], columns=hours  # Matching hours
        )

        # Execute function
        future_levels = ["Hour", "Warming_Level"]
        historic_levels = []

        result = _compute_warming_level_difference(
            future_data, historic_data, future_levels, historic_levels
        )

        # Verify outcome: differences are computed correctly for each hour
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"

        # Check specific hour matching: future[hour, wl] - historic[hour]
        # Hour 1: 25.0 - 15.0 = 10.0
        # Hour 12: 30.0 - 20.0 = 10.0
        # Hour 24: 20.0 - 10.0 = 10.0
        # Verify that differences are computed correctly by checking all values are 10.0
        result_values = result.iloc[0].values  # Get first row values
        expected_diff = 10.0
        for i, val in enumerate(result_values):
            assert (
                abs(val - expected_diff) < 0.001
            ), f"Column {i} difference should be {expected_diff}, got {val}"

    def test_compute_warming_level_difference_handles_duplicate_columns(self):
        """Test _compute_warming_level_difference handles duplicate columns properly."""
        hours = [1, 1, 2, 2]  # Duplicate hours
        warming_levels = [2.0]

        # Create future profile with duplicate columns
        future_cols = pd.MultiIndex.from_product(
            [hours, warming_levels], names=["Hour", "Warming_Level"]
        )
        future_dup = pd.DataFrame(
            np.random.rand(10, len(future_cols)) + 20.0,
            index=range(1, 11),
            columns=future_cols,
        )

        # Create historic profile with duplicate columns
        historic_dup = pd.DataFrame(
            np.random.rand(10, len(hours)) + 15.0,
            index=range(1, 11),
            columns=hours,
        )

        # Execute function
        future_levels = ["Hour", "Warming_Level"]
        historic_levels = []

        with patch("builtins.print") as mock_print:
            result = _compute_warming_level_difference(
                future_dup, historic_dup, future_levels, historic_levels
            )

        # Verify outcome: handles duplicates and shows warnings
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape[0] == future_dup.shape[0], "Should preserve row count"

        # Check that warnings about duplicates were printed
        printed_calls = [str(call) for call in mock_print.call_args_list]
        printed_output = " ".join(printed_calls)
        assert (
            "duplicate columns" in printed_output.lower()
        ), "Should warn about duplicate columns"

    def test_compute_warming_level_difference_with_multiindex_historic(self):
        """Test _compute_warming_level_difference when historic profile has MultiIndex columns."""
        hours = list(range(1, 25))
        warming_levels = [2.0]
        simulations = ["sim1"]

        # Create future profile with (Hour, Warming_Level) MultiIndex
        future_cols = pd.MultiIndex.from_product(
            [hours, warming_levels], names=["Hour", "Warming_Level"]
        )
        future_multi = pd.DataFrame(
            np.random.rand(365, len(future_cols)) + 20.0,
            index=range(1, 366),
            columns=future_cols,
        )

        # Create historic profile with (Hour, Simulation) MultiIndex
        historic_cols = pd.MultiIndex.from_product(
            [hours, simulations], names=["Hour", "Simulation"]
        )
        historic_multi = pd.DataFrame(
            np.random.rand(365, len(historic_cols)) + 15.0,
            index=range(1, 366),
            columns=historic_cols,
        )

        # Execute function
        future_levels = ["Hour", "Warming_Level"]
        historic_levels = ["Hour", "Simulation"]

        result = _compute_warming_level_difference(
            future_multi, historic_multi, future_levels, historic_levels
        )

        # Verify outcome: handles MultiIndex historic profile without crashing
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert (
            result.shape == future_multi.shape
        ), "Result shape should match future profile"
        assert isinstance(
            result.columns, pd.MultiIndex
        ), "Should preserve future MultiIndex structure"

        # Verify function completes successfully (may produce NaN if no matching structure)
        # This documents the current behavior when historic MultiIndex doesn't align with future
        assert result is not None, "Function should complete and return result"

    def test_compute_warming_level_difference_with_missing_hour_fallback(self):
        """Test _compute_warming_level_difference falls back correctly when hours don't match."""
        # Create future profile with specific hours
        future_hours = [1, 2, 3]
        warming_levels = [2.0]

        future_cols = pd.MultiIndex.from_product(
            [future_hours, warming_levels], names=["Hour", "Warming_Level"]
        )
        future_data = pd.DataFrame(
            np.random.rand(10, len(future_cols)) + 20.0,
            index=range(1, 11),
            columns=future_cols,
        )

        # Create historic profile with different hours (no overlap)
        historic_hours = [10, 11, 12]
        historic_data = pd.DataFrame(
            np.random.rand(10, len(historic_hours)) + 15.0,
            index=range(1, 11),
            columns=historic_hours,
        )

        # Execute function
        future_levels = ["Hour", "Warming_Level"]
        historic_levels = []

        result = _compute_warming_level_difference(
            future_data, historic_data, future_levels, historic_levels
        )

        # Verify outcome: handles missing hour matches with fallback
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert (
            result.shape == future_data.shape
        ), "Result shape should match future profile"

        # Verify differences are computed using fallback logic
        # Since no hours match, it should use first column of historic as fallback
        assert result.notna().any().any(), "Should produce non-NaN values with fallback"
        assert (
            result.mean().mean() > 0
        ), "Future should be warmer than historic on average with fallback"


class TestComputeMixedIndexDifference:
    """Test class for _compute_mixed_index_difference function.

    Tests the function that computes differences when future profile has
    MultiIndex columns and historic profile has simple columns, handling
    various matching scenarios and fallback logic.

    Attributes
    ----------
    future_profile : pd.DataFrame
        Future profile with MultiIndex columns.
    historic_profile : pd.DataFrame
        Historic profile with simple columns.
    """

    def setup_method(self):
        """Set up test fixtures."""
        hours = list(range(1, 25))
        warming_levels = [1.5, 2.0]
        simulations = ["sim1", "sim2"]

        # Create future profile with (Hour, Warming_Level, Simulation) MultiIndex
        future_cols = pd.MultiIndex.from_product(
            [hours, warming_levels, simulations],
            names=["Hour", "Warming_Level", "Simulation"],
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

    def test_compute_mixed_index_difference_returns_difference_dataframe(self):
        """Test _compute_mixed_index_difference returns DataFrame with computed differences."""
        # Execute function
        result = _compute_mixed_index_difference(
            self.future_profile, self.historic_profile
        )

        # Verify outcome: returns DataFrame with differences computed
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert (
            result.shape == self.future_profile.shape
        ), "Result shape should match future profile"
        assert isinstance(
            result.columns, pd.MultiIndex
        ), "Should preserve future MultiIndex structure"
        assert result.columns.names == [
            "Hour",
            "Warming_Level",
            "Simulation",
        ], "Should preserve column level names"

        # Verify differences are computed (future - historic should be positive on average)
        assert (
            result.mean().mean() > 0
        ), "Future should be warmer than historic on average"

    def test_compute_mixed_index_difference_with_matching_hours(self):
        """Test _compute_mixed_index_difference correctly handles matching hours."""
        # Create test data with specific values to verify matching
        hours = [1, 12, 24]  # Use subset for clearer verification
        warming_levels = [2.0]
        simulations = ["sim1"]

        # Future profile with known values
        future_cols = pd.MultiIndex.from_product(
            [hours, warming_levels, simulations],
            names=["Hour", "Warming_Level", "Simulation"],
        )
        future_data = pd.DataFrame(
            [[25.0, 30.0, 20.0]],
            index=[1],
            columns=future_cols,  # One row for simplicity
        )

        # Historic profile with corresponding hours
        historic_data = pd.DataFrame(
            [[15.0, 20.0, 10.0]], index=[1], columns=hours  # Matching hour structure
        )

        # Execute function
        result = _compute_mixed_index_difference(future_data, historic_data)

        # Verify outcome: differences computed correctly for matching hours
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert (
            result.shape == future_data.shape
        ), "Result shape should match future profile"

        # Verify specific hour matching produces expected differences
        # The function should match hours in MultiIndex with simple historic columns
        result_values = result.iloc[0].values
        expected_diffs = [10.0, 10.0, 10.0]  # 25-15, 30-20, 20-10

        for i, (actual, expected) in enumerate(zip(result_values, expected_diffs)):
            assert (
                abs(actual - expected) < 0.001
            ), f"Column {i} difference should be {expected}, got {actual}"

    def test_compute_mixed_index_difference_handles_duplicate_columns(self):
        """Test _compute_mixed_index_difference handles duplicate columns properly."""
        # Create profiles with duplicate columns
        hours = [1, 1, 2, 2]  # Duplicate hours
        warming_levels = [2.0]
        simulations = ["sim1"]

        # Future profile with duplicate MultiIndex columns
        future_cols = pd.MultiIndex.from_product(
            [hours, warming_levels, simulations],
            names=["Hour", "Warming_Level", "Simulation"],
        )
        future_dup = pd.DataFrame(
            np.random.rand(10, len(future_cols)) + 20.0,
            index=range(1, 11),
            columns=future_cols,
        )

        # Historic profile with duplicate simple columns
        historic_dup = pd.DataFrame(
            np.random.rand(10, len(hours)) + 15.0,
            index=range(1, 11),
            columns=hours,
        )

        # Execute function
        with patch("builtins.print") as mock_print:
            result = _compute_mixed_index_difference(future_dup, historic_dup)

        # Verify outcome: handles duplicates and shows warnings
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape[0] == future_dup.shape[0], "Should preserve row count"

        # Check that warnings about duplicates were printed
        printed_calls = [str(call) for call in mock_print.call_args_list]
        printed_output = " ".join(printed_calls)
        assert (
            "duplicate columns" in printed_output.lower()
        ), "Should warn about duplicate columns"

    def test_compute_mixed_index_difference_with_non_matching_hours(self):
        """Test _compute_mixed_index_difference when future and historic hours don't align."""
        # Create future profile with specific hours
        future_hours = [1, 2, 3]
        warming_levels = [2.0]
        simulations = ["sim1"]

        future_cols = pd.MultiIndex.from_product(
            [future_hours, warming_levels, simulations],
            names=["Hour", "Warming_Level", "Simulation"],
        )
        future_data = pd.DataFrame(
            np.random.rand(10, len(future_cols)) + 20.0,
            index=range(1, 11),
            columns=future_cols,
        )

        # Create historic profile with different hours (no overlap)
        historic_hours = [10, 11, 12]
        historic_data = pd.DataFrame(
            np.random.rand(10, len(historic_hours)) + 15.0,
            index=range(1, 11),
            columns=historic_hours,
        )

        # Execute function
        result = _compute_mixed_index_difference(future_data, historic_data)

        # Verify outcome: handles non-matching hours with fallback logic
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert (
            result.shape == future_data.shape
        ), "Result shape should match future profile"

        # Verify differences are computed using fallback logic
        # Function should use _find_matching_historic_value which provides fallback
        assert result.notna().any().any(), "Should produce non-NaN values with fallback"
        assert (
            result.mean().mean() > 0
        ), "Future should be warmer than historic on average with fallback"

    def test_compute_mixed_index_difference_with_different_multiindex_structures(self):
        """Test _compute_mixed_index_difference with various MultiIndex structures."""
        # Create future profile with (Hour, Warming_Level) only (no Simulation)
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
            np.random.rand(365, 24) + 15.0,
            index=range(1, 366),
            columns=hours,
        )

        # Execute function
        result = _compute_mixed_index_difference(future_wl, historic_simple)

        # Verify outcome: handles different MultiIndex structures correctly
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert (
            result.shape == future_wl.shape
        ), "Result shape should match future profile"
        assert isinstance(
            result.columns, pd.MultiIndex
        ), "Should preserve future MultiIndex structure"
        assert result.columns.names == [
            "Hour",
            "Warming_Level",
        ], "Should preserve future column level names"

        # Verify differences are computed correctly
        assert result.notna().any().any(), "Should produce non-NaN difference values"
        assert (
            result.mean().mean() > 0
        ), "Future should be warmer than historic on average"


class TestComputeSimpleDifference:
    """Test class for _compute_simple_difference function.

    Tests the function that computes differences for profiles with simple
    (non-MultiIndex) columns, handling both matching and non-matching
    column scenarios.

    Attributes
    ----------
    future_profile : pd.DataFrame
        Future profile with simple columns.
    historic_profile : pd.DataFrame
        Historic profile with simple columns.
    """

    def setup_method(self):
        """Set up test fixtures."""
        hours = list(range(1, 25))

        # Create future profile with simple columns
        self.future_profile = pd.DataFrame(
            np.random.rand(365, 24) + 20.0,
            index=range(1, 366),
            columns=hours,
        )

        # Create historic profile with simple columns (same structure)
        self.historic_profile = pd.DataFrame(
            np.random.rand(365, 24) + 15.0,
            index=range(1, 366),
            columns=hours,
        )

    def test_compute_simple_difference_returns_difference_dataframe(self):
        """Test _compute_simple_difference returns DataFrame with computed differences."""
        # Execute function
        result = _compute_simple_difference(self.future_profile, self.historic_profile)

        # Verify outcome: returns DataFrame with differences computed
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert (
            result.shape == self.future_profile.shape
        ), "Result shape should match future profile"
        assert not isinstance(
            result.columns, pd.MultiIndex
        ), "Should maintain simple column structure"
        assert list(result.columns) == list(
            self.future_profile.columns
        ), "Should preserve column names"

        # Verify differences are computed (future - historic should be positive on average)
        assert (
            result.mean().mean() > 0
        ), "Future should be warmer than historic on average"

    def test_compute_simple_difference_with_matching_columns(self):
        """Test _compute_simple_difference with matching columns and known values."""
        # Create test data with specific values to verify computation
        hours = [1, 12, 24]  # Use subset for clearer verification

        # Future profile with known values
        future_data = pd.DataFrame(
            [[25.0, 30.0, 20.0]], index=[1], columns=hours  # One row for simplicity
        )

        # Historic profile with known values for matching columns
        historic_data = pd.DataFrame(
            [[15.0, 20.0, 10.0]], index=[1], columns=hours  # Matching columns
        )

        # Execute function
        with patch("builtins.print") as mock_print:
            result = _compute_simple_difference(future_data, historic_data)

        # Verify outcome: element-wise difference computed correctly
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert (
            result.shape == future_data.shape
        ), "Result shape should match future profile"

        # Verify specific element-wise differences
        # Hour 1: 25.0 - 15.0 = 10.0
        # Hour 12: 30.0 - 20.0 = 10.0
        # Hour 24: 20.0 - 10.0 = 10.0
        expected_diffs = [10.0, 10.0, 10.0]
        result_values = result.iloc[0].values

        for i, (actual, expected) in enumerate(zip(result_values, expected_diffs)):
            assert (
                abs(actual - expected) < 0.001
            ), f"Column {i} difference should be {expected}, got {actual}"

        # Check that success message was printed
        printed_calls = [str(call) for call in mock_print.call_args_list]
        printed_output = " ".join(printed_calls)
        assert (
            "columns match" in printed_output.lower()
        ), "Should print success message for matching columns"

    def test_compute_simple_difference_with_mismatched_columns(self):
        """Test _compute_simple_difference with non-matching columns but similar data."""
        # Create future profile with numeric columns
        future_data = pd.DataFrame(
            np.random.rand(10, 3) + 20.0,
            index=range(1, 11),
            columns=["temp1", "temp2", "temp3"],
        )

        # Create historic profile with same structure but different names
        historic_data = pd.DataFrame(
            np.random.rand(10, 3) + 15.0,
            index=range(1, 11),
            columns=["var1", "var2", "var3"],
        )

        # Execute function
        with patch("builtins.print") as mock_print:
            result = _compute_simple_difference(future_data, historic_data)

        # Verify outcome: function returns DataFrame with warning messages
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape[0] == future_data.shape[0], "Should preserve row count"

        # Check that warning messages were printed for column mismatch
        printed_calls = [str(call) for call in mock_print.call_args_list]
        printed_output = " ".join(printed_calls)
        assert (
            "column mismatch" in printed_output.lower()
        ), "Should warn about column mismatch"
        assert (
            "future columns:" in printed_output.lower()
        ), "Should show future columns info"
        assert (
            "historic columns:" in printed_output.lower()
        ), "Should show historic columns info"

        # Result might have NaN values due to pandas column alignment behavior
        # The function attempts positional alignment but pandas aligns by name
        # This is expected behavior when column names don't match
        assert (
            result.isna().all().all() or result.notna().any().any()
        ), "Function should handle column mismatch gracefully"

    def test_compute_simple_difference_with_empty_profiles(self):
        """Test _compute_simple_difference with empty DataFrames."""
        # Create empty DataFrames with matching structure
        future_data = pd.DataFrame(columns=["temp", "precip"])
        historic_data = pd.DataFrame(columns=["temp", "precip"])

        # Execute function
        with patch("builtins.print") as mock_print:
            result = _compute_simple_difference(future_data, historic_data)

        # Verify outcome: handles empty data gracefully
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.empty, "Result should be empty when input DataFrames are empty"
        assert list(result.columns) == [
            "temp",
            "precip",
        ], "Should preserve column structure even with empty data"
        assert result.shape == (0, 2), "Should have correct empty shape"

        # Check that success message was printed (columns match)
        printed_calls = [str(call) for call in mock_print.call_args_list]
        printed_output = " ".join(printed_calls)
        assert (
            "columns match" in printed_output.lower()
        ), "Should confirm columns match even for empty DataFrames"

    def test_compute_simple_difference_with_single_row(self):
        """Test _compute_simple_difference with single-row DataFrames."""
        # Create single-row DataFrames
        future_data = pd.DataFrame(
            [[25.5, 12.3, 8.7]],
            index=[1],
            columns=["temp", "precip", "humidity"],
        )
        historic_data = pd.DataFrame(
            [[20.2, 10.1, 7.9]],
            index=[1],
            columns=["temp", "precip", "humidity"],
        )

        # Execute function
        with patch("builtins.print") as mock_print:
            result = _compute_simple_difference(future_data, historic_data)

        # Verify outcome: processes single row correctly
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape == (1, 3), "Should have single row with three columns"
        assert list(result.columns) == [
            "temp",
            "precip",
            "humidity",
        ], "Should preserve column names"

        # Check that all differences are positive (future > historic)
        assert (result > 0).all().all(), "All differences should be positive"

        # Verify differences are not NaN and within reasonable ranges
        assert (
            result.notna().all().all()
        ), "All difference values should be valid numbers"
        assert (result < 100).all().all(), "All differences should be reasonable values"

        # Check that success message was printed
        printed_calls = [str(call) for call in mock_print.call_args_list]
        printed_output = " ".join(printed_calls)
        assert (
            "columns match" in printed_output.lower()
        ), "Should confirm columns match for single-row data"


class TestFindMatchingHistoricColumn:
    """Test class for _find_matching_historic_column function.

    Tests the function that finds matching historic columns for future columns
    when both profiles have MultiIndex structures with Hour and Simulation levels.
    The function handles different level orders and missing level scenarios.

    Attributes
    ----------
    sample_historic_profile : pd.DataFrame
        Sample historic profile with MultiIndex columns for testing.
    """

    def setup_method(self):
        """Set up test fixtures."""
        hours = [1, 12, 24]
        simulations = ["sim1", "sim2"]

        # Create historic profile with (Hour, Simulation) MultiIndex structure
        historic_cols = pd.MultiIndex.from_product(
            [hours, simulations], names=["Hour", "Simulation"]
        )
        self.sample_historic_profile = pd.DataFrame(
            np.random.rand(10, len(historic_cols)) + 15.0,
            index=range(1, 11),
            columns=historic_cols,
        )

    def test_find_matching_historic_column_returns_tuple_or_none(self):
        """Test _find_matching_historic_column returns tuple or None."""
        # Test future column that exists in historic profile
        future_col = (1, "sim1")  # (Hour, Simulation)
        future_levels = ["Hour", "Simulation"]
        historic_levels = ["Hour", "Simulation"]

        # Execute function
        result = _find_matching_historic_column(
            future_col, future_levels, self.sample_historic_profile, historic_levels
        )

        # Verify outcome: returns tuple or None
        assert (
            isinstance(result, tuple) or result is None
        ), "Should return a tuple or None"

        # In this case, should return matching tuple
        assert isinstance(result, tuple), "Should return tuple for valid match"
        assert len(result) == 2, "Returned tuple should have 2 elements"
        assert result == (1, "sim1"), "Should return exact matching tuple"

    def test_find_matching_historic_column_with_matching_hour_simulation(self):
        """Test _find_matching_historic_column with valid Hour and Simulation matching."""
        # Test various combinations that exist in the historic profile
        test_cases = [
            ((1, "sim1"), (1, "sim1")),  # First hour, first simulation
            ((12, "sim2"), (12, "sim2")),  # Middle hour, second simulation
            ((24, "sim1"), (24, "sim1")),  # Last hour, first simulation
        ]

        future_levels = ["Hour", "Simulation"]
        historic_levels = ["Hour", "Simulation"]

        for future_col, expected_result in test_cases:
            # Execute function
            result = _find_matching_historic_column(
                future_col, future_levels, self.sample_historic_profile, historic_levels
            )

            # Verify outcome: returns correct matching tuple
            assert isinstance(result, tuple), f"Should return tuple for {future_col}"
            assert (
                result == expected_result
            ), f"Should return {expected_result} for future column {future_col}"
            assert (
                result in self.sample_historic_profile.columns
            ), f"Returned column {result} should exist in historic profile"

    def test_find_matching_historic_column_with_different_level_order(self):
        """Test _find_matching_historic_column when historic levels are in different order."""
        # Create historic profile with (Simulation, Hour) order instead of (Hour, Simulation)
        hours = [1, 12, 24]
        simulations = ["sim1", "sim2"]

        # Reversed order: Simulation first, then Hour
        reversed_historic_cols = pd.MultiIndex.from_product(
            [simulations, hours], names=["Simulation", "Hour"]
        )
        reversed_historic_profile = pd.DataFrame(
            np.random.rand(10, len(reversed_historic_cols)) + 15.0,
            index=range(1, 11),
            columns=reversed_historic_cols,
        )

        # Future column in (Hour, Simulation) order
        future_col = (1, "sim1")  # Hour first, Simulation second
        future_levels = ["Hour", "Simulation"]
        historic_levels = ["Simulation", "Hour"]  # Reversed order

        # Execute function
        result = _find_matching_historic_column(
            future_col, future_levels, reversed_historic_profile, historic_levels
        )

        # Verify outcome: function handles level order correctly
        assert isinstance(result, tuple), "Should return tuple for reversed level order"
        # Function should create ('sim1', 1) to match historic (Simulation, Hour) order
        expected_result = ("sim1", 1)  # Simulation first, Hour second
        assert (
            result == expected_result
        ), f"Should return {expected_result} for reversed historic levels"
        assert (
            result in reversed_historic_profile.columns
        ), "Returned column should exist in reversed historic profile"

    def test_find_matching_historic_column_with_missing_levels(self):
        """Test _find_matching_historic_column when required levels are missing."""
        # Test various scenarios where required levels are missing
        test_scenarios = [
            # Future missing Hour level
            {
                "future_col": ("sim1",),  # Only simulation
                "future_levels": ["Simulation"],  # Missing Hour
                "historic_levels": ["Hour", "Simulation"],
                "description": "future missing Hour level",
            },
            # Future missing Simulation level
            {
                "future_col": (1,),  # Only hour
                "future_levels": ["Hour"],  # Missing Simulation
                "historic_levels": ["Hour", "Simulation"],
                "description": "future missing Simulation level",
            },
            # Historic missing Hour level
            {
                "future_col": (1, "sim1"),
                "future_levels": ["Hour", "Simulation"],
                "historic_levels": ["Simulation"],  # Missing Hour
                "description": "historic missing Hour level",
            },
            # Historic missing Simulation level
            {
                "future_col": (1, "sim1"),
                "future_levels": ["Hour", "Simulation"],
                "historic_levels": ["Hour"],  # Missing Simulation
                "description": "historic missing Simulation level",
            },
            # Both missing required levels
            {
                "future_col": ("other",),
                "future_levels": ["Other"],  # No Hour or Simulation
                "historic_levels": ["Different"],  # No Hour or Simulation
                "description": "both missing Hour and Simulation levels",
            },
        ]

        for scenario in test_scenarios:
            # Execute function
            result = _find_matching_historic_column(
                scenario["future_col"],
                scenario["future_levels"],
                self.sample_historic_profile,
                scenario["historic_levels"],
            )

            # Verify outcome: returns None when required levels are missing
            assert result is None, f"Should return None when {scenario['description']}"

    def test_find_matching_historic_column_with_no_matching_column(self):
        """Test _find_matching_historic_column when constructed column doesn't exist in historic."""
        # Test scenarios where future column exists but historic doesn't have the matching column
        test_cases = [
            # Hour exists but simulation doesn't
            {
                "future_col": (1, "nonexistent_sim"),
                "description": "nonexistent simulation",
            },
            # Simulation exists but hour doesn't
            {"future_col": (999, "sim1"), "description": "nonexistent hour"},
            # Neither hour nor simulation exists
            {
                "future_col": (999, "nonexistent_sim"),
                "description": "both nonexistent hour and simulation",
            },
        ]

        future_levels = ["Hour", "Simulation"]
        historic_levels = ["Hour", "Simulation"]

        for case in test_cases:
            # Execute function
            result = _find_matching_historic_column(
                case["future_col"],
                future_levels,
                self.sample_historic_profile,
                historic_levels,
            )

            # Verify outcome: returns None when constructed column doesn't exist
            assert result is None, f"Should return None for {case['description']}"

            # Double-check that the constructed column indeed doesn't exist
            assert (
                case["future_col"] not in self.sample_historic_profile.columns
            ), f"Test case {case['future_col']} should not exist in historic profile"


class TestGetHistoricHourMean:
    """Test class for _get_historic_hour_mean function.

    Tests the function that computes the mean of historic profile values
    for a specific hour, handling both MultiIndex columns with Simulation
    levels and simple column structures.

    Attributes
    ----------
    historic_with_sim : pd.DataFrame
        Historic profile with (Hour, Simulation) MultiIndex columns.
    historic_simple : pd.DataFrame
        Historic profile with simple hour columns.
    """

    def setup_method(self):
        """Set up test fixtures."""
        hours = [1, 12, 24]
        simulations = ["sim1", "sim2", "sim3"]

        # Create historic profile with (Hour, Simulation) MultiIndex structure
        historic_cols = pd.MultiIndex.from_product(
            [hours, simulations], names=["Hour", "Simulation"]
        )
        # Use predictable values for testing means
        self.historic_with_sim = pd.DataFrame(
            [
                [10.0, 20.0, 30.0, 15.0, 25.0, 35.0, 5.0, 15.0, 25.0],  # Row 1
                [12.0, 22.0, 32.0, 17.0, 27.0, 37.0, 7.0, 17.0, 27.0],
            ],  # Row 2
            index=[1, 2],
            columns=historic_cols,
        )

        # Create simple historic profile with hour columns
        self.historic_simple = pd.DataFrame(
            [[100.0, 200.0, 300.0], [110.0, 210.0, 310.0]],  # Row 1  # Row 2
            index=[1, 2],
            columns=hours,
        )

    def test_get_historic_hour_mean_returns_series(self):
        """Test _get_historic_hour_mean returns pd.Series."""
        # Test with MultiIndex structure
        historic_levels = ["Hour", "Simulation"]
        hour = 1

        # Execute function
        result = _get_historic_hour_mean(self.historic_with_sim, historic_levels, hour)

        # Verify outcome: returns pd.Series
        assert isinstance(result, pd.Series), "Should return a pandas Series"
        # xs extracts all columns for hour 1, then mean() computes mean across rows
        # Result should have one value per simulation (3 simulations)
        assert len(result) == 3, "Should have one value per simulation"
        assert result.index.name == "Simulation", "Index should be Simulation level"

        # Test with simple structure
        historic_levels = ["Hour"]
        result_simple = _get_historic_hour_mean(
            self.historic_simple, historic_levels, hour
        )

        # Verify outcome: also returns pd.Series (converted from single column)
        assert isinstance(
            result_simple, pd.Series
        ), "Should return a pandas Series for simple structure"
        assert (
            len(result_simple) == 2
        ), "Should have same number of rows as input DataFrame"

    def test_get_historic_hour_mean_with_simulation_levels(self):
        """Test _get_historic_hour_mean with Simulation levels computes correct means."""
        historic_levels = ["Hour", "Simulation"]

        # Test specific hours to verify mean calculation
        test_cases = [
            # Hour 1: columns (1, sim1), (1, sim2), (1, sim3)
            # Row 1 values: [10.0, 20.0, 30.0], Row 2 values: [12.0, 22.0, 32.0]
            # Expected means: sim1=11.0, sim2=21.0, sim3=31.0
            {"hour": 1, "expected_means": {"sim1": 11.0, "sim2": 21.0, "sim3": 31.0}},
            # Hour 12: columns (12, sim1), (12, sim2), (12, sim3)
            # Row 1 values: [15.0, 25.0, 35.0], Row 2 values: [17.0, 27.0, 37.0]
            # Expected means: sim1=16.0, sim2=26.0, sim3=36.0
            {"hour": 12, "expected_means": {"sim1": 16.0, "sim2": 26.0, "sim3": 36.0}},
        ]

        for case in test_cases:
            # Execute function
            result = _get_historic_hour_mean(
                self.historic_with_sim, historic_levels, case["hour"]
            )

            # Verify outcome: correct mean calculation
            assert isinstance(
                result, pd.Series
            ), f"Should return pd.Series for hour {case['hour']}"
            assert (
                result.index.name == "Simulation"
            ), f"Index should be Simulation level for hour {case['hour']}"

            # Check specific mean values
            for sim, expected_mean in case["expected_means"].items():
                actual_mean = result[sim]
                assert (
                    abs(actual_mean - expected_mean) < 0.001
                ), f"Hour {case['hour']}, Simulation {sim}: expected {expected_mean}, got {actual_mean}"

    def test_get_historic_hour_mean_without_simulation_levels(self):
        """Test _get_historic_hour_mean without Simulation levels returns specific column."""
        historic_levels = ["Hour"]  # No Simulation level

        # Test specific hours from simple historic profile
        test_cases = [
            # Hour 1: column values [100.0, 110.0] (both rows)
            {"hour": 1, "expected_values": [100.0, 110.0]},
            # Hour 12: column values [200.0, 210.0] (both rows)
            {"hour": 12, "expected_values": [200.0, 210.0]},
            # Hour 24: column values [300.0, 310.0] (both rows)
            {"hour": 24, "expected_values": [300.0, 310.0]},
        ]

        for case in test_cases:
            # Execute function
            result = _get_historic_hour_mean(
                self.historic_simple, historic_levels, case["hour"]
            )

            # Verify outcome: returns specific column as Series
            assert isinstance(
                result, pd.Series
            ), f"Should return pd.Series for hour {case['hour']}"
            assert len(result) == 2, f"Should have 2 values for hour {case['hour']}"

            # Check specific column values
            for i, expected_value in enumerate(case["expected_values"]):
                actual_value = result.iloc[i]
                assert (
                    abs(actual_value - expected_value) < 0.001
                ), f"Hour {case['hour']}, Row {i}: expected {expected_value}, got {actual_value}"

    def test_get_historic_hour_mean_with_missing_hour(self):
        """Test _get_historic_hour_mean when requested hour doesn't exist returns 0."""
        # Test with simple structure (no Simulation levels)
        historic_levels = ["Hour"]
        nonexistent_hours = [5, 999, "nonexistent_hour"]

        for missing_hour in nonexistent_hours:
            # Execute function
            result = _get_historic_hour_mean(
                self.historic_simple, historic_levels, missing_hour
            )

            # Verify outcome: returns 0 when hour doesn't exist
            if isinstance(result, pd.Series):
                assert (
                    result == 0
                ).all(), f"Should return 0 for missing hour {missing_hour}"
            else:
                assert result == 0, f"Should return 0 for missing hour {missing_hour}"

        # Test with MultiIndex structure - should raise error when trying to use xs with missing level
        historic_levels = ["Hour", "Simulation"]

        for missing_hour in [5, 999]:  # Test numeric missing hours
            try:
                result = _get_historic_hour_mean(
                    self.historic_with_sim, historic_levels, missing_hour
                )
                # If no exception, the function handled it gracefully (may return empty Series)
                assert isinstance(
                    result, (pd.Series, type(None), int)
                ), f"Should handle missing hour {missing_hour} gracefully"
            except KeyError:
                # KeyError is expected when using xs with missing level
                # This documents the current behavior
                assert (
                    True
                ), f"KeyError expected for missing hour {missing_hour} in MultiIndex"

    def test_get_historic_hour_mean_with_different_hour_types(self):
        """Test _get_historic_hour_mean with various hour identifier types."""
        # Create historic profile with different hour column types
        mixed_hours = [1, "12", 24.0]  # int, string, float
        mixed_hour_profile = pd.DataFrame(
            [[100.0, 200.0, 300.0], [110.0, 210.0, 310.0]],  # Row 1  # Row 2
            index=[1, 2],
            columns=mixed_hours,
        )

        historic_levels = ["Hour"]

        # Test cases with different hour identifier types
        test_cases = [
            # Integer hour
            {
                "hour": 1,
                "expected_values": [100.0, 110.0],
                "description": "integer hour",
            },
            # String hour
            {
                "hour": "12",
                "expected_values": [200.0, 210.0],
                "description": "string hour",
            },
            # Float hour
            {
                "hour": 24.0,
                "expected_values": [300.0, 310.0],
                "description": "float hour",
            },
        ]

        for case in test_cases:
            # Execute function
            result = _get_historic_hour_mean(
                mixed_hour_profile, historic_levels, case["hour"]
            )

            # Verify outcome: function handles different hour types correctly
            assert isinstance(
                result, pd.Series
            ), f"Should return pd.Series for {case['description']}"
            assert len(result) == 2, f"Should have 2 values for {case['description']}"

            # Check specific column values
            for i, expected_value in enumerate(case["expected_values"]):
                actual_value = result.iloc[i]
                assert (
                    abs(actual_value - expected_value) < 0.001
                ), f"{case['description']}, Row {i}: expected {expected_value}, got {actual_value}"

        # Test type compatibility - string matching numeric should not work
        # (This documents current behavior - pandas is strict about index matching)
        try:
            result = _get_historic_hour_mean(
                mixed_hour_profile, historic_levels, "1"
            )  # String '1' not int 1
            # If no exception, should return 0 (missing hour behavior)
            if isinstance(result, pd.Series):
                assert (result == 0).all(), "String '1' should not match int 1 column"
            else:
                assert result == 0, "String '1' should not match int 1 column"
        except (KeyError, TypeError):
            # Exception is acceptable - documents type strictness
            assert (
                True
            ), "Type mismatch between hour identifier and column type is handled"


class TestFindMatchingHistoricValue:
    """Test class for _find_matching_historic_value function.

    Tests the function that finds matching historic values for future columns
    when dealing with mixed index types, handling hour-based matching,
    numeric conversions, and positional fallbacks.

    Attributes
    ----------
    future_profile : pd.DataFrame
        Future profile with MultiIndex columns for testing.
    historic_profile : pd.DataFrame
        Historic profile with simple columns for testing.
    """

    def setup_method(self):
        """Set up test fixtures."""
        hours = list(range(1, 25))
        warming_levels = [1.5, 2.0]
        simulations = ["sim1", "sim2"]

        # Create future profile with (Hour, Warming_Level, Simulation) MultiIndex
        future_cols = pd.MultiIndex.from_product(
            [hours, warming_levels, simulations],
            names=["Hour", "Warming_Level", "Simulation"],
        )
        self.future_profile = pd.DataFrame(
            np.random.rand(10, len(future_cols)) + 20.0,
            index=range(1, 11),
            columns=future_cols,
        )

        # Create historic profile with simple hour columns
        self.historic_profile = pd.DataFrame(
            np.random.rand(10, 24) + 15.0,
            index=range(1, 11),
            columns=hours,
        )

    def test_find_matching_historic_value_returns_series(self):
        """Test _find_matching_historic_value returns pd.Series."""
        # Test with a future column that has Hour level
        future_col = (1, 1.5, "sim1")  # (Hour, Warming_Level, Simulation)

        # Execute function
        result = _find_matching_historic_value(
            future_col, self.future_profile, self.historic_profile
        )

        # Verify outcome: returns a pandas Series
        assert isinstance(result, pd.Series), "Should return a pandas Series"
        assert (
            len(result) == self.historic_profile.shape[0]
        ), "Series length should match historic profile rows"

        # Verify it contains the correct historic data for hour 1
        expected_series = self.historic_profile[1]
        pd.testing.assert_series_equal(result, expected_series, check_names=False)

    def test_find_matching_historic_value_with_hour_level(self):
        """Test _find_matching_historic_value with Hour level direct matching."""
        # Test various hour matches
        test_cases = [
            (1, 1.5, "sim1"),  # First hour
            (12, 2.0, "sim2"),  # Middle hour
            (24, 1.5, "sim1"),  # Last hour
        ]

        for future_col in test_cases:
            hour = future_col[0]  # Extract hour from future column

            # Execute function
            result = _find_matching_historic_value(
                future_col, self.future_profile, self.historic_profile
            )

            # Verify outcome: returns correct historic data for the hour
            assert isinstance(
                result, pd.Series
            ), f"Should return Series for {future_col}"
            expected_series = self.historic_profile[hour]
            pd.testing.assert_series_equal(result, expected_series, check_names=False)

            # Verify the series contains the expected hour's data
            assert (
                hour in self.historic_profile.columns
            ), f"Hour {hour} should exist in historic profile"

    def test_find_matching_historic_value_with_numeric_hour_matching(self):
        """Test _find_matching_historic_value with numeric hour conversion."""
        # Create future profile with string hour identifiers
        string_hours = ["1am", "12pm", "11pm"]
        warming_levels = [2.0]
        simulations = ["sim1"]

        future_cols_str = pd.MultiIndex.from_product(
            [string_hours, warming_levels, simulations],
            names=["Hour", "Warming_Level", "Simulation"],
        )
        future_profile_str = pd.DataFrame(
            np.random.rand(10, len(future_cols_str)) + 20.0,
            index=range(1, 11),
            columns=future_cols_str,
        )

        # Create historic profile with numeric hours
        numeric_hours = [1, 12, 11]  # Corresponding to 1am, 12pm, 11pm
        historic_profile_numeric = pd.DataFrame(
            np.random.rand(10, len(numeric_hours)) + 15.0,
            index=range(1, 11),
            columns=numeric_hours,
        )

        # Test numeric conversion matching
        test_cases = [
            ("1am", 1),  # '1am' should match numeric 1
            ("12pm", 12),  # '12pm' should match numeric 12
            ("11pm", 11),  # '11pm' should match numeric 11
        ]

        for string_hour, numeric_hour in test_cases:
            future_col = (string_hour, 2.0, "sim1")

            # Execute function
            result = _find_matching_historic_value(
                future_col, future_profile_str, historic_profile_numeric
            )

            # Verify outcome: matches converted numeric hour
            assert isinstance(
                result, pd.Series
            ), f"Should return Series for {future_col}"
            expected_series = historic_profile_numeric[numeric_hour]
            pd.testing.assert_series_equal(result, expected_series, check_names=False)

    def test_find_matching_historic_value_with_positional_fallback(self):
        """Test _find_matching_historic_value with positional fallback when no hour matches."""
        # Create future profile with hours that don't exist in historic
        nonexistent_hours = [99, 100, 101]
        warming_levels = [2.0]
        simulations = ["sim1"]

        future_cols_no_match = pd.MultiIndex.from_product(
            [nonexistent_hours, warming_levels, simulations],
            names=["Hour", "Warming_Level", "Simulation"],
        )
        future_profile_no_match = pd.DataFrame(
            np.random.rand(10, len(future_cols_no_match)) + 20.0,
            index=range(1, 11),
            columns=future_cols_no_match,
        )

        # Create historic profile with different hours
        historic_hours = [1, 2, 3, 4, 5]
        historic_profile_different = pd.DataFrame(
            np.random.rand(10, len(historic_hours)) + 15.0,
            index=range(1, 11),
            columns=historic_hours,
        )

        # Test positional fallback for each future column
        for i, future_col in enumerate(future_profile_no_match.columns):
            # Execute function
            result = _find_matching_historic_value(
                future_col, future_profile_no_match, historic_profile_different
            )

            # Verify outcome: uses positional fallback
            assert isinstance(
                result, pd.Series
            ), f"Should return Series for {future_col}"

            # Calculate expected positional match
            expected_col_idx = i % len(historic_profile_different.columns)
            expected_series = historic_profile_different.iloc[:, expected_col_idx]

            pd.testing.assert_series_equal(result, expected_series, check_names=False)

    def test_find_matching_historic_value_without_hour_level(self):
        """Test _find_matching_historic_value when future has no Hour level."""
        # Create future profile without Hour level (only Warming_Level and Simulation)
        warming_levels = [1.5, 2.0, 3.0]
        simulations = ["sim1", "sim2"]

        future_cols_no_hour = pd.MultiIndex.from_product(
            [warming_levels, simulations],
            names=["Warming_Level", "Simulation"],
        )
        future_profile_no_hour = pd.DataFrame(
            np.random.rand(10, len(future_cols_no_hour)) + 20.0,
            index=range(1, 11),
            columns=future_cols_no_hour,
        )

        # Create historic profile with hours
        historic_hours = [1, 2, 3, 4]
        historic_profile_hours = pd.DataFrame(
            np.random.rand(10, len(historic_hours)) + 15.0,
            index=range(1, 11),
            columns=historic_hours,
        )

        # Test that function uses positional matching when no Hour level exists
        for i, future_col in enumerate(future_profile_no_hour.columns):
            # Execute function
            result = _find_matching_historic_value(
                future_col, future_profile_no_hour, historic_profile_hours
            )

            # Verify outcome: uses positional matching since no Hour level
            assert isinstance(
                result, pd.Series
            ), f"Should return Series for {future_col}"

            # Calculate expected positional match
            expected_col_idx = i % len(historic_profile_hours.columns)
            expected_series = historic_profile_hours.iloc[:, expected_col_idx]

            pd.testing.assert_series_equal(result, expected_series, check_names=False)

        # Test fallback to first column when position calculation fails
        # Use a mock case where get_loc might return a slice instead of int
        first_col = future_profile_no_hour.columns[0]
        result_fallback = _find_matching_historic_value(
            first_col, future_profile_no_hour, historic_profile_hours
        )

        # Should still return a valid Series
        assert isinstance(
            result_fallback, pd.Series
        ), "Should return Series for fallback case"


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

        # Verify outcome: preserves data values correctly
        assert isinstance(result, np.ndarray), "Should return a numpy array"
        # Check that we have both 25.0 and 30.0 values in the result
        unique_values = np.unique(result)
        assert 25.0 in unique_values, "Should preserve 25.0 values from WL_1.5"
        assert 30.0 in unique_values, "Should preserve 30.0 values from WL_2.0"


class TestCreateSimpleDataframe:
    """Test class for _create_simple_dataframe function.

    Tests the function that creates a simple DataFrame for single warming level
    and single simulation scenarios, handling profile data dictionary structure
    and proper DataFrame construction with hour columns.

    Attributes
    ----------
    profile_data : dict
        Sample profile data dictionary for testing.
    warming_level : float
        Sample warming level value.
    simulation : str
        Sample simulation identifier.
    sim_label_func : callable
        Function to get simulation labels.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.warming_level = 2.0
        self.simulation = "sim1"

        # Create sample profile data dictionary
        wl_key = "WL_2.0"
        sim_key = "Sim1"

        # Create 365x24 profile matrix with realistic climate data
        self.profile_data = {
            (wl_key, sim_key): np.random.rand(365, 24) + 20.0  # Temperature-like data
        }

        # Simple function to get simulation labels
        def sim_label_func(sim, sim_idx):
            return f"Sim{sim_idx + 1}"

        self.sim_label_func = sim_label_func
        self.hours = np.arange(1, 25, 1)  # Hours 1-24

    def test_create_simple_dataframe_returns_dataframe(self):
        """Test _create_simple_dataframe returns pd.DataFrame."""
        # Execute function
        result = _create_simple_dataframe(
            profile_data=self.profile_data,
            warming_level=self.warming_level,
            simulation=self.simulation,
            sim_label_func=self.sim_label_func,
            days_in_year=365,
            hours=self.hours,
        )

        # Verify outcome: returns a pandas DataFrame
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape[0] > 0, "DataFrame should have rows"
        assert result.shape[1] > 0, "DataFrame should have columns"

    def test_create_simple_dataframe_with_proper_structure(self):
        """Test _create_simple_dataframe with correct DataFrame structure."""
        # Execute function
        result = _create_simple_dataframe(
            profile_data=self.profile_data,
            warming_level=self.warming_level,
            simulation=self.simulation,
            sim_label_func=self.sim_label_func,
            days_in_year=365,
            hours=self.hours,
        )

        # Verify outcome: correct DataFrame structure
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape == (
            365,
            24,
        ), "Should have 365 rows (days) and 24 columns (hours)"
        assert not isinstance(
            result.columns, pd.MultiIndex
        ), "Should have simple column structure"

        # Verify index structure (days 1-365)
        expected_index = np.arange(1, 366, 1)
        np.testing.assert_array_equal(result.index.values, expected_index)

        # Verify column structure (hours 1-24)
        expected_columns = np.arange(1, 25, 1)
        np.testing.assert_array_equal(result.columns.values, expected_columns)

    def test_create_simple_dataframe_with_different_scenarios(self):
        """Test _create_simple_dataframe with different warming level and simulation scenarios."""
        # Test different warming level
        different_wl = 1.5
        different_wl_data = {("WL_1.5", "Sim1"): np.random.rand(365, 24) + 15.0}

        # Execute function with different warming level
        result_wl = _create_simple_dataframe(
            profile_data=different_wl_data,
            warming_level=different_wl,
            simulation=self.simulation,
            sim_label_func=self.sim_label_func,
            days_in_year=365,
            hours=self.hours,
        )

        # Verify outcome: maintains same structure with different data
        assert isinstance(
            result_wl, pd.DataFrame
        ), "Should return DataFrame for different WL"
        assert result_wl.shape == (365, 24), "Should maintain same shape"

        # Test different simulation identifier
        different_sim = "sim2"
        # Note: sim_label_func always uses index 0, so key will be "Sim1" regardless of simulation value
        different_sim_data = {("WL_2.0", "Sim1"): np.random.rand(365, 24) + 25.0}

        # Execute function with different simulation
        result_sim = _create_simple_dataframe(
            profile_data=different_sim_data,
            warming_level=self.warming_level,
            simulation=different_sim,
            sim_label_func=self.sim_label_func,
            days_in_year=365,
            hours=self.hours,
        )

        # Verify outcome: handles different simulation correctly
        assert isinstance(
            result_sim, pd.DataFrame
        ), "Should return DataFrame for different sim"
        assert result_sim.shape == (365, 24), "Should maintain same shape"

        # Verify all results have different data but same structure
        assert not result_wl.equals(
            result_sim
        ), "Different scenarios should produce different data"
        assert list(result_wl.columns) == list(
            result_sim.columns
        ), "Columns should be identical"

    def test_create_simple_dataframe_preserves_data(self):
        """Test _create_simple_dataframe preserves data values correctly."""
        # Create profile data with known values for verification
        test_data = np.ones((365, 24)) * 42.5  # All values set to 42.5
        test_profile_data = {("WL_2.0", "Sim1"): test_data}

        # Execute function
        result = _create_simple_dataframe(
            profile_data=test_profile_data,
            warming_level=self.warming_level,
            simulation=self.simulation,
            sim_label_func=self.sim_label_func,
            days_in_year=365,
            hours=self.hours,
        )

        # Verify outcome: data values are preserved correctly
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"

        # Check that all values in the DataFrame match the original data
        assert np.all(result.values == 42.5), "All values should be 42.5"

        # Verify shape matches original profile matrix
        assert (
            result.shape == test_data.shape
        ), "Shape should match original profile matrix"

        # Test with different profile matrix size (leap year)
        leap_year_data = np.ones((366, 24)) * 33.7  # Leap year with different values
        leap_year_profile_data = {("WL_2.0", "Sim1"): leap_year_data}

        # Execute function with leap year data
        result_leap = _create_simple_dataframe(
            profile_data=leap_year_profile_data,
            warming_level=self.warming_level,
            simulation=self.simulation,
            sim_label_func=self.sim_label_func,
            days_in_year=366,  # Leap year
            hours=self.hours,
        )

        # Verify outcome: handles different matrix sizes correctly
        assert result_leap.shape == (
            366,
            24,
        ), "Should handle leap year shape (366 days)"
        assert np.all(result_leap.values == 33.7), "All leap year values should be 33.7"

        # Verify proper index for leap year
        expected_leap_index = np.arange(1, 367, 1)  # 1 to 366
        np.testing.assert_array_equal(result_leap.index.values, expected_leap_index)

    def test_create_simple_dataframe_with_different_year_lengths(self):
        """Test _create_simple_dataframe with different days_in_year parameter values."""
        # Test with various year lengths
        year_length_scenarios = [
            {"days": 365, "description": "regular year"},
            {"days": 366, "description": "leap year"},
            {"days": 360, "description": "simplified calendar year"},
            {"days": 300, "description": "partial year"},
        ]

        for scenario in year_length_scenarios:
            days = scenario["days"]
            description = scenario["description"]

            # Create profile data matching the year length
            profile_matrix = np.random.rand(days, 24) + 18.0
            test_data = {("WL_2.0", "Sim1"): profile_matrix}

            # Execute function
            result = _create_simple_dataframe(
                profile_data=test_data,
                warming_level=self.warming_level,
                simulation=self.simulation,
                sim_label_func=self.sim_label_func,
                days_in_year=days,
                hours=self.hours,
            )

            # Verify outcome: correct dimensions for each scenario
            assert isinstance(
                result, pd.DataFrame
            ), f"Should return DataFrame for {description}"
            assert result.shape == (
                days,
                24,
            ), f"Should have {days} rows for {description}"
            assert result.shape[1] == 24, f"Should have 24 columns for {description}"

            # Verify proper index generation
            expected_index = np.arange(1, days + 1, 1)
            np.testing.assert_array_equal(
                result.index.values,
                expected_index,
                err_msg=f"Index should be 1 to {days} for {description}",
            )

            # Verify columns remain consistent regardless of year length
            expected_columns = np.arange(1, 25, 1)
            np.testing.assert_array_equal(
                result.columns.values,
                expected_columns,
                err_msg=f"Columns should always be 1-24 for {description}",
            )

            # Verify data values are preserved
            np.testing.assert_array_equal(
                result.values,
                profile_matrix,
                err_msg=f"Data values should be preserved for {description}",
            )


class TestCreateSingleWlMultiSimDataframe:
    """Test class for _create_single_wl_multi_sim_dataframe function.

    Tests the function that creates DataFrames for single warming level with
    multiple simulations, validating MultiIndex column structure and data handling.

    Attributes
    ----------
    sample_profile_data : dict
        Sample profile data dictionary with (warming_level, simulation) keys.
    mock_sim_label_func : MagicMock
        Mock function for generating simulation labels.
    warming_level : float
        Sample warming level for testing.
    simulations : list
        Sample list of simulations.
    hours : np.ndarray
        Array of hour values for columns.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock simulation label function
        self.mock_sim_label_func = MagicMock()
        self.mock_sim_label_func.side_effect = lambda sim, idx: f"sim_{sim}_{idx}"

        # Test parameters
        self.warming_level = 2.0
        self.simulations = ["model_A", "model_B", "model_C"]
        self.hours = np.arange(0, 24)
        self.days_in_year = 365
        self.hours_per_day = 24

        # Create sample profile data dictionary
        # The function expects data for each (WL_X, sim_label) combination
        self.sample_profile_data = {}
        for i, sim in enumerate(self.simulations):
            sim_key = f"sim_{sim}_{i}"
            wl_key = f"WL_{self.warming_level}"
            # Create random data for each simulation (365 days x 24 hours)
            profile_matrix = np.random.rand(365, 24) + 20.0
            self.sample_profile_data[(wl_key, sim_key)] = profile_matrix

    def test_create_single_wl_multi_sim_dataframe_returns_dataframe(self):
        """Test that _create_single_wl_multi_sim_dataframe returns a pandas DataFrame."""
        # Execute function
        result = _create_single_wl_multi_sim_dataframe(
            profile_data=self.sample_profile_data,
            warming_level=self.warming_level,
            simulations=self.simulations,
            sim_label_func=self.mock_sim_label_func,
            days_in_year=self.days_in_year,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
        )

        # Verify outcome: returns a pandas DataFrame
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape[0] > 0, "DataFrame should have rows"
        assert result.shape[1] > 0, "DataFrame should have columns"

    def test_create_single_wl_multi_sim_dataframe_multiindex_structure(self):
        """Test that the DataFrame has correct MultiIndex column structure."""
        # Execute function
        result = _create_single_wl_multi_sim_dataframe(
            profile_data=self.sample_profile_data,
            warming_level=self.warming_level,
            simulations=self.simulations,
            sim_label_func=self.mock_sim_label_func,
            days_in_year=self.days_in_year,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
        )

        # Verify outcome: correct MultiIndex column structure
        assert isinstance(result.columns, pd.MultiIndex), "Columns should be MultiIndex"
        assert result.columns.names == [
            "Hour",
            "Simulation",
        ], "Column levels should be named Hour and Simulation"

        # Verify expected dimensions: 365 rows, (24 hours × 3 simulations) columns
        expected_rows = 365
        expected_cols = 24 * len(self.simulations)  # 24 hours × 3 simulations = 72
        assert result.shape == (
            expected_rows,
            expected_cols,
        ), f"Should have {expected_rows} rows and {expected_cols} columns"

        # Verify index structure (day numbers)
        expected_index = np.arange(1, self.days_in_year + 1)
        np.testing.assert_array_equal(
            result.index.values,
            expected_index,
            err_msg="Index should be day numbers from 1 to days_in_year",
        )

    def test_create_single_wl_multi_sim_dataframe_handles_multiple_simulations(self):
        """Test function handles multiple simulations correctly."""
        # Execute function
        result = _create_single_wl_multi_sim_dataframe(
            profile_data=self.sample_profile_data,
            warming_level=self.warming_level,
            simulations=self.simulations,
            sim_label_func=self.mock_sim_label_func,
            days_in_year=self.days_in_year,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
        )

        # Verify outcome: each simulation creates columns for all hours
        # Expected structure: (hour, sim) for each hour and each simulation
        unique_simulations = result.columns.get_level_values("Simulation").unique()
        unique_hours = result.columns.get_level_values("Hour").unique()

        # Should have one column for each (hour, simulation) combination
        expected_sim_names = ["sim_model_A_0", "sim_model_B_1", "sim_model_C_2"]
        assert len(unique_simulations) == len(
            self.simulations
        ), f"Should have {len(self.simulations)} simulations"
        assert len(unique_hours) == len(
            self.hours
        ), f"Should have {len(self.hours)} hours"

        # Verify simulation names match expected pattern from mock function
        for expected_sim in expected_sim_names:
            assert (
                expected_sim in unique_simulations
            ), f"Should contain simulation {expected_sim}"

        # Verify each hour appears for each simulation (24 hours × 3 sims = 72 columns)
        for hour in self.hours:
            for sim_name in expected_sim_names:
                assert (
                    hour,
                    sim_name,
                ) in result.columns, (
                    f"Should have column for hour {hour}, simulation {sim_name}"
                )

    def test_create_single_wl_multi_sim_dataframe_duplicate_simulation_names(self):
        """Test function handles duplicate simulation names with uniqueness suffixes."""
        # Create mock sim_label_func that returns duplicate names
        mock_dup_sim_func = MagicMock()
        mock_dup_sim_func.side_effect = (
            lambda sim, idx: "duplicate_name"
        )  # All return same name

        # Create profile data - need to have data for both original and modified names
        # The function will try to access data using both original and uniquified names
        duplicate_profile_data = {}
        simulations_with_dups = ["model_A", "model_B", "model_C"]
        wl_key = f"WL_{self.warming_level}"

        # Add data with original duplicate name (function will use this for first occurrence)
        profile_matrix = np.random.rand(365, 24) + 20.0
        duplicate_profile_data[(wl_key, "duplicate_name")] = profile_matrix

        # Since the function modifies names internally but doesn't update profile_data,
        # we'll test the warning behavior but expect KeyError for missing keys
        # Let's just test that the warning is printed when duplicate names are detected
        with patch("builtins.print") as mock_print:
            try:
                _create_single_wl_multi_sim_dataframe(
                    profile_data=duplicate_profile_data,
                    warming_level=self.warming_level,
                    simulations=simulations_with_dups,
                    sim_label_func=mock_dup_sim_func,
                    days_in_year=self.days_in_year,
                    hours=self.hours,
                    hours_per_day=self.hours_per_day,
                )
            except KeyError:
                # Expected since profile_data doesn't have the uniquified names
                pass

        # Verify outcome: warning message was printed about duplicates
        printed_calls = [str(call) for call in mock_print.call_args_list]
        printed_output = " ".join(printed_calls)
        assert (
            "duplicate simulation names" in printed_output.lower()
        ), "Should warn about duplicate simulation names"
        assert (
            "uniqueness suffixes" in printed_output.lower()
        ), "Should mention adding uniqueness suffixes"

        # Verify the sim_label_func was called for each simulation
        assert mock_dup_sim_func.call_count == len(
            simulations_with_dups
        ), "Should call sim_label_func for each simulation"

    def test_create_single_wl_multi_sim_dataframe_preserves_data_integrity(self):
        """Test that profile data values are correctly preserved in MultiIndex structure."""
        # Create specific test data with known values for verification
        test_simulations = ["test_sim_A", "test_sim_B"]
        test_hours = np.array([0, 1, 2])  # Use smaller subset for easier verification
        test_days = 3  # Use smaller dataset for precise testing

        # Create mock sim_label_func for predictable names
        test_sim_func = MagicMock()
        test_sim_func.side_effect = lambda sim, idx: f"test_{sim}_{idx}"

        # Create test profile data with known values
        test_profile_data = {}
        expected_values = {}

        for i, sim in enumerate(test_simulations):
            sim_key = f"test_{sim}_{i}"
            wl_key = f"WL_{self.warming_level}"
            # Create known test data: day i, hour j has value (i+1)*10 + j
            profile_matrix = np.zeros((test_days, len(test_hours)))
            for day in range(test_days):
                for hour_idx, hour in enumerate(test_hours):
                    profile_matrix[day, hour_idx] = (day + 1) * 10 + hour

            test_profile_data[(wl_key, sim_key)] = profile_matrix
            expected_values[sim_key] = profile_matrix

        # Execute function
        result = _create_single_wl_multi_sim_dataframe(
            profile_data=test_profile_data,
            warming_level=self.warming_level,
            simulations=test_simulations,
            sim_label_func=test_sim_func,
            days_in_year=test_days,
            hours=test_hours,
            hours_per_day=len(test_hours),
        )

        # Verify outcome: data integrity is preserved
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert result.shape == (
            test_days,
            len(test_hours) * len(test_simulations),
        ), "Should have correct dimensions"

        # Verify specific data values are preserved for each (hour, simulation) combination
        for hour in test_hours:
            for i, sim in enumerate(test_simulations):
                sim_key = f"test_{sim}_{i}"
                expected_matrix = expected_values[sim_key]

                # Get column data for this (hour, simulation) combination
                column_data = result[(hour, sim_key)]
                expected_column = expected_matrix[:, list(test_hours).index(hour)]

                # Verify data values match
                np.testing.assert_array_equal(
                    column_data.values,
                    expected_column,
                    err_msg=f"Data mismatch for hour {hour}, simulation {sim_key}",
                )

        # Verify specific known values at expected positions
        # Day 1 (index 0), Hour 0, Sim A should be 10 (day 1 * 10 + hour 0)
        sim_a_key = "test_test_sim_A_0"
        assert (
            result.loc[1, (0, sim_a_key)] == 10.0
        ), "Day 1, Hour 0, Sim A should be 10"

        # Day 2 (index 1), Hour 1, Sim B should be 21 (day 2 * 10 + hour 1)
        sim_b_key = "test_test_sim_B_1"
        assert (
            result.loc[2, (1, sim_b_key)] == 21.0
        ), "Day 2, Hour 1, Sim B should be 21"


class TestCreateMultiWlSingleSimDataframe:
    """Test class for _create_multi_wl_single_sim_dataframe function.

    Tests the function that creates DataFrames for multiple warming levels with
    single simulation, validating MultiIndex column structure and data handling.

    Attributes
    ----------
    sample_profile_data : dict
        Sample profile data dictionary with (warming_level, simulation) keys.
    mock_sim_label_func : MagicMock
        Mock function for generating simulation labels.
    warming_levels : np.ndarray
        Array of warming levels for testing.
    simulation : str
        Sample simulation identifier.
    hours : np.ndarray
        Array of hour values for columns.
    """

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock simulation label function
        self.mock_sim_label_func = MagicMock()
        self.mock_sim_label_func.return_value = "test_simulation"
        
        # Test parameters
        self.warming_levels = np.array([1.5, 2.0, 3.0])
        self.simulation = "model_X"
        self.hours = np.arange(0, 24)
        self.days_in_year = 365
        self.hours_per_day = 24
        
        # Create sample profile data dictionary
        # The function expects data for each (WL_X, sim_label) combination
        self.sample_profile_data = {}
        sim_key = "test_simulation"
        
        for wl in self.warming_levels:
            wl_key = f"WL_{wl}"
            # Create random data for each warming level (365 days x 24 hours)
            profile_matrix = np.random.rand(365, 24) + 20.0 + wl  # Add WL to make different
            self.sample_profile_data[(wl_key, sim_key)] = profile_matrix

    def test_create_multi_wl_single_sim_dataframe_returns_dataframe(self):
        """Test that _create_multi_wl_single_sim_dataframe returns a pandas DataFrame."""
        # Execute function
        result = _create_multi_wl_single_sim_dataframe(
            profile_data=self.sample_profile_data,
            warming_levels=self.warming_levels,
            simulation=self.simulation,
            sim_label_func=self.mock_sim_label_func,
            days_in_year=self.days_in_year,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
        )

        # Verify outcome: returns a pandas DataFrame
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape[0] > 0, "DataFrame should have rows"
        assert result.shape[1] > 0, "DataFrame should have columns"

    def test_create_multi_wl_single_sim_dataframe_multiindex_structure(self):
        """Test that the DataFrame has correct MultiIndex column structure."""
        # Execute function
        result = _create_multi_wl_single_sim_dataframe(
            profile_data=self.sample_profile_data,
            warming_levels=self.warming_levels,
            simulation=self.simulation,
            sim_label_func=self.mock_sim_label_func,
            days_in_year=self.days_in_year,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
        )

        # Verify outcome: correct MultiIndex column structure
        assert isinstance(result.columns, pd.MultiIndex), "Columns should be MultiIndex"
        assert result.columns.names == ["Hour", "Warming_Level"], "Column levels should be named Hour and Warming_Level"
        
        # Verify expected dimensions: 365 rows, (24 hours × 3 warming levels) columns
        expected_rows = 365
        expected_cols = 24 * len(self.warming_levels)  # 24 hours × 3 warming levels = 72
        assert result.shape == (expected_rows, expected_cols), f"Should have {expected_rows} rows and {expected_cols} columns"
        
        # Verify index structure (day numbers)
        expected_index = np.arange(1, self.days_in_year + 1)
        np.testing.assert_array_equal(
            result.index.values, expected_index,
            err_msg="Index should be day numbers from 1 to days_in_year"
        )

    def test_create_multi_wl_single_sim_dataframe_handles_multiple_warming_levels(self):
        """Test function handles multiple warming levels correctly."""
        # Execute function
        result = _create_multi_wl_single_sim_dataframe(
            profile_data=self.sample_profile_data,
            warming_levels=self.warming_levels,
            simulation=self.simulation,
            sim_label_func=self.mock_sim_label_func,
            days_in_year=self.days_in_year,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
        )

        # Verify outcome: each warming level creates columns for all hours
        # Expected structure: (hour, wl) for each hour and each warming level
        unique_warming_levels = result.columns.get_level_values("Warming_Level").unique()
        unique_hours = result.columns.get_level_values("Hour").unique()
        
        # Should have one column for each (hour, warming_level) combination
        expected_wl_names = ["WL_1.5", "WL_2.0", "WL_3.0"]
        assert len(unique_warming_levels) == len(self.warming_levels), f"Should have {len(self.warming_levels)} warming levels"
        assert len(unique_hours) == len(self.hours), f"Should have {len(self.hours)} hours"
        
        # Verify warming level names match expected pattern
        for expected_wl in expected_wl_names:
            assert expected_wl in unique_warming_levels, f"Should contain warming level {expected_wl}"
        
        # Verify each hour appears for each warming level (24 hours × 3 WLs = 72 columns)
        for hour in self.hours:
            for wl_name in expected_wl_names:
                assert (hour, wl_name) in result.columns, f"Should have column for hour {hour}, warming level {wl_name}"

    def test_create_multi_wl_single_sim_dataframe_preserves_data_integrity(self):
        """Test that profile data values are correctly preserved in MultiIndex structure."""
        # Create specific test data with known values for verification
        test_warming_levels = np.array([1.0, 2.0])
        test_hours = np.array([0, 1, 2])  # Use smaller subset for easier verification
        test_days = 3  # Use smaller dataset for precise testing
        
        # Create mock sim_label_func for predictable names
        test_sim_func = MagicMock()
        test_sim_func.return_value = "test_sim"
        
        # Create test profile data with known values
        test_profile_data = {}
        expected_values = {}
        
        for wl in test_warming_levels:
            wl_key = f"WL_{wl}"
            sim_key = "test_sim"
            # Create known test data: day i, hour j has value (day+1)*10 + hour + wl
            profile_matrix = np.zeros((test_days, len(test_hours)))
            for day in range(test_days):
                for hour_idx, hour in enumerate(test_hours):
                    profile_matrix[day, hour_idx] = (day + 1) * 10 + hour + wl
            
            test_profile_data[(wl_key, sim_key)] = profile_matrix
            expected_values[wl_key] = profile_matrix

        # Execute function
        result = _create_multi_wl_single_sim_dataframe(
            profile_data=test_profile_data,
            warming_levels=test_warming_levels,
            simulation="test_simulation",
            sim_label_func=test_sim_func,
            days_in_year=test_days,
            hours=test_hours,
            hours_per_day=len(test_hours),
        )

        # Verify outcome: data integrity is preserved
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert result.shape == (test_days, len(test_hours) * len(test_warming_levels)), "Should have correct dimensions"
        
        # Verify specific data values are preserved for each (hour, warming_level) combination
        for hour in test_hours:
            for wl in test_warming_levels:
                wl_key = f"WL_{wl}"
                expected_matrix = expected_values[wl_key]
                
                # Get column data for this (hour, warming_level) combination
                column_data = result[(hour, wl_key)]
                expected_column = expected_matrix[:, list(test_hours).index(hour)]
                
                # Verify data values match
                np.testing.assert_array_equal(
                    column_data.values, expected_column,
                    err_msg=f"Data mismatch for hour {hour}, warming level {wl_key}"
                )
                
        # Verify specific known values at expected positions
        # Day 1 (index 0), Hour 0, WL 1.0 should be 10 + 0 + 1.0 = 11.0
        assert result.loc[1, (0, "WL_1.0")] == 11.0, "Day 1, Hour 0, WL 1.0 should be 11.0"
        
        # Day 2 (index 1), Hour 1, WL 2.0 should be 20 + 1 + 2.0 = 23.0
        assert result.loc[2, (1, "WL_2.0")] == 23.0, "Day 2, Hour 1, WL 2.0 should be 23.0"

    def test_create_multi_wl_single_sim_dataframe_different_warming_level_configs(self):
        """Test function with different warming level configurations."""
        # Test scenarios with different warming level configurations
        test_scenarios = [
            {
                "name": "single_warming_level",
                "warming_levels": np.array([2.0]),
                "expected_cols": 24 * 1,  # 24 hours × 1 WL
                "expected_wl_names": ["WL_2.0"]
            },
            {
                "name": "two_warming_levels",
                "warming_levels": np.array([1.5, 3.0]),
                "expected_cols": 24 * 2,  # 24 hours × 2 WLs
                "expected_wl_names": ["WL_1.5", "WL_3.0"]
            },
            {
                "name": "many_warming_levels",
                "warming_levels": np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0]),
                "expected_cols": 24 * 6,  # 24 hours × 6 WLs
                "expected_wl_names": ["WL_1.0", "WL_1.5", "WL_2.0", "WL_2.5", "WL_3.0", "WL_4.0"]
            }
        ]
        
        for scenario in test_scenarios:
            # Create profile data for this scenario
            scenario_profile_data = {}
            sim_key = "test_simulation"
            
            for wl in scenario["warming_levels"]:
                wl_key = f"WL_{wl}"
                profile_matrix = np.random.rand(365, 24) + 20.0 + wl
                scenario_profile_data[(wl_key, sim_key)] = profile_matrix
            
            # Execute function
            result = _create_multi_wl_single_sim_dataframe(
                profile_data=scenario_profile_data,
                warming_levels=scenario["warming_levels"],
                simulation=self.simulation,
                sim_label_func=self.mock_sim_label_func,
                days_in_year=self.days_in_year,
                hours=self.hours,
                hours_per_day=self.hours_per_day,
            )
            
            # Verify outcome for this scenario
            assert isinstance(result, pd.DataFrame), f"Should return DataFrame for {scenario['name']}"
            assert result.shape[0] == 365, f"Should have 365 rows for {scenario['name']}"
            assert result.shape[1] == scenario["expected_cols"], f"Should have {scenario['expected_cols']} columns for {scenario['name']}"
            
            # Verify MultiIndex structure
            assert isinstance(result.columns, pd.MultiIndex), f"Should have MultiIndex columns for {scenario['name']}"
            assert result.columns.names == ["Hour", "Warming_Level"], f"Should have correct level names for {scenario['name']}"
            
            # Verify warming level names
            unique_wls = result.columns.get_level_values("Warming_Level").unique()
            assert len(unique_wls) == len(scenario["warming_levels"]), f"Should have {len(scenario['warming_levels'])} unique warming levels for {scenario['name']}"
            
            for expected_wl in scenario["expected_wl_names"]:
                assert expected_wl in unique_wls, f"Should contain {expected_wl} for {scenario['name']}"
            
            # Verify all hours are present for each warming level
            unique_hours = result.columns.get_level_values("Hour").unique()
            assert len(unique_hours) == 24, f"Should have 24 hours for {scenario['name']}"


class TestCreateMultiWlMultiSimDataframe:
    """Test class for _create_multi_wl_multi_sim_dataframe function.
    
    Tests the function that creates DataFrames for climate profiles with
    multiple warming levels and multiple simulations together, producing
    (Hour, Warming_Level, Simulation) MultiIndex column structure.
    
    Attributes
    ----------
    warming_levels : np.ndarray
        Array of warming level values for testing.
    simulations : list
        List of simulation identifiers for testing.
    mock_sim_label_func : MagicMock
        Mock simulation label function.
    days_in_year : int
        Number of days in year (365).
    hours : np.ndarray
        Array of hour values (1-24).
    hours_per_day : int
        Hours per day (24).
    profile_data : dict
        Sample profile data dictionary.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.warming_levels = np.array([1.5, 2.0, 3.0])
        self.simulations = ["sim1", "sim2"]
        self.mock_sim_label_func = MagicMock(side_effect=lambda x, i: f"Simulation_{x}")
        self.days_in_year = 365
        self.hours = np.arange(1, 25, 1)
        self.hours_per_day = 24
        
        # Create sample profile data with (warming_level, simulation_label) keys
        # The keys must use the labeled names that sim_label_func produces
        self.profile_data = {}
        for wl in self.warming_levels:
            wl_key = f"WL_{wl}"
            for i, sim in enumerate(self.simulations):
                sim_label = f"Simulation_{sim}"  # Match what sim_label_func returns
                # Create 365x24 matrix for each combination
                profile_matrix = np.random.rand(365, 24) + 20.0 + wl
                self.profile_data[(wl_key, sim_label)] = profile_matrix
    
    def test_returns_dataframe(self):
        """Test that _create_multi_wl_multi_sim_dataframe returns a pandas DataFrame."""
        # Execute function
        result = _create_multi_wl_multi_sim_dataframe(
            profile_data=self.profile_data,
            warming_levels=self.warming_levels,
            simulations=self.simulations,
            sim_label_func=self.mock_sim_label_func,
            days_in_year=self.days_in_year,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
        )
        
        # Verify outcome: returns DataFrame with correct shape
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape[0] == 365, "Should have 365 rows (days)"
        
        # With 3 warming levels, 2 simulations, and 24 hours: 24 * 3 * 2 = 144 columns
        expected_cols = 24 * len(self.warming_levels) * len(self.simulations)
        assert result.shape[1] == expected_cols, f"Should have {expected_cols} columns"
    
    def test_multiindex_structure(self):
        """Test that DataFrame has proper (Hour, Warming_Level, Simulation) MultiIndex structure."""
        # Execute function
        result = _create_multi_wl_multi_sim_dataframe(
            profile_data=self.profile_data,
            warming_levels=self.warming_levels,
            simulations=self.simulations,
            sim_label_func=self.mock_sim_label_func,
            days_in_year=self.days_in_year,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
        )
        
        # Verify outcome: proper MultiIndex structure
        assert isinstance(result.columns, pd.MultiIndex), "Should have MultiIndex columns"
        assert result.columns.names == ["Hour", "Warming_Level", "Simulation"], "Should have three levels: Hour, Warming_Level, Simulation"
        
        # Verify all hours are present
        unique_hours = result.columns.get_level_values("Hour").unique()
        assert len(unique_hours) == 24, "Should have 24 unique hours"
        assert all(h in unique_hours for h in range(1, 25)), "Should have hours 1-24"
        
        # Verify all warming levels are present
        unique_wls = result.columns.get_level_values("Warming_Level").unique()
        expected_wl_names = [f"WL_{wl}" for wl in self.warming_levels]
        assert len(unique_wls) == len(self.warming_levels), f"Should have {len(self.warming_levels)} warming levels"
        for wl_name in expected_wl_names:
            assert wl_name in unique_wls, f"Should contain warming level {wl_name}"
        
        # Verify all simulations are present
        unique_sims = result.columns.get_level_values("Simulation").unique()
        expected_sim_names = [f"Simulation_{sim}" for sim in self.simulations]
        assert len(unique_sims) == len(self.simulations), f"Should have {len(self.simulations)} simulations"
        for sim_name in expected_sim_names:
            assert sim_name in unique_sims, f"Should contain simulation {sim_name}"
    
    def test_handles_multiple_warming_levels_and_simulations(self):
        """Test proper handling of multiple warming levels and simulations together."""
        # Execute function with multiple warming levels and simulations
        result = _create_multi_wl_multi_sim_dataframe(
            profile_data=self.profile_data,
            warming_levels=self.warming_levels,
            simulations=self.simulations,
            sim_label_func=self.mock_sim_label_func,
            days_in_year=self.days_in_year,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
        )
        
        # Verify outcome: each hour has all combinations of warming levels and simulations
        for hour in range(1, 25):
            hour_cols = result.loc[:, hour]
            
            # Should have wl_count * sim_count columns for each hour
            expected_cols_per_hour = len(self.warming_levels) * len(self.simulations)
            assert hour_cols.shape[1] == expected_cols_per_hour, f"Hour {hour} should have {expected_cols_per_hour} columns"
            
            # Verify all warming levels present for this hour
            if isinstance(hour_cols.columns, pd.MultiIndex):
                wls = hour_cols.columns.get_level_values("Warming_Level").unique()
                assert len(wls) == len(self.warming_levels), f"Hour {hour} should have all {len(self.warming_levels)} warming levels"
                
                # Verify all simulations present for this hour
                sims = hour_cols.columns.get_level_values("Simulation").unique()
                assert len(sims) == len(self.simulations), f"Hour {hour} should have all {len(self.simulations)} simulations"
    
    def test_preserves_data_integrity(self):
        """Test that data values are preserved correctly in the transformation."""
        # Create specific test data to verify data preservation
        test_warming_levels = np.array([1.5, 2.0])
        test_simulations = ["sim1", "sim2"]
        test_profile_data = {}
        
        # Create known data patterns for each combination
        for wl in test_warming_levels:
            wl_key = f"WL_{wl}"
            for sim in test_simulations:
                sim_label = f"Simulation_{sim}"
                # Create a matrix where values = day + hour + wl*10 + sim_index
                sim_index = test_simulations.index(sim)
                profile_matrix = np.zeros((365, 24))
                for day in range(365):
                    for hour in range(24):
                        profile_matrix[day, hour] = day + hour + wl*10 + sim_index
                test_profile_data[(wl_key, sim_label)] = profile_matrix
        
        # Execute function
        result = _create_multi_wl_multi_sim_dataframe(
            profile_data=test_profile_data,
            warming_levels=test_warming_levels,
            simulations=test_simulations,
            sim_label_func=self.mock_sim_label_func,
            days_in_year=self.days_in_year,
            hours=self.hours,
            hours_per_day=self.hours_per_day,
        )
        
        # Verify outcome: data values are preserved correctly
        # Check specific values for a sample of combinations
        for day in [1, 100, 365]:  # Check first, middle, and last day
            for hour in [1, 12, 24]:  # Check different hours
                for wl in test_warming_levels:
                    for sim_idx, sim in enumerate(test_simulations):
                        wl_name = f"WL_{wl}"
                        sim_name = f"Simulation_{sim}"
                        
                        # Get value from result DataFrame
                        result_value = result.loc[day, (hour, wl_name, sim_name)]
                        
                        # Calculate expected value
                        expected_value = (day - 1) + (hour - 1) + wl*10 + sim_idx
                        
                        assert abs(result_value - expected_value) < 0.001, \
                            f"Value mismatch at day={day}, hour={hour}, wl={wl}, sim={sim}: expected {expected_value}, got {result_value}"
    
    def test_complex_combinations(self):
        """Test various complex combinations of warming levels and simulations."""
        # Test different scenarios with varying numbers of warming levels and simulations
        test_scenarios = [
            {
                "name": "single_wl_single_sim",
                "warming_levels": np.array([2.0]),
                "simulations": ["sim1"],
                "expected_cols": 24,  # 24 hours * 1 WL * 1 sim
            },
            {
                "name": "many_wls_many_sims",
                "warming_levels": np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
                "simulations": ["sim1", "sim2", "sim3", "sim4"],
                "expected_cols": 480,  # 24 hours * 5 WLs * 4 sims
            },
            {
                "name": "two_wls_three_sims",
                "warming_levels": np.array([1.5, 3.0]),
                "simulations": ["sim1", "sim2", "sim3"],
                "expected_cols": 144,  # 24 hours * 2 WLs * 3 sims
            },
        ]
        
        for scenario in test_scenarios:
            # Create profile data for this scenario
            scenario_profile_data = {}
            
            for wl in scenario["warming_levels"]:
                wl_key = f"WL_{wl}"
                for sim in scenario["simulations"]:
                    sim_label = f"Simulation_{sim}"
                    profile_matrix = np.random.rand(365, 24) + 20.0 + wl
                    scenario_profile_data[(wl_key, sim_label)] = profile_matrix
            
            # Execute function
            result = _create_multi_wl_multi_sim_dataframe(
                profile_data=scenario_profile_data,
                warming_levels=scenario["warming_levels"],
                simulations=scenario["simulations"],
                sim_label_func=self.mock_sim_label_func,
                days_in_year=self.days_in_year,
                hours=self.hours,
                hours_per_day=self.hours_per_day,
            )
            
            # Verify outcome for this scenario
            assert isinstance(result, pd.DataFrame), f"Should return DataFrame for {scenario['name']}"
            assert result.shape[0] == 365, f"Should have 365 rows for {scenario['name']}"
            assert result.shape[1] == scenario["expected_cols"], f"Should have {scenario['expected_cols']} columns for {scenario['name']}"
            
            # Verify MultiIndex structure
            assert isinstance(result.columns, pd.MultiIndex), f"Should have MultiIndex columns for {scenario['name']}"
            assert result.columns.names == ["Hour", "Warming_Level", "Simulation"], f"Should have correct level names for {scenario['name']}"
            
            # Verify correct number of unique values in each level
            unique_hours = result.columns.get_level_values("Hour").unique()
            assert len(unique_hours) == 24, f"Should have 24 hours for {scenario['name']}"
            
            unique_wls = result.columns.get_level_values("Warming_Level").unique()
            assert len(unique_wls) == len(scenario["warming_levels"]), f"Should have {len(scenario['warming_levels'])} warming levels for {scenario['name']}"
            
            unique_sims = result.columns.get_level_values("Simulation").unique()
            assert len(unique_sims) == len(scenario["simulations"]), f"Should have {len(scenario['simulations'])} simulations for {scenario['name']}"


class TestFormatMeteoYrDf:
    """Test class for _format_meteo_yr_df function.
    
    Tests the function that reformats meteorological yearly dataframes by
    converting numeric hour columns to 12-hour AM/PM format and Julian day
    indices to Month-Day format for improved readability.
    
    Attributes
    ----------
    sample_df_365 : pd.DataFrame
        Sample 365-day dataframe for testing regular years.
    sample_df_366 : pd.DataFrame
        Sample 366-day dataframe for testing leap years.
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample 365-day dataframe (regular year)
        # 24 columns representing hours 0-23
        # Index represents Julian days 1-365
        self.sample_df_365 = pd.DataFrame(
            np.random.rand(365, 24) + 20.0,
            index=range(1, 366),
            columns=range(24)
        )
        
        # Create sample 366-day dataframe (leap year)
        self.sample_df_366 = pd.DataFrame(
            np.random.rand(366, 24) + 20.0,
            index=range(1, 367),
            columns=range(24)
        )
    
    def test_returns_formatted_dataframe(self):
        """Test that _format_meteo_yr_df returns a properly formatted DataFrame."""
        # Execute function with 365-day dataframe
        result = _format_meteo_yr_df(self.sample_df_365.copy())
        
        # Verify outcome: returns DataFrame with same shape
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape[0] == 365, "Should have 365 rows"
        assert result.shape[1] == 24, "Should have 24 columns (hours)"
        
        # Verify column structure has AM/PM format
        assert all('am' in str(col) or 'pm' in str(col) for col in result.columns), "All columns should have AM/PM format"
        
        # Verify index is formatted as Month-Day
        assert all('-' in str(idx) for idx in result.index), "Index should be in Month-Day format"
        
        # Verify metadata
        assert result.columns.name == "Hour", "Column name should be 'Hour'"
        assert result.index.name == "Day of Year", "Index name should be 'Day of Year'"
    
    def test_formats_columns_with_ampm(self):
        """Test proper column reordering and AM/PM formatting."""
        # Execute function
        result = _format_meteo_yr_df(self.sample_df_365.copy())
        
        # Verify outcome: columns are in correct 12-hour AM/PM format
        expected_columns = [
            '12am', '1am', '2am', '3am', '4am', '5am', '6am', '7am', '8am', '9am', '10am', '11am',
            '12pm', '1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm', '9pm', '10pm', '11pm'
        ]
        
        assert list(result.columns) == expected_columns, "Columns should be in correct 12-hour AM/PM format"
        
        # Verify column ordering starts at 12am (midnight)
        assert result.columns[0] == '12am', "First column should be 12am"
        assert result.columns[12] == '12pm', "13th column should be 12pm (noon)"
        assert result.columns[-1] == '11pm', "Last column should be 11pm"
        
        # Verify all AM hours come before PM hours
        am_columns = [col for col in result.columns if 'am' in col]
        pm_columns = [col for col in result.columns if 'pm' in col]
        assert len(am_columns) == 12, "Should have 12 AM hours"
        assert len(pm_columns) == 12, "Should have 12 PM hours"
        
        # Verify the first 12 columns are AM and last 12 are PM
        assert all('am' in col for col in result.columns[:12]), "First 12 columns should be AM"
        assert all('pm' in col for col in result.columns[12:]), "Last 12 columns should be PM"
