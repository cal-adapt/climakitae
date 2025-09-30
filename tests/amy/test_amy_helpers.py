"""
Unit tests for climakitae/explore/amy.py helper functions

This module contains comprehensive unit tests for the helper functions
in the amy module that support climate profile analysis.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from climakitae.explore.amy import (
    _compute_difference_profile,
    _stack_profile_data,
    get_profile_units,
    get_profile_metadata,
    set_profile_metadata,
    _format_meteo_yr_df,
)


class TestProfileUtilityFunctions:
    """Test class for profile utility functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample profile DataFrame with metadata
        self.sample_profile = pd.DataFrame(
            np.random.rand(365, 24),
            columns=np.arange(1, 25),
            index=pd.Index(
                [f"Day-{i:03d}" for i in range(1, 366)],
                name="Day of Year"
            ),
        )
        self.sample_profile.attrs = {
            "units": "degF",
            "variable_name": "tasmax",
            "method": "8760 analysis",
            "quantile": 0.5,
            "source": "CMIP6",
        }

    def test_get_profile_units_with_units(self):
        """Test get_profile_units when units are present."""
        units = get_profile_units(self.sample_profile)
        assert units == "degF"

    def test_get_profile_units_without_units(self):
        """Test get_profile_units when units are missing."""
        profile_no_units = self.sample_profile.copy()
        del profile_no_units.attrs["units"]
        
        units = get_profile_units(profile_no_units)
        assert units == "Unknown"

    def test_get_profile_units_empty_attrs(self):
        """Test get_profile_units with empty attributes."""
        profile_empty = pd.DataFrame(np.random.rand(10, 5))
        profile_empty.attrs = {}
        
        units = get_profile_units(profile_empty)
        assert units == "Unknown"

    def test_get_profile_metadata_complete(self):
        """Test get_profile_metadata with complete metadata."""
        metadata = get_profile_metadata(self.sample_profile)
        
        assert isinstance(metadata, dict)
        assert metadata["units"] == "degF"
        assert metadata["variable_name"] == "tasmax"
        assert metadata["method"] == "8760 analysis"
        assert metadata["quantile"] == 0.5
        assert metadata["source"] == "CMIP6"

    def test_get_profile_metadata_empty_attrs(self):
        """Test get_profile_metadata with empty attributes."""
        profile_empty = pd.DataFrame(np.random.rand(10, 5))
        profile_empty.attrs = {}
        
        metadata = get_profile_metadata(profile_empty)
        assert isinstance(metadata, dict)
        assert len(metadata) == 0

    def test_set_profile_metadata_valid_dict(self):
        """Test set_profile_metadata with valid dictionary."""
        new_metadata = {
            "author": "Jane Doe",
            "created_date": "2024-01-01",
            "notes": "Test profile for unit testing"
        }
        
        original_keys = set(self.sample_profile.attrs.keys())
        set_profile_metadata(self.sample_profile, new_metadata)
        
        # Check that new metadata was added
        assert self.sample_profile.attrs["author"] == "Jane Doe"
        assert self.sample_profile.attrs["created_date"] == "2024-01-01"
        assert self.sample_profile.attrs["notes"] == "Test profile for unit testing"
        
        # Check that original metadata is still present
        for key in original_keys:
            assert key in self.sample_profile.attrs

    def test_set_profile_metadata_overwrite_existing(self):
        """Test set_profile_metadata overwrites existing values."""
        new_metadata = {"units": "degC", "source": "ERA5"}
        
        set_profile_metadata(self.sample_profile, new_metadata)
        
        assert self.sample_profile.attrs["units"] == "degC"
        assert self.sample_profile.attrs["source"] == "ERA5"
        # Other attributes should remain
        assert self.sample_profile.attrs["variable_name"] == "tasmax"

    def test_set_profile_metadata_empty_dict(self):
        """Test set_profile_metadata with empty dictionary."""
        original_attrs = dict(self.sample_profile.attrs)
        
        set_profile_metadata(self.sample_profile, {})
        
        # Attributes should remain unchanged
        assert dict(self.sample_profile.attrs) == original_attrs

    def test_set_profile_metadata_invalid_input(self):
        """Test set_profile_metadata with invalid input type."""
        with pytest.raises(ValueError, match="Metadata must be provided as a dictionary"):
            set_profile_metadata(self.sample_profile, "not_a_dict")  # type: ignore
        
        with pytest.raises(ValueError, match="Metadata must be provided as a dictionary"):
            set_profile_metadata(self.sample_profile, 123)  # type: ignore
        
        with pytest.raises(ValueError, match="Metadata must be provided as a dictionary"):
            set_profile_metadata(self.sample_profile, None)  # type: ignore


# Removed TestFetchPrimaryDataVariable since _fetch_primary_data_variable 
# is a nested function inside get_climate_profile and not directly testable


class TestFormatMeteoYrDataFrame:
    """Test class for _format_meteo_yr_df function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample DataFrame with 24 hour columns (mimicking compute_amy output)
        self.sample_df_365 = pd.DataFrame(
            np.random.rand(365, 24),
            columns=np.arange(1, 25),
            index=np.arange(1, 366),
        )
        
        self.sample_df_366 = pd.DataFrame(
            np.random.rand(366, 24),
            columns=np.arange(1, 25),
            index=np.arange(1, 367),
        )

    def test_format_meteo_yr_df_365_days(self):
        """Test _format_meteo_yr_df with 365 days (non-leap year)."""
        result = _format_meteo_yr_df(self.sample_df_365)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (365, 24)
        assert result.index.name == "Day of Year"
        assert result.columns.name == "Hour"
        
        # Check that columns are formatted as expected (12am, 1am, ..., 11pm)
        expected_columns = ['12am', '1am', '2am', '3am', '4am', '5am', '6am', '7am', 
                           '8am', '9am', '10am', '11am', '12pm', '1pm', '2pm', '3pm', 
                           '4pm', '5pm', '6pm', '7pm', '8pm', '9pm', '10pm', '11pm']
        
        assert list(result.columns) == expected_columns

    def test_format_meteo_yr_df_366_days(self):
        """Test _format_meteo_yr_df with 366 days (leap year)."""
        result = _format_meteo_yr_df(self.sample_df_366)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (366, 24)
        assert result.index.name == "Day of Year"
        assert result.columns.name == "Hour"

    def test_format_meteo_yr_df_date_formatting(self):
        """Test that _format_meteo_yr_df properly formats dates."""
        result = _format_meteo_yr_df(self.sample_df_365)
        
        # Check that index contains month-day format
        assert all("-" in str(idx) for idx in result.index)
        
        # Check first few entries are January dates
        first_entries = result.index[:5]
        assert all(str(idx).startswith("Jan") for idx in first_entries)

    def test_format_meteo_yr_df_empty_dataframe(self):
        """Test _format_meteo_yr_df with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            _format_meteo_yr_df(empty_df)

    def test_format_meteo_yr_df_wrong_columns(self):
        """Test _format_meteo_yr_df with incorrect number of columns."""
        wrong_df = pd.DataFrame(
            np.random.rand(365, 12),  # Only 12 columns instead of 24
            columns=np.arange(1, 13),
            index=np.arange(1, 366),
        )
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((IndexError, KeyError)):
            _format_meteo_yr_df(wrong_df)


class TestComputeDifferenceProfile:
    """Test class for _compute_difference_profile function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample profiles for testing difference calculations
        self.simple_future = pd.DataFrame(
            np.random.rand(365, 24) + 20,  # Future temps around 20°C
            columns=np.arange(1, 25),
            index=pd.Index([f"Day-{i:03d}" for i in range(1, 366)], name="Day of Year"),
        )
        
        self.simple_historic = pd.DataFrame(
            np.random.rand(365, 24) + 15,  # Historic temps around 15°C
            columns=np.arange(1, 25),
            index=self.simple_future.index,
        )
        
        # Create MultiIndex profiles for more complex testing
        hours = list(range(1, 25))
        warming_levels = ["WL_1.5", "WL_2.0", "WL_3.0"]
        simulations = ["sim1", "sim2", "sim3"]
        
        # Future with MultiIndex: (Hour, Warming_Level)
        multi_cols_wl = pd.MultiIndex.from_product(
            [hours, warming_levels], names=["Hour", "Warming_Level"]
        )
        self.multi_future_wl = pd.DataFrame(
            np.random.rand(365, len(multi_cols_wl)) + 20,
            columns=multi_cols_wl,
            index=self.simple_future.index,
        )
        
        # Future with MultiIndex: (Hour, Simulation)
        multi_cols_sim = pd.MultiIndex.from_product(
            [hours, simulations], names=["Hour", "Simulation"]
        )
        self.multi_future_sim = pd.DataFrame(
            np.random.rand(365, len(multi_cols_sim)) + 20,
            columns=multi_cols_sim,
            index=self.simple_future.index,
        )
        
        # Historic with MultiIndex: (Hour, Simulation)
        self.multi_historic_sim = pd.DataFrame(
            np.random.rand(365, len(multi_cols_sim)) + 15,
            columns=multi_cols_sim,
            index=self.simple_future.index,
        )

    def test_compute_difference_profile_simple_columns(self):
        """Test _compute_difference_profile with simple column structure."""
        result = _compute_difference_profile(self.simple_future, self.simple_historic)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == self.simple_future.shape
        
        # Check that result is approximately future - historic
        expected_diff = self.simple_future - self.simple_historic
        pd.testing.assert_frame_equal(result, expected_diff)

    @patch("climakitae.explore.amy._compute_mixed_index_difference")
    def test_compute_difference_profile_mixed_index(self, mock_mixed_diff):
        """Test _compute_difference_profile with mixed index types."""
        mock_mixed_diff.return_value = self.simple_future.copy()
        
        result = _compute_difference_profile(self.multi_future_wl, self.simple_historic)
        
        # Should call mixed index difference function
        mock_mixed_diff.assert_called_once_with(self.multi_future_wl, self.simple_historic)
        assert isinstance(result, pd.DataFrame)

    @patch("climakitae.explore.amy._compute_multiindex_difference")
    def test_compute_difference_profile_both_multiindex(self, mock_multi_diff):
        """Test _compute_difference_profile with both profiles having MultiIndex."""
        mock_multi_diff.return_value = self.multi_future_sim.copy()
        
        result = _compute_difference_profile(self.multi_future_sim, self.multi_historic_sim)
        
        # Should call multiindex difference function
        mock_multi_diff.assert_called_once_with(self.multi_future_sim, self.multi_historic_sim)
        assert isinstance(result, pd.DataFrame)

    def test_compute_difference_profile_shape_mismatch(self):
        """Test _compute_difference_profile with mismatched shapes."""
        wrong_shape_historic = pd.DataFrame(
            np.random.rand(300, 24),  # Different number of rows
            columns=np.arange(1, 25),
            index=pd.Index([f"Day-{i:03d}" for i in range(1, 301)], name="Day of Year"),
        )
        
        # Should handle or raise appropriate error
        with pytest.raises((ValueError, KeyError)):
            _compute_difference_profile(self.simple_future, wrong_shape_historic)


class TestStackProfileData:
    """Test class for _stack_profile_data function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample profile data dictionary
        self.profile_data = {
            ("WL_1.5", "sim1"): np.random.rand(365, 24),
            ("WL_1.5", "sim2"): np.random.rand(365, 24),
            ("WL_2.0", "sim1"): np.random.rand(365, 24),
            ("WL_2.0", "sim2"): np.random.rand(365, 24),
        }
        
        self.wl_names = ["WL_1.5", "WL_2.0"]
        self.sim_names = ["sim1", "sim2"]
        self.hours_per_day = 24

    def test_stack_profile_data_hour_first_false(self):
        """Test _stack_profile_data with hour_first=False."""
        result = _stack_profile_data(
            profile_data=self.profile_data,
            hours_per_day=self.hours_per_day,
            wl_names=self.wl_names,
            sim_names=self.sim_names,
            hour_first=False,
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 365  # Number of days
        
        # Should have columns for each wl-sim-hour combination
        expected_cols = len(self.wl_names) * len(self.sim_names) * self.hours_per_day
        assert result.shape[1] == expected_cols

    def test_stack_profile_data_hour_first_true(self):
        """Test _stack_profile_data with hour_first=True."""
        result = _stack_profile_data(
            profile_data=self.profile_data,
            hours_per_day=self.hours_per_day,
            wl_names=self.wl_names,
            sim_names=self.sim_names,
            hour_first=True,
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 365  # Number of days
        
        # Should have columns for each hour-wl-sim combination
        expected_cols = len(self.wl_names) * len(self.sim_names) * self.hours_per_day
        assert result.shape[1] == expected_cols

    def test_stack_profile_data_three_level_true(self):
        """Test _stack_profile_data with three_level=True."""
        result = _stack_profile_data(
            profile_data=self.profile_data,
            hours_per_day=self.hours_per_day,
            wl_names=self.wl_names,
            sim_names=self.sim_names,
            three_level=True,
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 365  # Number of days
        
        # Should have columns for each hour-wl-sim combination
        expected_cols = len(self.wl_names) * len(self.sim_names) * self.hours_per_day
        assert result.shape[1] == expected_cols

    def test_stack_profile_data_empty_data(self):
        """Test _stack_profile_data with empty profile_data."""
        empty_data = {}
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((KeyError, IndexError)):
            _stack_profile_data(
                profile_data=empty_data,
                hours_per_day=self.hours_per_day,
                wl_names=self.wl_names,
                sim_names=self.sim_names,
            )

    def test_stack_profile_data_missing_key(self):
        """Test _stack_profile_data with missing key in profile_data."""
        incomplete_data = {
            ("WL_1.5", "sim1"): np.random.rand(365, 24),
            # Missing other combinations
        }
        
        with pytest.raises(KeyError):
            _stack_profile_data(
                profile_data=incomplete_data,
                hours_per_day=self.hours_per_day,
                wl_names=self.wl_names,
                sim_names=self.sim_names,
            )