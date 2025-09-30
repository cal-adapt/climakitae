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
    set_profile_metadata
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
        self.mock_retrieve_patcher = patch('climakitae.explore.amy.retrieve_profile_data')
        self.mock_compute_patcher = patch('climakitae.explore.amy.compute_profile')
        self.mock_print_patcher = patch('builtins.print')
        
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
        mock_historic_data.data_vars = {'tasmax': MagicMock()}
        mock_future_data = MagicMock(spec=xr.Dataset)
        mock_future_data.data_vars = {'tasmax': MagicMock()}
        
        self.mock_retrieve_profile_data.return_value = (mock_historic_data, mock_future_data)
        
        # Create mock profile DataFrames
        mock_future_profile = pd.DataFrame(np.random.rand(365, 24))
        mock_historic_profile = pd.DataFrame(np.random.rand(365, 24))
        self.mock_compute_profile.side_effect = [mock_future_profile, mock_historic_profile]
        
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
        mock_future_data.data_vars = {'tasmax': MagicMock()}
        
        # When no_delta=True, only future data is returned, historic is None
        self.mock_retrieve_profile_data.return_value = (None, mock_future_data)
        
        # Create mock future profile
        mock_future_profile = pd.DataFrame(np.random.rand(365, 24))
        self.mock_compute_profile.return_value = mock_future_profile
        
        # Execute function with no_delta=True
        result = get_climate_profile(warming_level=[2.0], no_delta=True)
        
        # Verify outcome: returns the raw future profile (no difference calculation)
        assert isinstance(result, pd.DataFrame), "Should return a pandas DataFrame"
        assert result.shape == mock_future_profile.shape, "Should return the original future profile shape"
        # Verify compute_profile was called only once (for future data only)
        assert self.mock_compute_profile.call_count == 1, "Should call compute_profile only once for future data"

    def test_get_climate_profile_raises_error_when_no_data_returned(self):
        """Test that get_climate_profile raises ValueError when no data is retrieved."""
        # Setup scenario where both datasets are None
        self.mock_retrieve_profile_data.return_value = (None, None)
        
        # Execute and verify outcome: should raise ValueError
        with pytest.raises(ValueError, match="No data returned for either historical or future datasets"):
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
        time_delta = pd.date_range('2020-01-01', periods=8760, freq='h')  # 1 year of hourly data
        warming_levels = [1.5]
        simulations = ['sim1']
        
        # Create test data with proper dimensions
        data = np.random.rand(len(warming_levels), len(time_delta), len(simulations))
        
        self.sample_data = xr.DataArray(
            data,
            dims=['warming_level', 'time_delta', 'simulation'],
            coords={
                'warming_level': warming_levels,
                'time_delta': time_delta,
                'simulation': simulations
            },
            attrs={'units': 'degC', 'variable_id': 'tasmax'}
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
        assert result.index.dtype == int or isinstance(result.index, pd.Index), "Should have proper index"
        assert hasattr(result, 'attrs'), "Should preserve metadata in attrs"

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
        assert 'units' in result.attrs, "Should preserve units from input data"
        assert 'quantile' in result.attrs, "Should include quantile information"
        assert 'method' in result.attrs, "Should include method description"
        assert result.attrs['quantile'] == 0.75, "Should record the correct quantile used"
        assert result.attrs['units'] == 'degC', "Should preserve original units"


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
            np.random.rand(365, 24),
            index=range(1, 366),
            columns=range(1, 25)
        )
        # Add metadata attributes
        self.sample_profile.attrs = {
            'units': 'degF',
            'variable_id': 'tasmax',
            'quantile': 0.5,
            'method': '8760 analysis',
            'description': 'Test climate profile'
        }

    def test_get_profile_units_returns_correct_units(self):
        """Test that get_profile_units extracts units from DataFrame metadata."""
        # Execute function
        units = get_profile_units(self.sample_profile)
        
        # Verify outcome: returns the correct units
        assert units == 'degF', "Should return the units from DataFrame metadata"

    def test_get_profile_units_returns_unknown_when_missing(self):
        """Test that get_profile_units returns 'Unknown' when units are not present."""
        # Create DataFrame without units
        profile_no_units = pd.DataFrame(np.random.rand(10, 5))
        
        # Execute function
        units = get_profile_units(profile_no_units)
        
        # Verify outcome: returns 'Unknown' for missing units
        assert units == 'Unknown', "Should return 'Unknown' when units metadata is missing"

    def test_get_profile_metadata_returns_all_attributes(self):
        """Test that get_profile_metadata returns complete metadata dictionary."""
        # Execute function
        metadata = get_profile_metadata(self.sample_profile)
        
        # Verify outcome: returns dictionary with all metadata
        assert isinstance(metadata, dict), "Should return a dictionary"
        assert metadata['units'] == 'degF', "Should include units"
        assert metadata['variable_id'] == 'tasmax', "Should include variable_id"
        assert metadata['quantile'] == 0.5, "Should include quantile"
        assert metadata['method'] == '8760 analysis', "Should include method"
        assert len(metadata) == 5, "Should return all metadata attributes"

    def test_set_profile_metadata_updates_dataframe_attrs(self):
        """Test that set_profile_metadata properly updates DataFrame attributes."""
        # Setup new metadata to add
        new_metadata = {
            'author': 'Test User',
            'created_date': '2023-01-01',
            'notes': 'Test profile data'
        }
        
        # Execute function
        set_profile_metadata(self.sample_profile, new_metadata)
        
        # Verify outcome: DataFrame attrs are updated
        assert 'author' in self.sample_profile.attrs, "Should add new author attribute"
        assert 'created_date' in self.sample_profile.attrs, "Should add new created_date attribute"  
        assert 'notes' in self.sample_profile.attrs, "Should add new notes attribute"
        assert self.sample_profile.attrs['author'] == 'Test User', "Should set correct author value"
        
        # Original attributes should still exist
        assert self.sample_profile.attrs['units'] == 'degF', "Should preserve original units"

    def test_set_profile_metadata_raises_error_for_non_dict_input(self):
        """Test that set_profile_metadata raises ValueError for non-dictionary input."""
        # Execute and verify outcome: should raise ValueError for non-dict input
        with pytest.raises(ValueError, match="Metadata must be provided as a dictionary"):
            set_profile_metadata(self.sample_profile, "not_a_dict")
