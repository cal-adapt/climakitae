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

from climakitae.explore.amy import retrieve_profile_data, get_climate_profile


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
