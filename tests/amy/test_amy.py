"""
Unit tests for climakitae/explore/amy.py

This module contains comprehensive unit tests for the AMY (Average Meteorological Year)
and climate profile computation functions that provide climate profile analysis.
"""

import pytest
import xarray as xr
from unittest.mock import MagicMock, patch

from climakitae.explore.amy import retrieve_profile_data


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
        self.mock_get_data_patcher = patch('climakitae.explore.amy.get_data')
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
