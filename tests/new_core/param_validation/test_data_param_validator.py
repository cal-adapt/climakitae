"""
Unit tests for climakitae/new_core/param_validation/data_param_validator.py

This module contains comprehensive unit tests for the DataValidator class
that validates data catalog parameters for CADCAT catalog.
"""

import warnings
from unittest.mock import MagicMock, patch

from climakitae.core.constants import UNSET
from climakitae.new_core.param_validation.data_param_validator import DataValidator

# Suppress known external warnings that are not relevant to our tests
warnings.filterwarnings(
    "ignore",
    message="The 'shapely.geos' module is deprecated",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated", category=DeprecationWarning
)


class TestDataValidator:
    """Test class for DataValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock DataCatalog
        self.mock_catalog = MagicMock()
        self.mock_catalog.data = MagicMock()
        
        # Create validator instance
        self.validator = DataValidator(self.mock_catalog)

    def test_init_successful(self):
        """Test successful initialization of DataValidator.
        
        Tests that DataValidator initializes correctly with proper
        all_catalog_keys and catalog assignment.
        """
        expected_keys = {
            "activity_id": UNSET,
            "institution_id": UNSET,
            "source_id": UNSET,
            "experiment_id": UNSET,
            "table_id": UNSET,
            "grid_label": UNSET,
            "variable_id": UNSET,
        }
        
        assert self.validator.all_catalog_keys == expected_keys
        assert self.validator.catalog == self.mock_catalog.data

    def test_is_valid_query_with_localize_check_failure(self):
        """Test is_valid_query when initial checks fail.
        
        Tests that is_valid_query returns None when
        _check_query_for_wrf_and_localize returns False.
        """
        query = {
            "processes": {"localize": {}},
            "activity_id": "LOCA2",  # Not WRF, should fail
            "variable_id": "tas"
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.validator.is_valid_query(query)
            
            assert result is None
            assert len(w) == 1
            assert "Localize processor is not supported for LOCA2 datasets" in str(w[0].message)