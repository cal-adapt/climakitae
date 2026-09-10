"""
Unit tests for climakitae/new_core/param_validation/data_param_validator.py

This module contains comprehensive unit tests for the Sup3rCCValidator class
that validates data catalog parameters for CADCAT catalog.
"""

import warnings
from unittest.mock import MagicMock, patch

from climakitae.core.constants import UNSET
from climakitae.new_core.param_validation.sup3rcc_param_validator import (
    Sup3rCCValidator,
)

# Suppress known external warnings that are not relevant to our tests
warnings.filterwarnings(
    "ignore",
    message="The 'shapely.geos' module is deprecated",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated", category=DeprecationWarning
)


class TestSup3rCCValidator:
    """Test class for Sup3rCCValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock DataCatalog
        mock_catalog = MagicMock()
        mock_sup3r_catalog = MagicMock()
        mock_catalog.sup3rcc = mock_sup3r_catalog

        # Create validator instance
        self.mock_catalog = mock_catalog
        self.validator = Sup3rCCValidator(mock_catalog)

    def test_init_successful(self):
        """Test successful initialization of Sup3rCCValidator.

        Tests that Sup3rCCValidator initializes correctly with proper
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
        assert self.validator.catalog == self.mock_catalog.sup3rcc

    def test_is_valid_query_calls_parent_when_checks_pass(self):
        """Test is_valid_query calls parent method when initial checks pass.

        Tests that is_valid_query calls the parent _is_valid_query method
        when all initial checks pass.
        """
        # Include all required keys: activity_id, institution_id, table_id, grid_label, variable_id
        query = {
            "activity_id": "Sup3r",
            "institution_id": "NLR",
            "table_id": "1hr",
            "grid_label": "conus4km",
            "variable_id": "t2",
        }  # No localize processor, should pass initial checks
        expected_result = query.copy()

        # Mock the parent class _is_valid_query method
        with patch(
            "climakitae.new_core.param_validation.abc_param_validation.ParameterValidator._is_valid_query",
            return_value=expected_result,
        ) as mock_parent:
            result = self.validator.is_valid_query(query)

            # Should call parent method once and return its result
            mock_parent.assert_called_once_with(query)
            assert result == expected_result

    def test_default_processors_with_empty_query(self):
        """Test get_default_processors with empty query.

        Tests that get_default_processors returns the correct defaults for CADCAT
        with no experiment_id specified.
        """
        query = {}
        defaults = self.validator.get_default_processors(query)

        # Check universal defaults from parent class
        assert defaults["update_attributes"] is UNSET

        # Check catalog-specific defaults
        assert defaults["drop_leap_days"] == "yes"
        assert defaults["concat"] == "time"  # Default when no experiment_id

    def test_default_processors_with_historical_experiment_id(self):
        """Test get_default_processors with historical experiment_id.

        Tests that concat dimension is set to 'sim' for historical data.
        """
        query = {"experiment_id": "historical"}
        defaults = self.validator.get_default_processors(query)

        # Check universal defaults
        assert defaults["update_attributes"] is UNSET

        # Check catalog-specific defaults
        assert defaults["drop_leap_days"] == "yes"
        assert defaults["concat"] == "sim"  # Changes to sim for historical

    def test_default_processors_with_ssp_experiment_id(self):
        """Test get_default_processors with SSP experiment_id.

        Tests that concat dimension remains 'time' for SSP scenarios.
        """
        query = {"experiment_id": "ssp245"}
        defaults = self.validator.get_default_processors(query)

        # Check universal defaults
        assert defaults["update_attributes"] is UNSET

        # Check catalog-specific defaults
        assert defaults["drop_leap_days"] == "yes"
        assert defaults["concat"] == "time"  # Stays time for SSP

    def test_default_processors_with_ssp_concat_to_historical(self):
        """Test get_default_processors to confirm historical is appended to ssp.

        Tests that concat is 'time' when experiment_id includes SSP scenarios.
        """
        query = {"experiment_id": ["historical", "ssp245"]}
        defaults = self.validator.get_default_processors(query)

        # Check universal defaults
        assert defaults["update_attributes"] is UNSET

        # Check catalog-specific defaults
        assert defaults["drop_leap_days"] == "yes"
        assert defaults["concat"] == "time"  # Stays time for SSP
