"""
Unit tests for climakitae/new_core/param_validation/renewables_param_validator.py

This module contains comprehensive unit tests for the RenewablesValidator class
that provides parameter validation for renewable energy dataset queries.
"""

import warnings
from unittest.mock import MagicMock

from climakitae.core.constants import UNSET
from climakitae.new_core.param_validation.renewables_param_validator import (
    RenewablesValidator,
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


class TestRenewablesValidatorInit:
    """Test class for RenewablesValidator initialization."""

    def test_init_successful(self):
        """Test successful initialization of RenewablesValidator.

        Tests that the validator initializes correctly with a mock catalog
        and sets up the expected attributes and catalog keys.
        """
        # Create mock DataCatalog
        mock_data_catalog = MagicMock()
        mock_renewables_catalog = MagicMock()
        mock_data_catalog.renewables = mock_renewables_catalog

        # Initialize validator
        validator = RenewablesValidator(mock_data_catalog)

        # Verify initialization
        assert validator.catalog is mock_renewables_catalog
        assert hasattr(validator, "all_catalog_keys")

        # Verify all expected catalog keys are set to UNSET
        expected_keys = {
            "installation",
            "activity_id",
            "institution_id",
            "source_id",
            "experiment_id",
            "table_id",
            "grid_label",
            "variable_id",
        }
        assert set(validator.all_catalog_keys.keys()) == expected_keys

        # Verify all values are set to UNSET
        for value in validator.all_catalog_keys.values():
            assert value is UNSET


class TestRenewablesValidatorValidation:
    """Test class for RenewablesValidator validation methods."""

    def test_is_valid_query_calls_parent_method(self):
        """Test that is_valid_query calls the parent _is_valid_query method.

        Tests that the is_valid_query method properly delegates to the parent
        class method and returns the expected result.
        """
        from unittest.mock import patch
        from climakitae.new_core.param_validation.abc_param_validation import (
            ParameterValidator,
        )

        # Create mock DataCatalog
        mock_data_catalog = MagicMock()
        mock_renewables_catalog = MagicMock()
        mock_data_catalog.renewables = mock_renewables_catalog

        # Initialize validator
        validator = RenewablesValidator(mock_data_catalog)

        # Mock the parent class method
        with patch.object(ParameterValidator, "_is_valid_query") as mock_parent_method:
            mock_parent_method.return_value = {"result": "test"}

            # Test query
            test_query = {"variable_id": "test_variable"}
            result = validator.is_valid_query(test_query)

            # Verify parent method was called with correct arguments
            mock_parent_method.assert_called_once_with(test_query)
            assert result == {"result": "test"}


class TestRenewablesValidatorRegistration:
    """Test class for RenewablesValidator registration."""

    def test_validator_registration(self):
        """Test that RenewablesValidator is properly registered for renewables catalog.

        Tests that the validator class is correctly registered with the
        catalog registration system using the CATALOG_REN_ENERGY_GEN constant.
        """
        from climakitae.core.constants import CATALOG_REN_ENERGY_GEN
        from climakitae.new_core.param_validation.abc_param_validation import (
            _CATALOG_VALIDATOR_REGISTRY,
        )

        # Verify that the renewables validator is registered
        assert CATALOG_REN_ENERGY_GEN in _CATALOG_VALIDATOR_REGISTRY
        assert (
            _CATALOG_VALIDATOR_REGISTRY[CATALOG_REN_ENERGY_GEN] is RenewablesValidator
        )

    def test_is_valid_query_with_none_return(self):
        """Test is_valid_query when parent method returns None.

        Tests that the is_valid_query method properly handles cases where
        the parent _is_valid_query method returns None.
        """
        from unittest.mock import patch
        from climakitae.new_core.param_validation.abc_param_validation import (
            ParameterValidator,
        )

        # Create mock DataCatalog
        mock_data_catalog = MagicMock()
        mock_renewables_catalog = MagicMock()
        mock_data_catalog.renewables = mock_renewables_catalog

        # Initialize validator
        validator = RenewablesValidator(mock_data_catalog)

        # Mock the parent class method to return None
        with patch.object(ParameterValidator, "_is_valid_query") as mock_parent_method:
            mock_parent_method.return_value = None

            # Test query
            test_query = {"invalid_key": "test_value"}
            result = validator.is_valid_query(test_query)

            # Verify parent method was called and None was returned
            mock_parent_method.assert_called_once_with(test_query)
            assert result is None
