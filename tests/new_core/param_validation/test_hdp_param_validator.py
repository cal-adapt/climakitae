"""
Unit tests for climakitae/new_core/param_validation/hdp_param_validator.py

This module contains comprehensive unit tests for the HDPValidator class
that provides parameter validation for historical data platform queries.
"""

import warnings
from unittest.mock import MagicMock, patch

from climakitae.core.constants import UNSET
from climakitae.new_core.param_validation.hdp_param_validator import (
    HDPValidator,
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


class TestHDPValidatorInit:
    """Test class for HDPValidator initialization."""

    def test_init_successful(self):
        """Test successful initialization of HDPValidator.

        Tests that the validator initializes correctly with a mock catalog
        and sets up the expected attributes and catalog keys.
        """
        # Create mock DataCatalog
        mock_data_catalog = MagicMock()
        mock_hdp_catalog = MagicMock()
        mock_data_catalog.hdp = mock_hdp_catalog

        # Initialize validator
        validator = HDPValidator(mock_data_catalog)

        # Verify initialization
        assert validator.catalog is mock_hdp_catalog
        assert hasattr(validator, "all_catalog_keys")

        # Verify all expected catalog keys are set to UNSET
        expected_keys = {
            "network_id",
            "station_id",
        }
        assert set(validator.all_catalog_keys.keys()) == expected_keys

        # Verify all values are set to UNSET
        for value in validator.all_catalog_keys.values():
            assert value is UNSET


class TestHDPValidatorValidation:
    """Test class for HDPValidator validation methods."""

    def test_is_valid_query_calls_parent_method(self):
        """Test that is_valid_query calls the parent _is_valid_query method.

        Tests that the is_valid_query method properly delegates to the parent
        class method and returns the expected result.
        """
        from climakitae.new_core.param_validation.abc_param_validation import (
            ParameterValidator,
        )

        # Create mock DataCatalog
        mock_data_catalog = MagicMock()
        mock_hdp_catalog = MagicMock()
        mock_data_catalog.hdp = mock_hdp_catalog

        # Initialize validator
        validator = HDPValidator(mock_data_catalog)

        # Mock the parent class method
        with patch.object(ParameterValidator, "_is_valid_query") as mock_parent_method:
            mock_parent_method.return_value = {"result": "test"}

            # Test query (must include network_id which is required)
            test_query = {"network_id": "TEST", "variable_id": "test_variable"}
            result = validator.is_valid_query(test_query)

            # Verify parent method was called with correct arguments
            mock_parent_method.assert_called_once_with(test_query)
            assert result == {"result": "test"}


class TestHDPValidatorRegistration:
    """Test class for HDPValidator registration."""

    def test_validator_registration_decorator(self):
        """Test that the register_catalog_validator decorator works correctly.

        Tests that the decorator properly registers the validator class
        when it is applied, simulating the registration process.
        """
        from climakitae.core.constants import CATALOG_HDP
        from climakitae.new_core.param_validation.abc_param_validation import (
            register_catalog_validator,
        )

        # Create a mock registry to test registration in isolation
        mock_registry = {}

        # Test the decorator functionality
        with patch(
            "climakitae.new_core.param_validation.abc_param_validation._CATALOG_VALIDATOR_REGISTRY",
            mock_registry,
        ):
            # Apply the decorator to a test class
            @register_catalog_validator(CATALOG_HDP)
            class TestValidator:
                pass

            # Verify the registration worked
            assert CATALOG_HDP in mock_registry
            assert mock_registry[CATALOG_HDP] is TestValidator

        # Also verify that HDPValidator is properly designed to be a validator
        mock_data_catalog = MagicMock()
        mock_hdp_catalog = MagicMock()
        mock_data_catalog.hdp = mock_hdp_catalog

        # Should be able to instantiate the validator
        validator = HDPValidator(mock_data_catalog)
        assert validator is not None
        assert validator.catalog is mock_hdp_catalog

    def test_is_valid_query_with_none_return(self):
        """Test is_valid_query when parent method returns None.

        Tests that the is_valid_query method properly handles cases where
        the parent _is_valid_query method returns None.
        """
        from climakitae.new_core.param_validation.abc_param_validation import (
            ParameterValidator,
        )

        # Create mock DataCatalog
        mock_data_catalog = MagicMock()
        mock_hdp_catalog = MagicMock()
        mock_data_catalog.hdp = mock_hdp_catalog

        # Initialize validator
        validator = HDPValidator(mock_data_catalog)

        # Mock the parent class method to return None
        with patch.object(ParameterValidator, "_is_valid_query") as mock_parent_method:
            mock_parent_method.return_value = None

            # Test query with network_id (to pass initial check)
            test_query = {"network_id": "TEST", "invalid_key": "test_value"}
            result = validator.is_valid_query(test_query)

            # Verify parent method was called and None was returned
            mock_parent_method.assert_called_once_with(test_query)
            assert result is None

class TestHDPValidatorNetworkIdRequirement:
    """Test class for network_id requirement validation."""

    def test_query_without_network_id_fails(self):
        """Test that query without network_id fails validation."""
        # Create mock DataCatalog
        mock_data_catalog = MagicMock()
        mock_hdp_catalog = MagicMock()
        mock_data_catalog.hdp = mock_hdp_catalog

        # Initialize validator
        validator = HDPValidator(mock_data_catalog)

        # Test query without network_id
        test_query = {"station_id": "TEST_1"}
        result = validator.is_valid_query(test_query)

        # Should return None (validation failed)
        assert result is None

    def test_query_with_network_id_string_passes_initial_check(self):
        """Test that query with network_id as string passes initial validation check."""
        from climakitae.new_core.param_validation.abc_param_validation import (
            ParameterValidator,
        )

        # Create mock DataCatalog
        mock_data_catalog = MagicMock()
        mock_hdp_catalog = MagicMock()
        mock_data_catalog.hdp = mock_hdp_catalog

        # Initialize validator
        validator = HDPValidator(mock_data_catalog)

        # Mock the parent class method
        with patch.object(ParameterValidator, "_is_valid_query") as mock_parent_method:
            mock_parent_method.return_value = {"network_id": "TEST"}

            # Test query with network_id as string
            test_query = {"network_id": "TEST"}
            result = validator.is_valid_query(test_query)

            # Should pass initial check and call parent method
            mock_parent_method.assert_called_once()
            assert result == {"network_id": "TEST"}

    def test_query_with_single_item_network_id_list_converted(self):
        """Test that single-item network_id list is converted to string."""
        from climakitae.new_core.param_validation.abc_param_validation import (
            ParameterValidator,
        )

        # Create mock DataCatalog
        mock_data_catalog = MagicMock()
        mock_hdp_catalog = MagicMock()
        mock_data_catalog.hdp = mock_hdp_catalog

        # Initialize validator
        validator = HDPValidator(mock_data_catalog)

        # Mock the parent class method
        with patch.object(ParameterValidator, "_is_valid_query") as mock_parent_method:
            mock_parent_method.return_value = {"network_id": "TEST"}

            # Test query with single-item list
            test_query = {"network_id": ["TEST"]}
            result = validator.is_valid_query(test_query)

            # Should convert to string and pass
            assert test_query["network_id"] == "TEST"  # Modified in place
            mock_parent_method.assert_called_once()

    def test_query_with_multiple_network_ids_fails(self):
        """Test that query with multiple network_ids fails validation."""
        # Create mock DataCatalog
        mock_data_catalog = MagicMock()
        mock_hdp_catalog = MagicMock()
        mock_data_catalog.hdp = mock_hdp_catalog

        # Initialize validator
        validator = HDPValidator(mock_data_catalog)

        # Test query with multiple network_ids
        test_query = {"network_id": ["TEST1", "TEST2"]}
        result = validator.is_valid_query(test_query)

        # Should return None (validation failed)
        assert result is None

    def test_query_with_empty_network_id_list_fails(self):
        """Test that query with empty network_id list fails validation."""
        # Create mock DataCatalog
        mock_data_catalog = MagicMock()
        mock_hdp_catalog = MagicMock()
        mock_data_catalog.hdp = mock_hdp_catalog

        # Initialize validator
        validator = HDPValidator(mock_data_catalog)

        # Test query with empty list
        test_query = {"network_id": []}
        result = validator.is_valid_query(test_query)

        # Should return None (validation failed)
        assert result is None

class TestHDPValidatorDefaultProcessors:
    """Test class for default processors."""

    def test_default_processors_set_correctly(self):
        """Test that default processors are set correctly.

        Tests that get_default_processors returns the correct defaults for HDP,
        including both universal defaults from parent class and HDP-specific defaults.
        """
        # Create mock DataCatalog
        mock_data_catalog = MagicMock()
        mock_hdp_catalog = MagicMock()
        mock_data_catalog.hdp = mock_hdp_catalog

        # Initialize validator
        validator = HDPValidator(mock_data_catalog)

        # Get default processors
        defaults = validator.get_default_processors({})

        # Check universal defaults from parent class
        assert defaults["update_attributes"] is UNSET

        # Check HDP-specific defaults
        assert defaults["concat"] == "station_id"

    def test_default_processors_with_query_parameters(self):
        """Test that default processors work with query parameters.

        Tests that query parameters don't affect the HDP default processors
        (unlike CADCAT where experiment_id affects concat dimension).
        """
        # Create mock DataCatalog
        mock_data_catalog = MagicMock()
        mock_hdp_catalog = MagicMock()
        mock_data_catalog.hdp = mock_hdp_catalog

        # Initialize validator
        validator = HDPValidator(mock_data_catalog)

        # Get default processors with a query
        query = {"network_id": "ASOSAWOS", "station_id": "ASOSAWOS_1"}
        defaults = validator.get_default_processors(query)

        # Defaults should remain the same regardless of query parameters
        assert defaults["update_attributes"] is UNSET
        assert defaults["concat"] == "station_id"


class TestHDPValidatorStationIdValidation:
    """Test class for station_id validation."""

    def test_query_with_invalid_station_id_fails(self):
        """Test that query with invalid station_id fails validation."""
        import pandas as pd

        # Create mock DataCatalog
        mock_data_catalog = MagicMock()
        mock_hdp_catalog = MagicMock()
        mock_data_catalog.hdp = mock_hdp_catalog

        # Mock catalog search to return only 2 out of 3 requested stations
        mock_search_result = MagicMock()
        mock_search_result.df = pd.DataFrame(
            {
                "station_id": ["ASOSAWOS_1", "ASOSAWOS_2"],
                "network_id": ["ASOSAWOS", "ASOSAWOS"],
            }
        )
        mock_hdp_catalog.search.return_value = mock_search_result

        # Initialize validator
        validator = HDPValidator(mock_data_catalog)

        # Test query with 3 stations, but only 2 exist
        test_query = {
            "network_id": "ASOSAWOS",
            "station_id": ["ASOSAWOS_1", "ASOSAWOS_2", "nicole"],
        }
        result = validator.is_valid_query(test_query)

        # Should return None (validation failed)
        assert result is None

    def test_query_with_all_valid_station_ids_passes(self):
        """Test that query with all valid station_ids passes."""
        import pandas as pd
        from climakitae.new_core.param_validation.abc_param_validation import (
            ParameterValidator,
        )

        # Create mock DataCatalog
        mock_data_catalog = MagicMock()
        mock_hdp_catalog = MagicMock()
        mock_data_catalog.hdp = mock_hdp_catalog

        # Mock catalog search to return all requested stations
        mock_search_result = MagicMock()
        mock_search_result.df = pd.DataFrame(
            {
                "station_id": ["ASOSAWOS_1", "ASOSAWOS_2"],
                "network_id": ["ASOSAWOS", "ASOSAWOS"],
            }
        )
        mock_hdp_catalog.search.return_value = mock_search_result

        # Initialize validator
        validator = HDPValidator(mock_data_catalog)

        # Mock the parent class method
        with patch.object(ParameterValidator, "_is_valid_query") as mock_parent_method:
            mock_parent_method.return_value = {
                "network_id": "ASOSAWOS",
                "station_id": ["ASOSAWOS_1", "ASOSAWOS_2"],
            }

            # Test query with valid stations
            test_query = {
                "network_id": "ASOSAWOS",
                "station_id": ["ASOSAWOS_1", "ASOSAWOS_2"],
            }
            result = validator.is_valid_query(test_query)

            # Should pass validation
            assert result is not None
            mock_parent_method.assert_called_once()

    def test_query_without_station_id_passes(self):
        """Test that query without station_id passes (station_id is optional)."""
        from climakitae.new_core.param_validation.abc_param_validation import (
            ParameterValidator,
        )

        # Create mock DataCatalog
        mock_data_catalog = MagicMock()
        mock_hdp_catalog = MagicMock()
        mock_data_catalog.hdp = mock_hdp_catalog

        # Initialize validator
        validator = HDPValidator(mock_data_catalog)

        # Mock the parent class method
        with patch.object(ParameterValidator, "_is_valid_query") as mock_parent_method:
            mock_parent_method.return_value = {"network_id": "ASOSAWOS"}

            # Test query without station_id
            test_query = {"network_id": "ASOSAWOS"}
            result = validator.is_valid_query(test_query)

            # Should pass validation (station_id is optional)
            assert result is not None
            mock_parent_method.assert_called_once()

    def test_query_with_single_invalid_station_id_string_fails(self):
        """Test that query with single invalid station_id string fails."""
        import pandas as pd

        # Create mock DataCatalog
        mock_data_catalog = MagicMock()
        mock_hdp_catalog = MagicMock()
        mock_data_catalog.hdp = mock_hdp_catalog

        # Mock catalog search to return empty result
        mock_search_result = MagicMock()
        mock_search_result.df = pd.DataFrame(
            {
                "station_id": [],
                "network_id": [],
            }
        )
        mock_hdp_catalog.search.return_value = mock_search_result

        # Initialize validator
        validator = HDPValidator(mock_data_catalog)

        # Test query with single invalid station (as string)
        test_query = {"network_id": "ASOSAWOS", "station_id": "invalid_station"}
        result = validator.is_valid_query(test_query)

        # Should return None (validation failed)
        assert result is None
