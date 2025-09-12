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
            "variable_id": "tas",
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.validator.is_valid_query(query)

            assert result is None
            assert len(w) == 1
            assert "Localize processor is not supported for LOCA2 datasets" in str(
                w[0].message
            )

    def test_check_query_for_wrf_and_localize_valid_no_localize(self):
        """Test _check_query_for_wrf_and_localize with no localize processor.

        Tests that queries without the localize processor return True.
        """
        query = {"variable_id": "tas", "activity_id": "LOCA2"}

        result = self.validator._check_query_for_wrf_and_localize(query)

        assert result is True

    def test_check_query_for_wrf_and_localize_valid_with_wrf_and_t2(self):
        """Test _check_query_for_wrf_and_localize with valid localize configuration.

        Tests that queries with localize processor, WRF activity_id, and t2 variable
        return True.
        """
        query = {
            "processes": {"localize": {}},
            "activity_id": "WRF",
            "variable_id": "t2",
        }

        result = self.validator._check_query_for_wrf_and_localize(query)

        assert result is True

    def test_check_query_for_wrf_and_localize_invalid_no_wrf(self):
        """Test _check_query_for_wrf_and_localize with localize but no WRF.

        Tests that queries with localize processor but without WRF activity_id
        return False and emit proper warning.
        """
        query = {
            "processes": {"localize": {}},
            "activity_id": "LOCA2",
            "variable_id": "t2",
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.validator._check_query_for_wrf_and_localize(query)

            assert result is False
            assert len(w) == 1
            assert "Localize processor is not supported for LOCA2 datasets" in str(
                w[0].message
            )
            assert "Please specify '.activity_id(WRF)'" in str(w[0].message)

    def test_check_query_for_wrf_and_localize_invalid_wrong_variable(self):
        """Test _check_query_for_wrf_and_localize with localize and wrong variable.

        Tests that queries with localize processor and WRF but wrong variable_id
        return False and emit proper warning.
        """
        query = {
            "processes": {"localize": {}},
            "activity_id": "WRF",
            "variable_id": "tas",  # Wrong variable, should be t2
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.validator._check_query_for_wrf_and_localize(query)

            assert result is False
            assert len(w) == 1
            assert (
                "Localize processor is not supported for any variable other than 't2'"
                in str(w[0].message)
            )
            assert "Please specify '.variable_id('t2')'" in str(w[0].message)

    def test_is_valid_query_calls_parent_when_checks_pass(self):
        """Test is_valid_query calls parent method when initial checks pass.

        Tests that is_valid_query calls the parent _is_valid_query method
        when all initial checks pass.
        """
        query = {
            "variable_id": "tas"
        }  # No localize processor, should pass initial checks
        expected_result = {"variable_id": "tas"}

        # Mock the parent class _is_valid_query method
        with patch(
            "climakitae.new_core.param_validation.abc_param_validation.ParameterValidator._is_valid_query",
            return_value=expected_result,
        ) as mock_parent:
            result = self.validator.is_valid_query(query)

            # Should call parent method once and return its result
            mock_parent.assert_called_once_with(query)
            assert result == expected_result
