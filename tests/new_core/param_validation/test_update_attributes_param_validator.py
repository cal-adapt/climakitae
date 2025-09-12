"""
Unit tests for climakitae/new_core/param_validation/update_attributes_param_validator.py

This module contains comprehensive unit tests for the validate_update_attributes_param
function that provides parameter validation for the UpdateAttributes processor.
"""

import warnings

from climakitae.core.constants import UNSET
from climakitae.new_core.param_validation.update_attributes_param_validator import (
    validate_update_attributes_param,
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


class TestValidateUpdateAttributesParam:
    """Test class for validate_update_attributes_param function."""

    def test_validate_with_string_value(self):
        """Test validate_update_attributes_param with string input.
        
        Tests that the validator accepts string values and returns True,
        demonstrating the permissive nature of the UpdateAttributes processor.
        """
        # Test with various string values
        result = validate_update_attributes_param("test_string")
        assert result is True
        
        result = validate_update_attributes_param("")
        assert result is True
        
        result = validate_update_attributes_param("complex string with spaces and 123")
        assert result is True

    def test_validate_with_numeric_values(self):
        """Test validate_update_attributes_param with numeric inputs.
        
        Tests that the validator accepts various numeric types (int, float)
        and returns True for all of them.
        """
        # Test with integer values
        result = validate_update_attributes_param(42)
        assert result is True
        
        result = validate_update_attributes_param(0)
        assert result is True
        
        result = validate_update_attributes_param(-100)
        assert result is True
        
        # Test with float values
        result = validate_update_attributes_param(3.14159)
        assert result is True
        
        result = validate_update_attributes_param(0.0)
        assert result is True
        
        result = validate_update_attributes_param(-2.5)
        assert result is True