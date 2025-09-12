"""
Unit tests for climakitae/new_core/param_validation/concat_param_validator.py

This module contains comprehensive unit tests for the concat parameter validation
functionality that validates dimension names for the Concat Processor.
"""

import warnings

from climakitae.new_core.param_validation.concat_param_validator import (
    validate_concat_param,
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


class TestValidateConcatParam:
    """Test class for validate_concat_param function validation logic."""

    def test_validate_concat_param_valid_string(self):
        """Test validation with valid string input."""
        result = validate_concat_param("sim")
        assert result is True

        result = validate_concat_param("time")
        assert result is True

        result = validate_concat_param("ensemble")
        assert result is True

    def test_validate_concat_param_valid_string_with_whitespace(self):
        """Test validation with valid string that has leading/trailing whitespace."""
        result = validate_concat_param("  sim  ")
        assert result is True

        result = validate_concat_param("\ttime\n")
        assert result is True

        result = validate_concat_param(" ensemble ")
        assert result is True
