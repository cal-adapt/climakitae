"""
Unit tests for climakitae/new_core/param_validation/convert_to_local_time_param_validator.py

This module contains comprehensive unit tests for the ConvertToLocalTime processor
parameter validation functionality.
"""

import warnings

import pytest

from climakitae.new_core.param_validation.convert_to_local_time_param_validator import (
    validate_convert_to_local_time_param,
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


class TestValidateConvertToLocalTimeParam:
    """Test class for validate_convert_to_local_time_param function."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("yes", True),
            ("no", True),
        ],
        ids=["yes", "no"],
    )
    def test_validate_convert_to_local_time_param_valid_values(self, value, expected):
        """Test validate_convert_to_local_time_param with valid values.

        Tests validation with supported string values "yes" and "no".
        Both should return True.
        """
        result = validate_convert_to_local_time_param(value)
        assert result == expected

    @pytest.mark.parametrize(
        "value",
        [
            "true",
            "false",
            "YES",
            "NO",
            "y",
            "n",
            "1",
            "0",
            "invalid",
            "maybe",
            "",
        ],
        ids=[
            "true",
            "false",
            "YES_uppercase",
            "NO_uppercase",
            "y_short",
            "n_short",
            "numeric_1",
            "numeric_0",
            "invalid_word",
            "maybe",
            "empty_string",
        ],
    )
    def test_validate_convert_to_local_time_param_invalid_value(self, value):
        """Test validate_convert_to_local_time_param with invalid string values.

        Tests validation with various invalid string values that should
        trigger warnings and return False.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_convert_to_local_time_param(value)

            assert result is False
            assert len(w) == 1
            assert "Invalid value" in str(w[0].message)
            assert "ConvertToLocalTime Processor" in str(w[0].message)
            assert "Supported values are: ['yes', 'no']" in str(w[0].message)

    @pytest.mark.parametrize(
        "input_value",
        [
            123,
            12.5,
            True,
            False,
            None,
            ["yes"],
            {"value": "yes"},
            ("yes",),
        ],
        ids=[
            "integer",
            "float",
            "boolean_true",
            "boolean_false",
            "none",
            "list",
            "dict",
            "tuple",
        ],
    )
    def test_validate_convert_to_local_time_param_invalid_type(self, input_value):
        """Test validate_convert_to_local_time_param with non-string types.

        Tests validation with various non-string types that should
        trigger type warnings and return False.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_convert_to_local_time_param(input_value)

            assert result is False
            assert len(w) == 1
            assert "ConvertToLocalTime Processor expects a string value" in str(
                w[0].message
            )
            assert "Please check the configuration" in str(w[0].message)
