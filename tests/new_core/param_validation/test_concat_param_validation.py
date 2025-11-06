"""
Unit tests for climakitae/new_core/param_validation/concat_param_validator.py

This module contains comprehensive unit tests for the concat parameter validation
functionality that validates dimension names for the Concat Processor.
"""

import warnings

import pytest

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

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            ("sim", True),
            ("time", True),
            ("ensemble", True),
        ],
    )
    def test_validate_concat_param_valid_string(self, input_value, expected):
        """Test validation with valid string input."""
        result = validate_concat_param(input_value)
        assert result is expected

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            ("  sim  ", True),
            ("\ttime\n", True),
            (" ensemble ", True),
        ],
    )
    def test_validate_concat_param_valid_string_with_whitespace(
        self, input_value, expected
    ):
        """Test validation with valid string that has leading/trailing whitespace."""
        result = validate_concat_param(input_value)
        assert result is expected

    def test_validate_concat_param_empty_string(self):
        """Test validation with empty string input."""
        with pytest.warns(UserWarning, match="dimension name cannot be empty"):
            result = validate_concat_param("")
            assert result is False

    @pytest.mark.parametrize(
        "input_value",
        [
            "   ",  # spaces only
            "\t\t",  # tabs only
            "\n\n",  # newlines only
            " \t\n ",  # mixed whitespace
        ],
    )
    def test_validate_concat_param_whitespace_only_string(self, input_value):
        """Test validation with whitespace-only string input."""
        with pytest.warns(UserWarning, match="dimension name cannot be empty"):
            result = validate_concat_param(input_value)
            assert result is False

    @pytest.mark.parametrize(
        "input_value",
        [
            123,  # integer
            12.5,  # float
            True,  # boolean
            None,  # None type
            ["sim"],  # list
            {"dim": "sim"},  # dictionary
            ("sim",),  # tuple
        ],
    )
    def test_validate_concat_param_invalid_type(self, input_value):
        """Test validation with invalid input types."""
        with pytest.warns(UserWarning, match="expects a string value"):
            result = validate_concat_param(input_value)
            assert result is False
