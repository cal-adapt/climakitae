"""
Unit tests for climakitae/new_core/param_validation/drop_leap_days_param_validator.py

This module contains comprehensive unit tests for the Drop Leap Days parameter
validation functionality that validates string parameters for the
Drop Leap Days Processor.
"""

import pytest

from climakitae.new_core.param_validation.drop_leap_days_param_validator import (
    validate_drop_leap_days_param,
)


class TestValidateDropLeapDaysParam:
    """Test class for validate_drop_leap_days_param function."""

    @pytest.mark.parametrize(
        "value,expected,warning_match",
        [
            ("yes", True, None),
            ("no", True, None),
            ("YES", True, None),
            ("No", True, None),
            ("Yes", True, None),
            ("NO", True, None),
            ("invalid", False, "Invalid value"),
            ("maybe", False, "Invalid value"),
            (True, False, "expects a string"),
            (False, False, "expects a string"),
            (1, False, "expects a string"),
            (0, False, "expects a string"),
            (None, False, "expects a string"),
            ([], False, "expects a string"),
            ({}, False, "expects a string"),
        ],
    )
    def test_validate_drop_leap_days_param(self, value, expected, warning_match):
        """Test validate_drop_leap_days_param with various inputs.

        Tests validation with different types of inputs to ensure correct overall validation.
        """
        if warning_match:
            with pytest.warns(UserWarning, match=warning_match):
                result = validate_drop_leap_days_param(value)
        else:
            result = validate_drop_leap_days_param(value)
        assert result == expected

    def test_validate_drop_leap_days_param_with_kwargs(self):
        """Test that extra kwargs are ignored (for signature compatibility)."""
        result = validate_drop_leap_days_param("yes", query={"some": "query"})
        assert result is True
