"""
Unit tests for climakitae/new_core/param_validation/drop_leap_days_param_validator.py

This module contains comprehensive unit tests for the Drop Leap Days parameter
validation functionality that validates boolean parameters for the
Drop Leap Days Processor.
"""

import pytest

from climakitae.core.constants import UNSET
from climakitae.new_core.param_validation.drop_leap_days_param_validator import (
    _check_input_type,
    validate_drop_leap_days_param,
)


class TestCheckInputType:
    """Tests for _check_input_type in validate_drop_leap_days_param."""

    @pytest.mark.parametrize(
        "value,expected,warning_match",
        [
            (True, True, None),
            (False, True, None),
            ("True", False, "expects a boolean"),
            ("False", False, "expects a boolean"),
            (1, False, "expects a boolean"),
            (0, False, "expects a boolean"),
            (None, False, "expects a boolean"),
            ([], False, "expects a boolean"),
            ({}, False, "expects a boolean"),
            ("yes", False, "expects a boolean"),
        ],
    )
    def test_check_input_type(self, value, expected, warning_match):
        """Test _check_input_type with various input types."""
        if warning_match:
            with pytest.warns(UserWarning, match=warning_match):
                result = _check_input_type(value)
        else:
            result = _check_input_type(value)

        assert result == expected


class TestValidateDropLeapDaysParam:
    """Test class for validate_drop_leap_days_param function."""

    @pytest.mark.parametrize(
        "value,expected,warning_match",
        [
            (True, True, None),
            (False, True, None),
            ("True", False, "expects a boolean"),
            ("False", False, "expects a boolean"),
            (1, False, "expects a boolean"),
            (0, False, "expects a boolean"),
            (None, False, "expects a boolean"),
            ([], False, "expects a boolean"),
            ({}, False, "expects a boolean"),
            ("yes", False, "expects a boolean"),
            ("no", False, "expects a boolean"),
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

    def test_validate_drop_leap_days_param_unset(self):
        """Test validate_drop_leap_days_param with UNSET value.

        Tests validation when the value is UNSET, which should return True.
        """
        result = validate_drop_leap_days_param(UNSET)
        assert result is True

    def test_validate_drop_leap_days_param_with_kwargs(self):
        """Test that extra kwargs are ignored (for signature compatibility)."""
        result = validate_drop_leap_days_param(True, query={"some": "query"})
        assert result is True
