"""
Unit tests for climakitae/new_core/param_validation/convert_units_param_validator.py

This module contains comprehensive unit tests for the Convert Units parameter
validation functionality that validates unit conversion parameters for the
Convert Units Processor.
"""

import pytest

from climakitae.core.constants import UNSET
from climakitae.new_core.param_validation.convert_units_param_validator import (
    _check_input_types,
    _check_unit_validity,
    validate_convert_units_param,
)


class TestCheckInputTypes:
    """Tests for _check_input_types in validate_convert_units_param."""

    @pytest.mark.parametrize(
        "value,expected,warning_match",
        [
            ("K", True, None),
            (["K", "degC"], False, "expects a string"),
            (123, False, "expects a string"),
            (None, False, "expects a string"),
            (
                {"weird string that would be caught in _check_unit_validity": "K"},
                False,
                "expects a string",
            ),
            (
                ["one string", 123],
                False,
                "expects a string",
            ),
        ],
    )
    def test_check_input_types(self, value, expected, warning_match):
        """Test _check_input_types with various input types."""
        if warning_match:
            with pytest.warns(UserWarning, match=warning_match):
                result = _check_input_types(value)
        else:
            result = _check_input_types(value)

        assert result == expected


class TestCheckUnitValidity:
    """Test class for checking the unit validity of validate_convert_units_param using the _check_unit_validity function."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("K", True),
            ("degC", True),
            ("degF", True),
            ("Pa", True),
            ("hPa", True),
            ("mb", True),
            ("inHg", True),
            ("m/s", True),
            ("m s-1", True),
            ("mph", True),
            ("knots", True),
            ("[0 to 100]", True),
            ("fraction", True),
            ("mm", True),
            ("inches", True),
            ("mm/d", True),
            ("inches/d", True),
            ("mm/h", True),
            ("inches/h", True),
            ("kg/kg", True),
            ("g/kg", True),
            ("invalid_unit", False),
        ],
    )
    def test_check_unit_validity(self, value, expected):
        """Test _check_unit_validity with various unit inputs.

        Tests validation with different unit strings to ensure correct unit validity checking.
        """
        if expected is False:
            with pytest.warns(
                UserWarning,
                match="Unsupported unit:",
            ):
                result = _check_unit_validity(value)
        else:
            result = _check_unit_validity(value)
        assert result == expected


class TestValidateConvertUnitsParam:
    """Test class for validate_convert_units_param function."""

    @pytest.mark.parametrize(
        "value,expected,warning_match",
        [
            ("K", True, None),
            (["K", "degC"], False, "expects a string"),
            ("invalid_unit", False, "Unsupported unit:"),
            (["K", "invalid_unit"], False, "expects a string"),
            (123, False, "expects a string"),
            (None, False, "expects a string"),
            ({"unit": "K"}, False, "expects a string"),
            (
                ["one string", 123],
                False,
                "expects a string",
            ),
        ],
    )
    def test_validate_convert_units_param(self, value, expected, warning_match):
        """Test validate_convert_units_param with various inputs.

        Tests validation with different types of inputs to ensure correct overall validation.
        """
        if warning_match:
            with pytest.warns(UserWarning, match=warning_match):
                result = validate_convert_units_param(value)
        else:
            result = validate_convert_units_param(value)
        assert result == expected

    def test_validate_convert_units_param_unset(self):
        """Test validate_convert_units_param with UNSET value.

        Tests validation when the value is UNSET, which should return True.
        """
        result = validate_convert_units_param(UNSET)
        assert result is True
