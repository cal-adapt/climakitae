"""
Unit tests for climakitae/new_core/param_validation/convert_units_param_validator.py

This module contains comprehensive unit tests for the Convert Units parameter
validation functionality that validates unit conversion parameters for the
Convert Units Processor.
"""

import warnings

import pytest

from climakitae.new_core.param_validation.convert_units_param_validator import (
    _check_input_types,
    _check_unit_validity,
    validate_convert_units_param,
)


class TestCheckInputTypes:
    """Test class for checking the input types of validate_convert_units_param using the _check_input_types function."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("K", True),
            (["K", "degC"], True),
            (123, False),
            (None, False),
            ({"unit": "K"}, False),
        ],
    )
    def test_check_input_types(self, value, expected):
        """Test _check_input_types with various input types.

        Tests validation with different types of inputs to ensure correct type checking.
        """
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
            (["K", "degC"], True),
            (["Pa", "hPa"], True),
            ("invalid_unit", False),
            (["K", "invalid_unit"], False),
        ],
    )
    def test_check_unit_validity(self, value, expected):
        """Test _check_unit_validity with various unit inputs.

        Tests validation with different unit strings and lists to ensure correct unit validity checking.
        """
        result = _check_unit_validity(value)
        assert result == expected


class TestValidateConvertUnitsValidateConvertUnitsParam:
    """Test class for validate_convert_units_param function."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("K", True),
            (["K", "degC"], True),
            ("invalid_unit", False),
            (["K", "invalid_unit"], False),
            (123, False),
            (None, False),
            ({"unit": "K"}, False),
        ],
    )
    def test_validate_convert_units_param(self, value, expected):
        """Test validate_convert_units_param with various inputs.

        Tests validation with different types of inputs to ensure correct overall validation.
        """
        result = validate_convert_units_param(value)
        assert result == expected

    def test_validate_convert_units_param_unset(self):
        """Test validate_convert_units_param with UNSET value.

        Tests validation when the value is UNSET, which should return True.
        """
        from climakitae.new_core.param_validation.base_param_validator import UNSET

        result = validate_convert_units_param(UNSET)
        assert result is True

    @pytest.mark.parametrize(
        "value",
        ["invalid_unit", ["K", "invalid_unit"], 123, None, {"unit": "K"}],
    )
    def test_validate_convert_units_param_warnings(self, value):
        """Test validate_convert_units_param to ensure warnings are raised for invalid inputs.

        Tests that a UserWarning is raised when invalid inputs are provided.
        """
        with pytest.warns(
            UserWarning,
            match="Convert Units Processor",
        ):
            result = validate_convert_units_param(value)
            assert result is False
