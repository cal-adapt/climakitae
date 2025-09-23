"""
Unit tests for climakitae/new_core/param_validation/update_attributes_param_validator.py

This module contains comprehensive unit tests for the validate_update_attributes_param
function that provides parameter validation for the UpdateAttributes processor.
"""

import warnings

import pytest

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

    @pytest.mark.parametrize(
        "test_value,description",
        [
            # String values
            ("test_string", "regular string"),
            ("", "empty string"),
            ("complex string with spaces and 123", "complex string"),
            # Numeric values
            (42, "positive integer"),
            (0, "zero integer"),
            (-100, "negative integer"),
            (3.14159, "positive float"),
            (0.0, "zero float"),
            (-2.5, "negative float"),
            # Boolean values
            (True, "boolean True"),
            (False, "boolean False"),
            # Special values
            (None, "None value"),
            ("__UNSET__", "UNSET constant"),  # Use string placeholder
            # Complex data structures
            ({"key": "value", "number": 42}, "dictionary with mixed types"),
            ({}, "empty dictionary"),
            ([1, 2, 3, "test"], "list with mixed types"),
            ([], "empty list"),
            (
                {"nested": {"data": [1, 2, 3]}, "list": ["a", "b", {"inner": "value"}]},
                "nested data structure",
            ),
        ],
        ids=lambda x: x[1] if isinstance(x, tuple) else str(x),
    )
    def test_validate_accepts_all_input_types(self, test_value, description):
        """Test that validate_update_attributes_param accepts all input types.

        Tests the permissive nature of the UpdateAttributes processor validator
        by verifying it returns True for any input type.

        Parameters
        ----------
        test_value : Any
            The value to test with the validator.
        description : str
            Human-readable description of the test case.
        """
        # Resolve the UNSET placeholder to the actual constant
        if test_value == "__UNSET__":
            test_value = UNSET

        result = validate_update_attributes_param(test_value)
        assert result is True, f"Validator should accept {description}: {test_value}"


class TestUpdateAttributesValidatorRegistration:
    """Test class for UpdateAttributes validator registration."""

    def test_validator_registration(self):
        """Test that validate_update_attributes_param is properly registered.

        Tests that the validator function is correctly registered with the
        processor validation system under the "update_attributes" key.
        """
        from climakitae.new_core.param_validation.abc_param_validation import (
            _PROCESSOR_VALIDATOR_REGISTRY,
        )

        # Verify that the update_attributes validator is registered
        assert "update_attributes" in _PROCESSOR_VALIDATOR_REGISTRY
        assert (
            _PROCESSOR_VALIDATOR_REGISTRY["update_attributes"]
            is validate_update_attributes_param
        )

    @pytest.mark.parametrize(
        "value,kwargs,description",
        [
            (
                "test_value",
                {
                    "extra_param": "ignored",
                    "another_kwarg": 42,
                    "complex_kwarg": {"nested": "data"},
                },
                "string value with multiple kwargs",
            ),
            (
                None,
                {
                    "processor_name": "update_attributes",
                    "dataset_info": {"source": "test"},
                },
                "None value with processor-specific kwargs",
            ),
            (
                42,
                {"param1": "value1", "param2": [1, 2, 3]},
                "numeric value with mixed kwargs",
            ),
            ({"data": "test"}, {}, "dictionary value with no kwargs"),
        ],
        ids=lambda x: x[2] if isinstance(x, tuple) else str(x),
    )
    def test_validate_with_kwargs(self, value, kwargs, description):
        """Test validate_update_attributes_param with keyword arguments.

        Tests that the validator properly handles keyword arguments
        (which are ignored per the function signature) and still returns True.

        Parameters
        ----------
        value : Any
            The positional value to pass to the validator.
        kwargs : dict
            Keyword arguments to pass to the validator.
        description : str
            Human-readable description of the test case.
        """
        result = validate_update_attributes_param(value, **kwargs)
        assert result is True, f"Validator should return True for {description}"
