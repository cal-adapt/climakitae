"""
Unit tests for climakitae/new_core/processors/warming_param_validator.py.

This module contains comprehensive unit tests for the Warming Level processor
parameter validation functionality.
"""

import warnings

import pytest

from climakitae.new_core.param_validation.warming_param_validator import (
    _check_input_types,
    validate_warming_level_param,
)


class TestCheckInputTypes:
    """Test class for _check_input_types function."""

    def test_check_input_types_valid_dict(self):
        """Test _check_input_types with a valid dictionary input."""
        value = {
            "warming_levels": [1.5, 2.0],
            "warming_level_months": [1, 2, 3],
            "warming_level_window": 15,
        }
        result = _check_input_types(value)
        assert result is True

    @pytest.mark.parametrize("container_type", [list, set, tuple])
    def test_check_input_types_invalid_type(self, container_type: type):
        """Test _check_input_types with an invalid type input."""
        value = container_type(["1.5", "2.0"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _check_input_types(value)
            assert result is False
            assert len(w) == 1
            assert "Warming Level Processor expects a dictionary of parameters." in str(
                w[0].message
            )

    @pytest.mark.parametrize("wrong_type", [None, "1.5, 2.0", [1.5, "two"]])
    def test_check_input_types_missing_warming_levels(self, wrong_type: object):
        """Test _check_input_types with missing 'warming_levels' key."""
        value = {
            "warming_levels": wrong_type,
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _check_input_types(value)
            assert result is False
            assert len(w) == 1
            assert "Invalid 'warming_levels' parameter." in str(w[0].message)

    @pytest.mark.parametrize("wrong_months", [[0, 13], ["January", 2], [1.5, 2.5]])
    def test_check_input_types_invalid_warming_level_months(self, wrong_months):
        """Test _check_input_types with invalid 'warming_level_months'."""
        value = {
            "warming_levels": [1.5, 2.0],
            "warming_level_months": wrong_months,
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _check_input_types(value)
            assert result is False
            assert len(w) == 1
            assert "Invalid 'warming_level_months' parameter." in str(w[0].message)

    @pytest.mark.parametrize("wrong_window", [-1, "fifteen", 2.5])
    def test_check_input_types_invalid_warming_level_window(self, wrong_window):
        """Test _check_input_types with invalid 'warming_level_window'."""
        value = {
            "warming_levels": [1.5, 2.0],
            "warming_level_window": wrong_window,
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _check_input_types(value)
            assert result is False
            assert len(w) == 1
            assert "Invalid 'warming_level_window' parameter." in str(w[0].message)


# class TestCheckQuery:
#     """Test class for _check_query function."""

#     def test_check_query_invalid_activity_id(self):
#         """Test _check_query with a invalid activity_id."""
#         value = {
#             "activity_id": "other thing",
#         }
#         result = validate_warming_level_param(value)
#         assert result is False

#     def test_check_query_valid_activity_id(self):
#         """Test _check_query with a valid activity_id."""
#         value = {
#             "activity_id": "WRF",
#         }
#         result = validate_warming_level_param(value)
#         assert result is True

#     def test_check_query_no_activity_id(self):
#         """Test _check_query with no activity_id."""
#         value = {}
#         result = validate_warming_level_param(value)
#         assert result is True

#     def test_check_query_invalid_experiment_id(self):
#         """Test _check_query with a invalid experiment_id."""
#         value = {
#             "experiment_id": "other thing",
#         }
#         result = validate_warming_level_param(value)
#         with warnings.catch_warnings(record=True) as w:
#             warnings.simplefilter("always")
#             assert result is False
#             assert len(w) == 1
#             assert (
#                 "Warming level approach requires 'experiment_id' to be UNSET."
#                 in str(w[0].message)
#             )


# class TestValidateWarmingLevelParam:

#     pass
