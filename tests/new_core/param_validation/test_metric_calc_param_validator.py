"""
Unit tests for climakitae/new_core/param_validation/metric_calc_param_validator.py.

This module contains comprehensive unit tests for the MetricCalc parameter
validation functionality that validates parameters for the MetricCalc Processor.
"""

import pytest

from climakitae.new_core.param_validation.metric_calc_param_validator import (
    _validate_basic_metric_parameters,
    validate_metric_calc_param,
)


class TestValidateBasicMetricParameters:
    """Unit tests for the _validate_basic_metric_parameters function."""

    @pytest.mark.parametrize(
        "param_dict, expected",
        [
            # Valid cases
            ({"metric": "mean"}, True),
            ({"metric": "median", "dim": ["lat", "lon"]}, True),
            ({"percentiles": [10, 50, 90]}, True),
            ({"percentiles": [0, 100], "percentiles_only": True}, True),
            ({"dim": "time", "keepdims": True, "skipna": False}, True),
            # Invalid cases
            ({"metric": "unsupported_metric"}, False),
            ({"percentiles": [-10, 50]}, False),
            ({"percentiles": [10, 150]}, False),
            ({"percentiles_only": "yes"}, False),
            ({"dim": 123}, False),
            ({"keepdims": "no"}, False),
            ({"skipna": None}, False),
            # Non-dict input
            ("not_a_dict", False),
            (12345, False),
            (None, False),
        ],
    )
    def test_validate_basic_metric_parameters(self, param_dict, expected):
        """Test validate_metric_calc_param with various parameter combinations."""
        result = _validate_basic_metric_parameters(param_dict)
        assert result == expected

    # def test_validate_metric_calc_param_defaults(self):
    #     """Test validate_metric_calc_param with default parameters."""
    #     result = validate_metric_calc_param({})
    #     assert result is True

    # def test_validate_metric_calc_param_missing_keys(self):
    #     """Test validate_metric_calc_param with missing optional keys."""
    #     param_dict = {"metric": "max"}
    #     result = validate_metric_calc_param(param_dict)
    #     assert result is True
