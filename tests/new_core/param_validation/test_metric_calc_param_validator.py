"""
Unit tests for climakitae/new_core/param_validation/metric_calc_param_validator.py.

This module contains comprehensive unit tests for the MetricCalc parameter
validation functionality that validates parameters for the MetricCalc Processor.
"""

import logging

import numpy as np
import pytest

from climakitae.core.constants import UNSET
from climakitae.new_core.param_validation.metric_calc_param_validator import (
    _validate_basic_metric_parameters, validate_metric_calc_param)


class TestValidateBasicMetricParameters:
    """Unit tests for the _validate_basic_metric_parameters function."""

    @pytest.mark.parametrize(
        "metric, percentiles, percentiles_only, dim, skipna, expected",
        [
            ("mean", None, False, "time", True, True),
            ("median", [10, 50, 90], True, ["time", "lat"], False, True),
            ("max", None, False, "lat", True, True),
            ("min", np.array([0, 100]), False, "lon", False, True),
        ],
    )
    def test_valid_cases(
        self, metric, percentiles, percentiles_only, dim, skipna, expected
    ):
        """Test _validate_basic_metric_parameters with valid inputs."""
        result = _validate_basic_metric_parameters(
            metric, percentiles, percentiles_only, dim, skipna
        )
        assert result == expected

    def test_param_empty_dict(self):
        """Test validate_metric_calc_param with empty dict uses defaults."""
        result = validate_metric_calc_param({})
        assert result is True

    def test_param_with_only_percentiles(self):
        """Test with only percentiles specified."""
        param_dict = {"percentiles": [25, 50, 75]}
        result = validate_metric_calc_param(param_dict)
        assert result is True

    def test_param_with_kwargs(self):
        """Test that extra kwargs don't break validation."""
        param_dict = {
            "metric": "max",
            "percentiles": [90],
            "extra_param": "should be ignored",
        }
        result = validate_metric_calc_param(param_dict, extra_kwarg=123)
        assert result is True

    @pytest.mark.parametrize(
        "metric, warning_msg",
        [
            ("sum", "Invalid metric"),
        ],
    )
    def test_invalid_metric(self, metric, warning_msg, caplog):
        """Test _validate_basic_metric_parameters with invalid metric."""
        with caplog.at_level(logging.WARNING):
            result = _validate_basic_metric_parameters(
                metric, None, False, "time", True
            )
        assert result is False
        assert warning_msg in caplog.text

    @pytest.mark.parametrize(
        "percentiles, warning_msg",
        [
            ([-10, 50], "Invalid percentile value"),
            ([10, 150], "Invalid percentile value"),
            ("not_a_list", "Percentiles must be a list or numpy array"),
        ],
    )
    def test_invalid_percentiles(self, percentiles, warning_msg, caplog):
        """Test _validate_basic_metric_parameters with invalid percentiles."""
        with caplog.at_level(logging.WARNING):
            result = _validate_basic_metric_parameters(
                "mean", percentiles, False, "time", True
            )
        assert result is False
        assert warning_msg in caplog.text

    @pytest.mark.parametrize(
        "percentiles, expected",
        [
            ([0.0, 100.0], True),
            ([0, 100], True),
        ],
    )
    def test_percentiles_at_bounds(self, percentiles, expected):
        """Test _validate_basic_metric_parameters with percentiles at bounds 0 and 100."""
        result = _validate_basic_metric_parameters(
            "mean", percentiles, False, "time", True
        )
        assert result == expected

    def test_percentiles_empty_and_percentiles_only_true(self, caplog):
        """Test _validate_basic_metric_parameters with empty percentiles and percentiles_only True."""
        with caplog.at_level(logging.WARNING):
            result = _validate_basic_metric_parameters("mean", [], True, "time", True)
        assert result is False
        assert (
            "percentiles_only=True requires percentiles to be specified." in caplog.text
        )

    def test_percentiles_only_without_percentiles(self, caplog):
        """Test _validate_basic_metric_parameters with percentiles_only True but no percentiles."""
        with caplog.at_level(logging.WARNING):
            result = _validate_basic_metric_parameters("mean", None, True, "time", True)
        assert result is False
        assert (
            "percentiles_only=True requires percentiles to be specified." in caplog.text
        )

    @pytest.mark.parametrize(
        "percentiles_only, warning_msg",
        [
            ("not_a_bool", "Parameter 'percentiles_only' must be a boolean"),
        ],
    )
    def test_invalid_percentiles_only(self, percentiles_only, warning_msg, caplog):
        """Test _validate_basic_metric_parameters with invalid percentiles_only."""
        with caplog.at_level(logging.WARNING):
            result = _validate_basic_metric_parameters(
                "mean", [10, 50], percentiles_only, "time", True
            )
        assert result is False
        assert warning_msg in caplog.text

    @pytest.mark.parametrize(
        "dim, warning_msg",
        [
            (123, "Parameter 'dim' must be a string or list"),
            (["time", 456], "All dimension names must be strings"),
        ],
    )
    def test_invalid_dim(self, dim, warning_msg, caplog):
        """Test _validate_basic_metric_parameters with invalid dim."""
        with caplog.at_level(logging.WARNING):
            result = _validate_basic_metric_parameters("mean", None, False, dim, True)
        assert result is False
        assert warning_msg in caplog.text

    @pytest.mark.parametrize(
        "skipna, warning_msg",
        [
            ("not_a_bool", "Parameter 'skipna' must be a boolean"),
        ],
    )
    def test_invalid_skipna(self, skipna, warning_msg, caplog):
        """Test _validate_basic_metric_parameters with invalid skipna."""
        with caplog.at_level(logging.WARNING):
            result = _validate_basic_metric_parameters(
                "mean", None, False, "time", skipna
            )
        assert result is False
        assert warning_msg in caplog.text

    @pytest.mark.parametrize(
        "param, warning_msg",
        [
            ("metric", "Invalid metric"),
            ("percentiles_only", "Parameter 'percentiles_only' must be a boolean"),
            ("dim", "Parameter 'dim' must be a string or list"),
            ("skipna", "Parameter 'skipna' must be a boolean"),
        ],
    )
    def test_each_unset_parameter(self, param, warning_msg, caplog):
        """Test _validate_basic_metric_parameters with each UNSET parameter individually."""
        params = {
            "metric": "mean",
            "percentiles": UNSET,
            "percentiles_only": False,
            "dim": "time",
            "skipna": True,
        }
        test_params = params.copy()
        test_params[param] = UNSET
        with caplog.at_level(logging.WARNING):
            result = _validate_basic_metric_parameters(
                test_params["metric"],
                test_params["percentiles"],
                test_params["percentiles_only"],
                test_params["dim"],
                test_params["skipna"],
            )
        assert result is False
        assert warning_msg in caplog.text


class TestValidateMetricCalcParam:
    """Unit tests for the validate_metric_calc_param function."""

    def test_param_dict(self):
        """Test validate_metric_calc_param with a valid parameter dictionary."""
        param_dict = {
            "metric": "mean",
            "percentiles": [10, 50, 90],
            "percentiles_only": False,
            "dim": "time",
            "skipna": True,
        }
        result = validate_metric_calc_param(param_dict)
        assert result is True

    def test_param_not_dict(self, caplog):
        """Test validate_metric_calc_param with a non-dictionary parameter."""
        param_not_dict = ["not", "a", "dict"]
        with caplog.at_level(logging.WARNING):
            result = validate_metric_calc_param(param_not_dict)
        assert result is False
        assert "MetricCalc Processor expects a dictionary" in caplog.text

    def test_invalid_basic_parameters(self, caplog):
        """Test validate_metric_calc_param with invalid basic parameters."""
        param_dict = {
            "metric": "invalid_metric",
            "percentiles": [10, 50, 90],
            "percentiles_only": False,
            "dim": "time",
            "skipna": True,
        }
        with caplog.at_level(logging.WARNING):
            result = validate_metric_calc_param(param_dict)
        assert result is False
        assert "Invalid metric" in caplog.text
