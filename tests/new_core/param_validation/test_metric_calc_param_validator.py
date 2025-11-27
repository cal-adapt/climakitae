"""
Unit tests for climakitae/new_core/param_validation/metric_calc_param_validator.py.

This module contains comprehensive unit tests for the MetricCalc parameter
validation functionality that validates parameters for the MetricCalc Processor.
"""

import numpy as np
import pytest

from climakitae.new_core.param_validation.metric_calc_param_validator import (
    _validate_basic_metric_parameters,
    validate_metric_calc_param,
)


class TestValidateBasicMetricParameters:
    """Unit tests for the _validate_basic_metric_parameters function."""

    @pytest.mark.parametrize(
        "metric, percentiles, percentiles_only, dim, keepdims, skipna, expected",
        [
            ("mean", None, False, "time", False, True, True),
            ("median", [10, 50, 90], True, ["time", "lat"], True, False, True),
            ("max", None, False, "lat", False, True, True),
            ("min", np.array([0, 100]), False, "lon", True, False, True),
        ],
    )
    def test_valid_cases(
        self, metric, percentiles, percentiles_only, dim, keepdims, skipna, expected
    ):
        """Test _validate_basic_metric_parameters with valid inputs."""
        result = _validate_basic_metric_parameters(
            metric, percentiles, percentiles_only, dim, keepdims, skipna
        )
        assert result == expected

    @pytest.mark.parametrize(
        "metric, warning_msg",
        [
            ("sum", "Invalid metric"),
        ],
    )
    def test_invalid_metric(self, metric, warning_msg):
        """Test _validate_basic_metric_parameters with invalid metric."""
        with pytest.warns(UserWarning, match=warning_msg):
            result = _validate_basic_metric_parameters(
                metric, None, False, "time", False, True
            )
        assert result is False

    @pytest.mark.parametrize(
        "percentiles, warning_msg",
        [
            ([-10, 50], "Invalid percentile value"),
            ([10, 150], "Invalid percentile value"),
            ("not_a_list", "Percentiles must be a list or numpy array"),
        ],
    )
    def test_invalid_percentiles(self, percentiles, warning_msg):
        """Test _validate_basic_metric_parameters with invalid percentiles."""
        with pytest.warns(UserWarning, match=warning_msg):
            result = _validate_basic_metric_parameters(
                "mean", percentiles, False, "time", False, True
            )
        assert result is False

    @pytest.mark.parametrize(
        "percentiles_only, warning_msg",
        [
            ("not_a_bool", "Parameter 'percentiles_only' must be a boolean"),
        ],
    )
    def test_invalid_percentiles_only(self, percentiles_only, warning_msg):
        """Test _validate_basic_metric_parameters with invalid percentiles_only."""
        with pytest.warns(UserWarning, match=warning_msg):
            result = _validate_basic_metric_parameters(
                "mean", [10, 50], percentiles_only, "time", False, True
            )
        assert result is False

    @pytest.mark.parametrize(
        "dim, warning_msg",
        [
            (123, "Parameter 'dim' must be a string or list"),
            (["time", 456], "All dimension names must be strings"),
        ],
    )
    def test_invalid_dim(self, dim, warning_msg):
        """Test _validate_basic_metric_parameters with invalid dim."""
        with pytest.warns(UserWarning, match=warning_msg):
            result = _validate_basic_metric_parameters(
                "mean", None, False, dim, False, True
            )
        assert result is False

    @pytest.mark.parametrize(
        "keepdims, warning_msg",
        [
            ("not_a_bool", "Parameter 'keepdims' must be a boolean"),
        ],
    )
    def test_invalid_keepdims(self, keepdims, warning_msg):
        """Test _validate_basic_metric_parameters with invalid keepdims."""
        with pytest.warns(UserWarning, match=warning_msg):
            result = _validate_basic_metric_parameters(
                "mean", None, False, "time", keepdims, True
            )
        assert result is False

    @pytest.mark.parametrize(
        "skipna, warning_msg",
        [
            ("not_a_bool", "Parameter 'skipna' must be a boolean"),
        ],
    )
    def test_invalid_skipna(self, skipna, warning_msg):
        """Test _validate_basic_metric_parameters with invalid skipna."""
        with pytest.warns(UserWarning, match=warning_msg):
            result = _validate_basic_metric_parameters(
                "mean", None, False, "time", False, skipna
            )
        assert result is False

    def test_percentiles_only_without_percentiles(self):
        """Test _validate_basic_metric_parameters with percentiles_only True but no percentiles."""
        with pytest.warns(
            UserWarning,
            match="percentiles_only=True requires percentiles to be specified.",
        ):
            result = _validate_basic_metric_parameters(
                "mean", None, True, "time", False, True
            )
        assert result is False


class TestValidateMetricCalcParam:
    """Unit tests for the validate_metric_calc_param function."""

    def test_param_dict(self):
        """Test validate_metric_calc_param with a valid parameter dictionary."""
        param_dict = {
            "metric": "mean",
            "percentiles": [10, 50, 90],
            "percentiles_only": False,
            "dim": "time",
            "keepdims": True,
            "skipna": True,
        }
        result = validate_metric_calc_param(param_dict)
        assert result is True

    def test_param_not_dict(self):
        """Test validate_metric_calc_param with a non-dictionary parameter."""
        param_not_dict = ["not", "a", "dict"]
        with pytest.warns(
            UserWarning, match="MetricCalc Processor expects a dictionary"
        ):
            result = validate_metric_calc_param(param_not_dict)
        assert result is False

    def test_invalid_basic_parameters(self):
        """Test validate_metric_calc_param with invalid basic parameters."""
        param_dict = {
            "metric": "invalid_metric",
            "percentiles": [10, 50, 90],
            "percentiles_only": False,
            "dim": "time",
            "keepdims": True,
            "skipna": True,
        }
        with pytest.warns(UserWarning, match="Invalid metric"):
            result = validate_metric_calc_param(param_dict)
        assert result is False
