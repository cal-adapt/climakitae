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
    _validate_basic_metric_parameters,
    _validate_threshold_parameters,
    validate_metric_calc_param,
    _validate_one_in_x_parameters,
)


class TestValidateBasicMetricParameters:
    """Unit tests for the _validate_basic_metric_parameters function."""

    @pytest.mark.parametrize(
        "metric, percentiles, percentiles_only, dim, skipna, expected",
        [
            ("mean", None, False, "time", True, True),
            ("median", [10, 50, 90], True, ["time", "lat"], False, True),
            ("max", None, False, "lat", True, True),
            ("min", np.array([0, 100]), False, "lon", False, True),
            ("sum", None, False, ["lat", "lon"], True, True),
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
            ("mode", "Invalid metric"),
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

    def test_dim_or_dims_both_work(self):
        """Test that both 'dim' and 'dims' keys work in parameter dictionary."""
        param_dict_dim = {
            "metric": "mean",
            "dim": "time",
        }
        param_dict_dim2 = {
            "metric": "mean",
            "dim": ["time", "lat"],
        }
        param_dict_dims = {
            "metric": "mean",
            "dims": ["time", "lat"],
        }
        param_dict_dims2 = {
            "metric": "mean",
            "dims": "time",
        }
        result_dim = validate_metric_calc_param(param_dict_dim)
        result_dims = validate_metric_calc_param(param_dict_dims)
        result_dim2 = validate_metric_calc_param(param_dict_dim2)
        result_dims2 = validate_metric_calc_param(param_dict_dims2)
        assert result_dim is True
        assert result_dims is True
        assert result_dim2 is True
        assert result_dims2 is True


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


class TestValidateThresholdParameters:
    """Unit tests for the _validate_threshold_parameters function."""

    def test_valid_minimal_config(self):
        """Valid config with threshold_value and threshold_direction passes."""
        result = _validate_threshold_parameters(
            {"threshold_value": 110.0, "threshold_direction": "above"}
        )
        assert result is True

    def test_valid_full_config(self):
        """Valid config with all optional fields passes."""
        result = _validate_threshold_parameters(
            {
                "threshold_value": 32,
                "threshold_direction": "below",
                "period": (1, "year"),
                "duration": (3, "day"),
            }
        )
        assert result is True

    def test_not_a_dict(self, caplog):
        """Non-dict config returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_threshold_parameters("not_a_dict")
        assert result is False
        assert "thresholds configuration must be a dictionary" in caplog.text

    def test_missing_threshold_value(self, caplog):
        """Missing threshold_value returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_threshold_parameters({"threshold_direction": "above"})
        assert result is False
        assert "threshold_value is required" in caplog.text

    def test_invalid_threshold_value_type(self, caplog):
        """Non-numeric threshold_value returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_threshold_parameters({"threshold_value": "hot"})
        assert result is False
        assert "threshold_value must be a number" in caplog.text

    def test_nan_threshold_value(self, caplog):
        """float('nan') threshold_value returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_threshold_parameters({"threshold_value": float("nan")})
        assert result is False
        assert "threshold_value must not be NaN" in caplog.text

    def test_missing_threshold_direction(self, caplog):
        """Missing threshold_direction returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_threshold_parameters({"threshold_value": 110.0})
        assert result is False
        assert "Invalid threshold_direction" in caplog.text

    def test_invalid_threshold_direction(self, caplog):
        """Invalid threshold_direction returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_threshold_parameters(
                {
                    "threshold_value": 100,
                    "threshold_direction": "sideways",
                }
            )
        assert result is False
        assert "Invalid threshold_direction" in caplog.text

    @pytest.mark.parametrize(
        "period",
        [
            (1, "week"),  # bad unit
            (1, "day"),  # day no longer supported
            (1, "hour"),  # hour no longer supported
            (0, "month"),  # non-positive int
            (1.5, "year"),  # float instead of int
            (1,),  # wrong length
            "1year",  # not a tuple
        ],
    )
    def test_invalid_period(self, period, caplog):
        """Invalid period tuple returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_threshold_parameters(
                {
                    "threshold_value": 100,
                    "period": period,
                }
            )
        assert result is False

    @pytest.mark.parametrize(
        "duration",
        [
            (3, "week"),  # bad unit
            (0, "hour"),  # non-positive int
            (2.0, "day"),  # float instead of int
            (1,),  # wrong length
            "3days",  # not a tuple
        ],
    )
    def test_invalid_duration(self, duration, caplog):
        """Invalid duration tuple returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_threshold_parameters(
                {
                    "threshold_value": 100,
                    "duration": duration,
                }
            )
        assert result is False

    def test_via_dispatcher_valid(self):
        """Valid thresholds config passes through validate_metric_calc_param."""
        result = validate_metric_calc_param(
            {"thresholds": {"threshold_value": 95.0, "threshold_direction": "above"}}
        )
        assert result is True

    def test_via_dispatcher_invalid(self, caplog):
        """Invalid thresholds config is caught by validate_metric_calc_param."""
        with caplog.at_level(logging.WARNING):
            result = validate_metric_calc_param(
                {
                    "thresholds": {
                        "threshold_direction": "above"
                    }  # missing threshold_value
                }
            )
        assert result is False
        assert "threshold_value is required" in caplog.text

    def test_mutual_exclusivity_with_one_in_x(self, caplog):
        """Setting both thresholds and one_in_x returns False."""
        with caplog.at_level(logging.WARNING):
            result = validate_metric_calc_param(
                {
                    "thresholds": {"threshold_value": 100},
                    "one_in_x": {"return_periods": [10, 25]},
                }
            )
        assert result is False
        assert "Cannot set both 'thresholds' and 'one_in_x'" in caplog.text

    def test_mutual_exclusivity_with_metric(self, caplog):
        """Setting both thresholds and metric returns False."""
        with caplog.at_level(logging.WARNING):
            result = validate_metric_calc_param(
                {
                    "thresholds": {
                        "threshold_value": 100,
                        "threshold_direction": "above",
                    },
                    "metric": "mean",
                }
            )
        assert result is False
        assert "Cannot set both 'thresholds' and 'metric'" in caplog.text

    def test_mutual_exclusivity_with_percentiles(self, caplog):
        """Setting both thresholds and percentiles returns False."""
        with caplog.at_level(logging.WARNING):
            result = validate_metric_calc_param(
                {
                    "thresholds": {
                        "threshold_value": 100,
                        "threshold_direction": "above",
                    },
                    "percentiles": [10, 50, 90],
                }
            )
        assert result is False
        assert "Cannot set both 'thresholds' and 'percentiles'" in caplog.text


class TestValidateOneInXParameters:
    """Unit tests for the _validate_one_in_x_parameters function."""

    def test_valid_minimal_config(self):
        """Valid config with threshold_value and threshold_direction passes."""
        result = _validate_one_in_x_parameters({"return_periods": 10})
        assert result is True

    def test_valid_full_config(self):
        """Valid config with all optional fields passes."""
        result = _validate_one_in_x_parameters(
            {
                "return_periods": [10, 20],
                "distribution": "gev",
                "extremes_type": "max",
                "duration": (3, "day"),
                "variable_preprocessing": {
                    "precipitation": {"daily_aggregation": True}
                },
            }
        )
        assert result is True

    def test_not_a_dict(self, caplog):
        """Non-dict config returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_one_in_x_parameters("not_a_dict")
        assert result is False
        assert "one_in_x configuration must be a dictionary" in caplog.text

    def test_missing_return_parameter(self, caplog):
        """Missing return_periods or return_values returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_one_in_x_parameters({"distribution": "gev"})
        assert result is False
        assert (
            "Either return_periods or return_values is required for 1-in-X calculations."
            in caplog.text
        )

    def test_invalid_distribution(self, caplog):
        """Unsupported distribution returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_one_in_x_parameters(
                {"return_periods": 10, "distribution": "gnpareto"}
            )
        assert result is False
        assert "Invalid distribution 'gnpareto'. " in caplog.text

    def test_invalid_extremes_type(self, caplog):
        """Unsupported distribution returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_one_in_x_parameters(
                {"return_periods": 10, "extremes_type": "maximum"}
            )
        assert result is False
        assert "Invalid extremes_type 'maximum'." in caplog.text

    def test_invalid_event_duration(self, caplog):
        """Non-tuple event duration returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_one_in_x_parameters(
                {"return_periods": 10, "event_duration": 1}
            )
        assert result is False
        assert "event_duration must be a tuple of (int, str)." in caplog.text

    def test_invalid_block_size(self, caplog):
        """float(block_size) returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_one_in_x_parameters(
                {"return_periods": 10, "block_size": 1.5}
            )
        assert result is False
        assert "block_size must be a positive integer." in caplog.text

    def test_invalid_bootstrap_runs_size(self, caplog):
        """float(bootstrap_runs) returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_one_in_x_parameters(
                {"return_periods": 10, "bootstrap_runs": 1.5}
            )
        assert result is False
        assert "bootstrap_runs must be a positive integer." in caplog.text

    def test_invalid_alpha_size(self, caplog):
        """float(alpha_size) returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_one_in_x_parameters({"return_periods": 10, "alpha": -1})
        assert result is False
        assert "alpha must be a positive float less than 1." in caplog.text

    def test_invalid_goodness_of_fit(self, caplog):
        """Non-boolean goodness_of_fit returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_one_in_x_parameters(
                {"return_periods": 10, "goodness_of_fit_test": "True"}
            )
        assert result is False
        assert (
            "Parameter 'goodness_of_fit_test' must be a boolean, got <class 'str'>."
            in caplog.text
        )

    def test_invalid_check_ess(self, caplog):
        """Non-boolean check_ess returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_one_in_x_parameters(
                {"return_periods": 10, "check_ess": "True"}
            )
        assert result is False
        assert (
            "Parameter 'check_ess' must be a boolean, got <class 'str'>."
            in caplog.text
        )

    def test_invalid_multiple_points(self, caplog):
        """Non-boolean multiple_points returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_one_in_x_parameters(
                {"return_periods": 10, "multiple_points": "True"}
            )
        assert result is False
        assert (
            "Parameter 'multiple_points' must be a boolean, got <class 'str'>."
            in caplog.text

    @pytest.mark.parametrize(
        "period",
        [
            (1, "week"),  # bad unit
            (1, "day"),  # day no longer supported
            (1, "hour"),  # hour no longer supported
            (0, "month"),  # non-positive int
            (1.5, "year"),  # float instead of int
            (1,),  # wrong length
            "1year",  # not a tuple
        ],
    )
    def test_invalid_period(self, period, caplog):
        """Invalid period tuple returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_one_in_x_parameters(
                {
                    "return_value": 100,
                    "period": period,
                }
            )
        assert result is False

    @pytest.mark.parametrize(
        "duration",
        [
            (3, "week"),  # bad unit
            (0, "hour"),  # non-positive int
            (2.0, "day"),  # float instead of int
            (1,),  # wrong length
            "3days",  # not a tuple
        ],
    )
    def test_invalid_duration(self, duration, caplog):
        """Invalid duration tuple returns False."""
        with caplog.at_level(logging.WARNING):
            result = _validate_one_in_x_parameters(
                {
                    "return_values": 100,
                    "event_duration": duration,
                }
            )
        assert result is False

    def test_via_dispatcher_valid(self):
        """Valid thresholds config passes through validate_metric_calc_param."""
        result = validate_metric_calc_param({"one_in_x": {"return_values": 95.0}})
        assert result is True

    def test_via_dispatcher_invalid(self, caplog):
        """Invalid thresholds config is caught by validate_metric_calc_param."""
        with caplog.at_level(logging.WARNING):
            result = validate_metric_calc_param(
                {
                    "one_in_x": {
                        "distribution": "gev"
                    }  # missing return_values or return_periods
                }
            )
        assert result is False
        assert (
            "Either return_periods or return_values is required for 1-in-X calculations."
            in caplog.text
        )

    def test_mutual_exclusivity_with_thresholds(self, caplog):
        """Setting both thresholds and one_in_x returns False."""
        with caplog.at_level(logging.WARNING):
            result = validate_metric_calc_param(
                {
                    "one_in_x": {"return_periods": [10, 25]},
                    "thresholds": {"threshold_value": 100},
                }
            )
        assert result is False
        assert "Cannot set both 'thresholds' and 'one_in_x'" in caplog.text
