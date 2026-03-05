"""
Validator for parameters provided to MetricCalc Processor.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from climakitae.core.constants import UNSET
from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)

# Module logger
logger = logging.getLogger(__name__)


@register_processor_validator("metric_calc")
def validate_metric_calc_param(
    value: dict[str, Any], **kwargs: Any
) -> bool:  # noqa: ARG001
    """
    Validate the parameters provided to the MetricCalc Processor.

    Parameters
    ----------
    value : dict[str, Any]
        Configuration dictionary with the following supported keys:

        Basic Metrics:
        - metric (str, optional): Metric to calculate. Supported values:
          "min", "max", "mean", "median". Default: "mean"
        - percentiles (list, optional): List of percentiles to calculate (0-100).
          Default: None
        - percentiles_only (bool, optional): If True and percentiles are specified,
          only calculate percentiles (skip metric). Default: False
        - dim (str or list, optional): Dimension(s) along which to calculate the metric/percentiles.
          Default: "time"
        - skipna (bool, optional): Whether to skip NaN values in calculations. Default: True

    Returns
    -------
    bool
        True if all parameters are valid, False otherwise
    """
    if not isinstance(value, dict):
        logger.warning(
            "\n\nMetricCalc Processor expects a dictionary configuration. "
            "\nPlease check the configuration."
        )
        return False

    # Extract parameters for validation
    metric = value.get("metric", "mean")
    percentiles = value.get("percentiles", UNSET)
    percentiles_only = value.get("percentiles_only", False)
    dim = value.get("dim", "time")
    skipna = value.get("skipna", True)

    if not _validate_basic_metric_parameters(
        metric, percentiles, percentiles_only, dim, skipna
    ):
        return False

    one_in_x_config = value.get("one_in_x", UNSET)
    thresholds_config = value.get("thresholds", UNSET)

    if one_in_x_config is not UNSET and thresholds_config is not UNSET:
        logger.warning("\n\nCannot set both 'thresholds' and 'one_in_x'. Choose one.")
        return False

    if one_in_x_config is not UNSET:
        if not _validate_one_in_x_parameters(one_in_x_config):
            return False

    if thresholds_config is not UNSET:
        if not _validate_threshold_parameters(thresholds_config):
            return False

    return True  # All parameters are valid


def _validate_basic_metric_parameters(
    metric: str,
    percentiles: list | None,
    percentiles_only: bool,
    dim: str | list,
    skipna: bool,
) -> bool:
    """Validate parameters for basic metric calculations."""
    valid_metrics = ["min", "max", "mean", "median", "sum"]

    # Validate metric
    if metric not in valid_metrics:
        logger.warning(
            f"\n\nInvalid metric '{metric}'. "
            f"\nSupported metrics are: {valid_metrics}"
        )
        return False

    # Validate percentiles
    if percentiles is not None and percentiles is not UNSET:
        if isinstance(percentiles, np.ndarray):
            percentiles = percentiles.tolist()
        if not isinstance(percentiles, list):
            logger.warning(
                f"\n\nPercentiles must be a list or numpy array, got {type(percentiles)}. "
                "\nPlease check the configuration."
            )
            return False

        for p in percentiles:
            if not isinstance(p, (int, float)) or not (0 <= p <= 100):
                logger.warning(
                    f"\n\nInvalid percentile value '{p}'. "
                    "\nPercentiles must be numbers between 0 and 100."
                )
                return False

    # Validate dim type
    if not isinstance(dim, (str, list)):
        logger.warning(
            f"\n\nParameter 'dim' must be a string or list, got {type(dim)}. "
            "\nPlease check the configuration."
        )
        return False

    if isinstance(dim, list):
        for d in dim:
            if not isinstance(d, str):
                logger.warning(
                    "\n\nAll dimension names must be strings. "
                    "\nPlease check the configuration."
                )
                return False

    # Validate boolean parameters
    for param_name, param_value in [
        ("percentiles_only", percentiles_only),
        ("skipna", skipna),
    ]:
        if not isinstance(param_value, bool):
            logger.warning(
                f"\n\nParameter '{param_name}' must be a boolean, got {type(param_value)}. "
                "\nPlease check the configuration."
            )
            return False

    # Validate percentiles_only logic
    if percentiles_only and (
        percentiles is None or percentiles is UNSET or percentiles == []
    ):
        logger.warning(
            "\n\npercentiles_only=True requires percentiles to be specified. "
            "\nPlease provide a list of percentiles or set percentiles_only=False."
        )
        return False

    return True


def _validate_one_in_x_parameters(one_in_x_config: dict) -> bool:
    """Validate parameters for 1-in-X calculations."""
    if not isinstance(one_in_x_config, dict):
        logger.warning(
            "\n\none_in_x configuration must be a dictionary. "
            "\nPlease check the configuration."
        )
        return False

    # Extract parameters with defaults
    return_periods = one_in_x_config.get("return_periods")
    distribution = one_in_x_config.get("distribution", "gev")
    extremes_type = one_in_x_config.get("extremes_type", "max")
    event_duration = one_in_x_config.get("event_duration", (1, "day"))
    block_size = one_in_x_config.get("block_size", 1)
    goodness_of_fit_test = one_in_x_config.get("goodness_of_fit_test", True)
    print_goodness_of_fit = one_in_x_config.get("print_goodness_of_fit", True)
    variable_preprocessing = one_in_x_config.get("variable_preprocessing", {})

    # Validate return_periods (required parameter)
    if return_periods is None:
        logger.warning(
            "\n\nreturn_periods is required for 1-in-X calculations. "
            "\nPlease provide a list of return periods (e.g., [10, 25, 50, 100])."
        )
        return False

    # Convert to numpy array for validation
    if not isinstance(return_periods, (list, np.ndarray)):
        return_periods_array = np.array([return_periods])
    elif isinstance(return_periods, list):
        return_periods_array = np.array(return_periods)
    else:
        return_periods_array = return_periods

    if not isinstance(return_periods_array, np.ndarray):
        logger.warning(
            "\n\nreturn_periods must be convertible to numpy array. "
            "\nPlease check the configuration."
        )
        return False

    for rp in return_periods_array:
        if not isinstance(rp, (int, float, np.integer, np.floating)) or rp < 1:
            logger.warning(
                "\n\nAll return periods must be numbers >= 1, got %s (type: %s). "
                "\nPlease check the configuration.",
                rp,
                type(rp),
            )
            return False

    # Validate distribution
    valid_distributions = ["gev", "genpareto", "gamma"]
    if distribution not in valid_distributions:
        logger.warning(
            "\n\nInvalid distribution '%s'. " "\nSupported distributions are: %s",
            distribution,
            valid_distributions,
        )
        return False

    # Validate extremes_type
    valid_extremes = ["max", "min"]
    if extremes_type not in valid_extremes:
        logger.warning(
            "\n\nInvalid extremes_type '%s'. " "\nSupported types are: %s",
            extremes_type,
            valid_extremes,
        )
        return False

    # Validate event_duration
    if not isinstance(event_duration, tuple) or len(event_duration) != 2:
        logger.warning(
            "\n\nevent_duration must be a tuple of (int, str). "
            "\nExample: (1, 'day') or (6, 'hour')"
        )
        return False

    duration_num, duration_unit = event_duration
    if not isinstance(duration_num, int) or duration_num <= 0:
        logger.warning(
            "\n\nevent_duration number must be a positive integer. "
            "\nPlease check the configuration."
        )
        return False

    if duration_unit not in ["hour", "day"]:
        logger.warning(
            "\n\nevent_duration unit must be 'hour' or 'day'. "
            "\nPlease check the configuration."
        )
        return False

    # Validate block_size
    if not isinstance(block_size, int) or block_size <= 0:
        logger.warning(
            "\n\nblock_size must be a positive integer. "
            "\nPlease check the configuration."
        )
        return False

    # Validate boolean parameters
    for param_name, param_value in [
        ("goodness_of_fit_test", goodness_of_fit_test),
        ("print_goodness_of_fit", print_goodness_of_fit),
    ]:
        if not isinstance(param_value, bool):
            logger.warning(
                "\n\nParameter '%s' must be a boolean, got %s. "
                "\nPlease check the configuration.",
                param_name,
                type(param_value),
            )
            return False

    # Validate variable_preprocessing
    if not isinstance(variable_preprocessing, dict):
        logger.warning(
            "\n\nvariable_preprocessing must be a dictionary. "
            "\nPlease check the configuration."
        )
        return False

    return True


def _validate_threshold_parameters(thresholds_config: dict) -> bool:
    """Validate parameters for threshold exceedance calculations."""
    if not isinstance(thresholds_config, dict):
        logger.warning(
            "\n\nthresholds configuration must be a dictionary. "
            "\nPlease check the configuration."
        )
        return False

    valid_time_units = ("year", "month", "day", "hour")

    # Validate threshold_value (required)
    threshold_value = thresholds_config.get("threshold_value")
    if threshold_value is None:
        logger.warning(
            "\n\nthreshold_value is required for threshold calculations. "
            "\nPlease provide a numeric threshold value."
        )
        return False
    if not isinstance(threshold_value, (int, float)):
        logger.warning(
            "\n\nthreshold_value must be a number (int or float), got %s. "
            "\nPlease check the configuration.",
            type(threshold_value),
        )
        return False
    if math.isnan(threshold_value):
        logger.warning(
            "\n\nthreshold_value must not be NaN. "
            "\nPlease provide a finite numeric value."
        )
        return False

    # Validate threshold_direction (required)
    threshold_direction = thresholds_config.get("threshold_direction")
    if threshold_direction not in ("above", "below"):
        logger.warning(
            "\n\nInvalid threshold_direction %r. " "\nMust be 'above' or 'below'.",
            threshold_direction,
        )
        return False

    # Validate period (optional): tuple(int, str)
    period = thresholds_config.get("period")
    if period is not None:
        if not isinstance(period, tuple) or len(period) != 2:
            logger.warning(
                "\n\nperiod must be a tuple of (int, str), e.g. (1, 'year'). "
                "\nPlease check the configuration."
            )
            return False
        period_num, period_unit = period
        if not isinstance(period_num, int) or period_num <= 0:
            logger.warning(
                "\n\nperiod number must be a positive integer. "
                "\nPlease check the configuration."
            )
            return False
        if period_unit not in valid_time_units:
            logger.warning(
                "\n\nperiod unit must be one of %s, got '%s'. "
                "\nPlease check the configuration.",
                valid_time_units,
                period_unit,
            )
            return False

    # Validate duration (optional): tuple(int, str)
    duration = thresholds_config.get("duration")
    if duration is not None:
        if not isinstance(duration, tuple) or len(duration) != 2:
            logger.warning(
                "\n\nduration must be a tuple of (int, str), e.g. (3, 'day'). "
                "\nPlease check the configuration."
            )
            return False
        duration_num, duration_unit = duration
        if not isinstance(duration_num, int) or duration_num <= 0:
            logger.warning(
                "\n\nduration number must be a positive integer. "
                "\nPlease check the configuration."
            )
            return False
        if duration_unit not in valid_time_units:
            logger.warning(
                "\n\nduration unit must be one of %s, got '%s'. "
                "\nPlease check the configuration.",
                valid_time_units,
                duration_unit,
            )
            return False

    return True
