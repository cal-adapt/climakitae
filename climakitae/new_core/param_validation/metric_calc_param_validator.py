"""
Validator for parameters provided to MetricCalc Processor.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)

# Check if extreme value analysis functions are available
try:
    from climakitae.explore.threshold_tools import (
        get_block_maxima,
        get_ks_stat,
        get_return_value,
    )

    EXTREME_VALUE_ANALYSIS_AVAILABLE = True
except ImportError:
    EXTREME_VALUE_ANALYSIS_AVAILABLE = False


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
        - keepdims (bool, optional): Whether to keep the dimensions being reduced. Default: False
        - skipna (bool, optional): Whether to skip NaN values in calculations. Default: True

        1-in-X Return Value Analysis:
        - one_in_x (dict, optional): Configuration for 1-in-X extreme value analysis.
          If provided, performs extreme value analysis instead of basic metrics. Keys:
          - return_periods (list): List of return periods (e.g., [10, 25, 50, 100])
          - distribution (str, optional): Distribution for fitting ("gev", "genpareto", "gamma"). Default: "gev"
          - extremes_type (str, optional): "max" or "min". Default: "max"
          - event_duration (tuple, optional): Event duration as (int, str). Default: (1, "day")
          - block_size (int, optional): Block size in years. Default: 1
          - goodness_of_fit_test (bool, optional): Perform KS test. Default: True
          - print_goodness_of_fit (bool, optional): Print p-value results. Default: True
          - variable_preprocessing (dict, optional): Variable-specific preprocessing options

    Returns
    -------
    bool
        True if all parameters are valid, False otherwise
    """
    if not isinstance(value, dict):
        warnings.warn(
            "\n\nMetricCalc Processor expects a dictionary configuration. "
            "\nPlease check the configuration."
        )
        return False

    # Extract parameters for validation
    metric = value.get("metric", "mean")
    percentiles = value.get("percentiles", None)
    percentiles_only = value.get("percentiles_only", False)
    dim = value.get("dim", "time")
    keepdims = value.get("keepdims", False)
    skipna = value.get("skipna", True)
    one_in_x_config = value.get("one_in_x", None)

    # Check that only one calculation type is specified
    explicit_metric_specified = "metric" in value and value["metric"] != "mean"
    percentiles_specified = percentiles is not None
    basic_metrics_specified = explicit_metric_specified or percentiles_specified
    one_in_x_specified = one_in_x_config is not None

    if basic_metrics_specified and one_in_x_specified:
        warnings.warn(
            "\n\nCannot specify both basic metrics/percentiles and one_in_x calculations simultaneously. "
            "\nPlease choose either basic metrics/percentiles OR one_in_x analysis."
        )
        return False

    # Validate basic metric parameters
    if not one_in_x_specified:
        if not _validate_basic_metric_parameters(
            metric, percentiles, percentiles_only, dim, keepdims, skipna
        ):
            return False
    else:
        if not _validate_one_in_x_parameters(one_in_x_config):
            return False

    return True  # All parameters are valid


def _validate_basic_metric_parameters(
    metric: str,
    percentiles: list | None,
    percentiles_only: bool,
    dim: str | list,
    keepdims: bool,
    skipna: bool,
) -> bool:
    """Validate parameters for basic metric calculations."""
    valid_metrics = ["min", "max", "mean", "median"]

    # Validate metric
    if metric not in valid_metrics:
        warnings.warn(
            f"\n\nInvalid metric '{metric}'. "
            f"\nSupported metrics are: {valid_metrics}"
        )
        return False

    # Validate percentiles
    if percentiles is not None:
        if not isinstance(percentiles, list):
            warnings.warn(
                f"\n\nPercentiles must be a list, got {type(percentiles)}. "
                "\nPlease check the configuration."
            )
            return False

        for p in percentiles:
            if not isinstance(p, (int, float)) or not (0 <= p <= 100):
                warnings.warn(
                    f"\n\nInvalid percentile value '{p}'. "
                    "\nPercentiles must be numbers between 0 and 100."
                )
                return False

    # Validate dim type
    if not isinstance(dim, (str, list)):
        warnings.warn(
            f"\n\nParameter 'dim' must be a string or list, got {type(dim)}. "
            "\nPlease check the configuration."
        )
        return False

    if isinstance(dim, list):
        for d in dim:
            if not isinstance(d, str):
                warnings.warn(
                    "\n\nAll dimension names must be strings. "
                    "\nPlease check the configuration."
                )
                return False

    # Validate boolean parameters
    for param_name, param_value in [
        ("percentiles_only", percentiles_only),
        ("keepdims", keepdims),
        ("skipna", skipna),
    ]:
        if not isinstance(param_value, bool):
            warnings.warn(
                f"\n\nParameter '{param_name}' must be a boolean, got {type(param_value)}. "
                "\nPlease check the configuration."
            )
            return False

    # Validate percentiles_only logic
    if percentiles_only and percentiles is None:
        warnings.warn(
            "\n\npercentiles_only=True requires percentiles to be specified. "
            "\nPlease provide a list of percentiles or set percentiles_only=False."
        )
        return False

    return True


def _validate_one_in_x_parameters(one_in_x_config: dict) -> bool:
    """Validate parameters for 1-in-X calculations."""
    if not EXTREME_VALUE_ANALYSIS_AVAILABLE:
        warnings.warn(
            "\n\nExtreme value analysis functions are not available. "
            "\nPlease check your climakitae installation."
        )
        return False

    if not isinstance(one_in_x_config, dict):
        warnings.warn(
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
        warnings.warn(
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
        warnings.warn(
            "\n\nreturn_periods must be convertible to numpy array. "
            "\nPlease check the configuration."
        )
        return False

    for rp in return_periods_array:
        if not isinstance(rp, (int, float, np.integer, np.floating)) or rp < 1:
            warnings.warn(
                f"\n\nAll return periods must be numbers >= 1, got {rp} (type: {type(rp)}). "
                "\nPlease check the configuration."
            )
            return False

    # Validate distribution
    valid_distributions = ["gev", "genpareto", "gamma"]
    if distribution not in valid_distributions:
        warnings.warn(
            f"\n\nInvalid distribution '{distribution}'. "
            f"\nSupported distributions are: {valid_distributions}"
        )
        return False

    # Validate extremes_type
    valid_extremes = ["max", "min"]
    if extremes_type not in valid_extremes:
        warnings.warn(
            f"\n\nInvalid extremes_type '{extremes_type}'. "
            f"\nSupported types are: {valid_extremes}"
        )
        return False

    # Validate event_duration
    if not isinstance(event_duration, tuple) or len(event_duration) != 2:
        warnings.warn(
            "\n\nevent_duration must be a tuple of (int, str). "
            "\nExample: (1, 'day') or (6, 'hour')"
        )
        return False

    duration_num, duration_unit = event_duration
    if not isinstance(duration_num, int) or duration_num <= 0:
        warnings.warn(
            "\n\nevent_duration number must be a positive integer. "
            "\nPlease check the configuration."
        )
        return False

    if duration_unit not in ["hour", "day"]:
        warnings.warn(
            "\n\nevent_duration unit must be 'hour' or 'day'. "
            "\nPlease check the configuration."
        )
        return False

    # Validate block_size
    if not isinstance(block_size, int) or block_size <= 0:
        warnings.warn(
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
            warnings.warn(
                f"\n\nParameter '{param_name}' must be a boolean, got {type(param_value)}. "
                "\nPlease check the configuration."
            )
            return False

    # Validate variable_preprocessing
    if not isinstance(variable_preprocessing, dict):
        warnings.warn(
            "\n\nvariable_preprocessing must be a dictionary. "
            "\nPlease check the configuration."
        )
        return False

    return True
