"""
Validator for parameters provided to MetricCalc Processor.
"""

from __future__ import annotations

import warnings
from typing import Any

from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)


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

    valid_metrics = ["min", "max", "mean", "median"]

    # Validate metric parameter
    if "metric" in value:
        metric = value["metric"]
        if not isinstance(metric, str) or metric not in valid_metrics:
            warnings.warn(
                f"\n\nInvalid metric '{metric}'. "
                f"\nSupported metrics are: {valid_metrics}"
            )
            return False

    # Validate percentiles parameter
    if "percentiles" in value:
        percentiles = value["percentiles"]
        if percentiles is not None:
            if not isinstance(percentiles, list):
                warnings.warn(
                    "\n\nPercentiles must be a list of numbers. "
                    "\nPlease check the configuration."
                )
                return False

            for p in percentiles:
                if not isinstance(p, (int, float)) or p < 0 or p > 100:
                    warnings.warn(
                        f"\n\nInvalid percentile value '{p}'. "
                        "\nPercentiles must be numbers between 0 and 100."
                    )
                    return False

    # Validate boolean parameters
    for bool_param in ["percentiles_only", "keepdims", "skipna"]:
        if bool_param in value:
            if not isinstance(value[bool_param], bool):
                warnings.warn(
                    f"\n\nParameter '{bool_param}' must be a boolean. "
                    "\nPlease check the configuration."
                )
                return False

    # Validate dim parameter
    if "dim" in value:
        dim = value["dim"]
        if not isinstance(dim, (str, list)):
            warnings.warn(
                "\n\nParameter 'dim' must be a string or list of strings. "
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

    return True  # All parameters are valid
