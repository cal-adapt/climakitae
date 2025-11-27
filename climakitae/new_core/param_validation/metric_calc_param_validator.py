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

    if not _validate_basic_metric_parameters(
        metric, percentiles, percentiles_only, dim, keepdims, skipna
    ):
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
        if isinstance(percentiles, np.ndarray):
            percentiles = percentiles.tolist()
        if not isinstance(percentiles, list):
            warnings.warn(
                f"\n\nPercentiles must be a list or numpy array, got {type(percentiles)}. "
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
