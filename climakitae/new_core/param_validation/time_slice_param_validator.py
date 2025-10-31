"""
Validator for parameters provided to Warming Level Processor.
"""

from __future__ import annotations

import warnings
from typing import Any

from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)
from climakitae.new_core.param_validation.param_validation_tools import _coerce_to_dates


@register_processor_validator("time_slice")
def validate_time_slice_param(value: tuple[Any, Any], **kwargs) -> bool:
    """
    Validate the parameters provided to the time slice Processor.

    Parameters
    ----------
    value : tuple(date-like, date-like)
        The value to subset the data by. This should be a tuple of two
        date-like values.

    Returns
    -------
    bool
        True if all parameters are valid, False otherwise
    """
    if not isinstance(value, tuple) or len(value) != 2:
        warnings.warn(
            "\n\nTime Slice Processor expects a tuple of two date-like values. "
            "\nPlease check the configuration."
        )
        return False
    try:
        value = _coerce_to_dates(value)
    except ValueError as e:
        warnings.warn(
            f"\n\nInvalid date-like values provided: {e}. "
            "\nExpected a tuple of two date-like values."
        )
        return False

    return True  # All parameters are valid
