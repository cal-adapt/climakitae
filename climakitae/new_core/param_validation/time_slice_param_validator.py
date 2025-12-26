"""
Validator for parameters provided to Warming Level Processor.
"""

from __future__ import annotations

import logging
from typing import Any

from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)
from climakitae.new_core.param_validation.param_validation_tools import _coerce_to_dates

# Module logger
logger = logging.getLogger(__name__)


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
    logger.debug(
        "validate_time_slice_param called with value=%s kwargs=%s", value, kwargs
    )

    if not isinstance(value, tuple) or len(value) != 2:
        msg = "Time Slice Processor expects a tuple of two date-like values. Please check the configuration."
        logger.warning(msg)
        return False
    try:
        value = _coerce_to_dates(value)
    except ValueError as e:
        msg = f"Invalid date-like values provided: {e}. Expected a tuple of two date-like values."
        logger.warning(msg)
        return False

    return True  # All parameters are valid
