"""
Validator for parameters provided to Warming Level Processor.
"""

from __future__ import annotations

import logging
from typing import Any

from climakitae.core.constants import UNSET
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
    if isinstance(value, dict):
        time_slice = value.get("dates", None)
        season_filter = value.get("seasons", UNSET)
    else:
        time_slice = value
        season_filter = UNSET

    if not isinstance(time_slice, tuple) or len(time_slice) != 2:
        msg = "Time Slice Processor expects a tuple of two date-like values. Please check the configuration."
        logger.warning(msg)
        return False

    if season_filter is not UNSET:
        msg = (
            "\nIf provided, 'seasons' parameter must be a list of season names or a single season name. "
            "(e.g., ['DJF', 'MAM', 'JJA', 'SON']). Please check the configuration."
        )
        if isinstance(season_filter, (list, tuple)):
            if not all(
                isinstance(season, str) and season in ["DJF", "MAM", "JJA", "SON"]
                for season in season_filter
            ):
                logger.warning(msg)
                return False
        if isinstance(season_filter, str):
            if season_filter not in ["DJF", "MAM", "JJA", "SON"]:
                logger.warning(msg)
                return False

    try:
        value = _coerce_to_dates(time_slice)
    except ValueError as e:
        msg = f"Invalid date-like values provided: {e}. Expected a tuple of two date-like values."
        logger.warning(msg)
        return False
    return True  # All parameters are valid
