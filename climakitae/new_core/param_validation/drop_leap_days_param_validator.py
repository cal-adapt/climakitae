"""
Validator for parameters provided to Drop Leap Days Processor.
"""

from __future__ import annotations

import logging
from typing import Any

from climakitae.core.constants import UNSET
from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)

# Module logger
logger = logging.getLogger(__name__)


@register_processor_validator("drop_leap_days")
def validate_drop_leap_days_param(
    value: bool,
    **kwargs: Any,
) -> bool:
    """
    Validate the parameters provided to the Drop Leap Days Processor.

    This function checks the value provided to the Drop Leap Days Processor
    and ensures that it meets the expected criteria. Will raise a user warning
    and return False if the value is not valid.

    Parameters
    ----------
    value : bool
        Whether to drop leap days. Must be a boolean value (True or False).

    Returns
    -------
    bool
        True if the parameter is valid, False otherwise.
    """
    # kwargs unused but required for signature compatibility
    del kwargs

    if value is UNSET:
        # UNSET is valid - processor will use default behavior
        return True

    if not _check_input_type(value):
        return False

    return True


def _check_input_type(value: Any) -> bool:
    """
    Check if the input value is a boolean.

    Parameters
    ----------
    value : Any
        The value to check.

    Returns
    -------
    bool
        True if the input type is valid (boolean), False otherwise.
    """
    if isinstance(value, bool):
        return True

    logger.warning(
        "Drop Leap Days Processor expects a boolean (True or False). Received type: %s",
        type(value).__name__,
    )
    return False
