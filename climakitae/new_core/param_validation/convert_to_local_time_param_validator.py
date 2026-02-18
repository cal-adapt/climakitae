"""
Validator for parameters provided to Convert to Local Time processor.
"""

from __future__ import annotations

import logging
from typing import Any

from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)

# Module logger
logger = logging.getLogger(__name__)


@register_processor_validator("convert_to_local_time")
def validate_convert_to_local_time_param(
    value: str, **kwargs: Any
) -> bool:  # noqa: ARG001
    """Validate the parameters provided to the ConvertToLocalTime Processor.

    Parameters
    ----------
    value : str
        The value to control leap day dropping behavior. Supported values:
        "yes": Convert time to local time
        "no" (default): Keep original timezone

    Returns
    -------
    bool
        True if all parameters are valid, False otherwise

    """
    # Module logger
    logger = logging.getLogger(__name__)

    if not isinstance(value, str):
        msg = (
            "\nConvertToLocalTime Processor expects a string value. "
            "\nPlease check the configuration."
        )
        logger.warning(msg)
        return False

    valid_values = ["yes", "no"]

    if value not in valid_values:
        msg = (
            f"\n\nInvalid value '{value}' for ConvertToLocalTime Processor. "
            f"\nSupported values are: {valid_values}"
        )
        logger.warning(msg)
        return False

    return True  # All parameters are valid
