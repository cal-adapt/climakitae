"""
Validator for parameters provided to Convert to Local Time processor.
"""

from __future__ import annotations

import logging
from typing import Any

from climakitae.core.constants import UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)

# Module logger
logger = logging.getLogger(__name__)


@register_processor_validator("convert_to_local_time")
def validate_convert_to_local_time_param(
    value: Dict[str, Any], **kwargs: Any
) -> bool:  # noqa: ARG001
    """Validate the parameters provided to the ConvertToLocalTime Processor.

    Parameters
    ----------
    value : Union[str, list, dict[str, Any]]
        The configuration dictionary to validate. Expected keys:
        - convert : str
            The value to control leap day dropping behavior. Supported values:
            "yes": Convert time to local time
            "no" (default): Keep original timezone
        - drop_duplicate_times : str
            Set to "yes" to drop duplicate time stamps from daylight savings time.
            Default value is "no".

    Returns
    -------
    bool
        True if all parameters are valid, False otherwise

    """
    # Module logger
    logger = logging.getLogger(__name__)

    valid_values = ["yes", "no"]

    for setting in value:
        if not isinstance(value[setting], str):
            msg = (
                "\nConvertToLocalTime Processor expects string values. "
                "\nPlease check the configuration."
            )
            logger.warning(msg)
            return False

        if value[setting] not in valid_values:
            msg = (
                f"\n\nInvalid value '{value}' for ConvertToLocalTime Processor '{setting}' setting. "
                f"\nSupported values are: {valid_values}"
            )
            logger.warning(msg)
            return False

    return True  # All parameters are valid
