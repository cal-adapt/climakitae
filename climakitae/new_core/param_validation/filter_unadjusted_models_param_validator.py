"""Validator for parameters provided to FilterunadjustedModels Processor."""

from __future__ import annotations

import logging
from typing import Any

from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)


@register_processor_validator("filter_unadjusted_models")
def validate_filter_unadjusted_models_param(
    value: str, **kwargs: Any
) -> bool:  # noqa: ARG001
    """Validate the parameters provided to the FilterUnadjustedModels Processor.

    Parameters
    ----------
    value : str
        The value to control filtering behavior. Supported values:
        "yes" (default): Filter out unadjusted models
        "no": Include unadjusted models

    Returns
    -------
    bool
        True if all parameters are valid, False otherwise

    """
    # Module logger
    logger = logging.getLogger(__name__)

    if not isinstance(value, str):
        msg = (
            "\n\nFilterunadjustedModels Processor expects a string value. "
            "\nPlease check the configuration."
        )
        logger.warning(msg)
        return False

    valid_values = ["yes", "no"]

    if value not in valid_values:
        msg = (
            f"\n\nInvalid value '{value}' for FilterUnadjustedModels Processor. "
            f"\nSupported values are: {valid_values}"
        )
        logger.warning(msg)
        return False

    return True  # All parameters are valid
