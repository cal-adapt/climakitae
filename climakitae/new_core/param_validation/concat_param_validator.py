"""Validator for parameters provided to Concat Processor."""

from __future__ import annotations

import logging
from typing import Any

from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)


# Module logger
logger = logging.getLogger(__name__)


@register_processor_validator("concat")
def validate_concat_param(value: str, **kwargs: Any) -> bool:  # noqa: ARG001
    """Validate the parameters provided to the Concat Processor.

    Parameters
    ----------
    value : str
        The dimension name along which to concatenate datasets.
        Default: "sim"

    Returns
    -------
    bool
        True if all parameters are valid, False otherwise

    """
    logger.debug("validate_concat_param called with value: %s", value)

    if not isinstance(value, str):
        msg = "Concat Processor expects a string value for dimension name. Please check the configuration."
        logger.warning(msg, stacklevel=999)
        return False

    if not value.strip():
        msg = "Concat Processor dimension name cannot be empty. Please provide a valid dimension name."
        logger.warning(msg, stacklevel=999)
        return False

    return True  # All parameters are valid
