"""Validator for parameters provided to Concat Processor."""

from __future__ import annotations

import warnings
from typing import Any

from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)


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
    if not isinstance(value, str):
        warnings.warn(
            "\n\nConcat Processor expects a string value for dimension name. "
            "\nPlease check the configuration.",
            UserWarning,
            stacklevel=999,
        )
        return False

    if not value.strip():
        warnings.warn(
            "\n\nConcat Processor dimension name cannot be empty. "
            "\nPlease provide a valid dimension name.",
            UserWarning,
            stacklevel=999,
        )
        return False

    return True  # All parameters are valid
