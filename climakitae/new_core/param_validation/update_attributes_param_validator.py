"""Validator for parameters provided to UpdateAttributes Processor."""

from __future__ import annotations

from typing import Any

from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)


@register_processor_validator("update_attributes")
def validate_update_attributes_param(value: Any, **kwargs: Any) -> bool:  # noqa: ARG001
    """Validate the parameters provided to the UpdateAttributes Processor.

    Parameters
    ----------
    value : Any
        The value to update the attributes with. Can be any type,
        including UNSET for default behavior.

    Returns
    -------
    bool
        True if all parameters are valid, False otherwise

    """
    # UpdateAttributes processor accepts any value type, including UNSET
    # This is a very permissive validator since the processor is designed
    # to handle various attribute update scenarios

    # The only case we might want to warn about is None, but even that
    # could be a valid use case for clearing attributes

    # We accept the value parameter but don't need to validate it specifically
    # since this processor is designed to be very flexible
    _ = value  # Acknowledge the parameter

    return True  # All parameters are valid
