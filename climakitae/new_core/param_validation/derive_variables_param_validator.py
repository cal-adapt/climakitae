"""Validator for the internal DeriveVariables processor."""

from __future__ import annotations

import logging
from typing import Any

from climakitae.core.constants import UNSET
from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)

logger = logging.getLogger(__name__)


@register_processor_validator("derive_variables")
def validate_derive_variables_param(value: Any, **kwargs: Any) -> bool:  # noqa: ARG001
    """Validate the value for the internal DeriveVariables processor.

    Validators add ``derive_variables`` automatically when a query requests a
    registered derived variable. The processor takes no configuration.

    Parameters
    ----------
    value : Any
        Processor value. Only ``UNSET`` is valid.

    Returns
    -------
    bool
        True if the value is ``UNSET``, False otherwise.

    """
    if value is UNSET:
        return True
    logger.warning(
        "\n\nThe derive_variables processor is internal and takes no value. "
        "\nRequest a derived variable with .variable() instead."
    )
    return False
