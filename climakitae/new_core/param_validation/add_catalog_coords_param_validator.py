"""Validator for parameters provided to AddCatalogCoords Processor."""

from __future__ import annotations

import logging
from typing import Any

from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)


# Module logger
logger = logging.getLogger(__name__)


@register_processor_validator("add_catalog_coords")
def validate_add_catalog_coords_param(value: Any, **kwargs: Any) -> bool:  # noqa: ARG001
    """Validate the parameters provided to the AddCatalogCoords Processor.

    The AddCatalogCoords processor does not require any parameters - it
    automatically adds network_id as a coordinate based on station_id from
    the context.

    Parameters
    ----------
    value : Any
        The value is not used by this processor (typically UNSET).

    Returns
    -------
    bool
        Always returns True since no validation is needed.

    """
    logger.debug("validate_add_catalog_coords_param called with value: %s", value)
    return True  # No validation needed - processor uses context
