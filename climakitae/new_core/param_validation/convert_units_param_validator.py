"""
Validator for parameters provided to Convert Units Processor.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from climakitae.core.constants import UNIT_OPTIONS, UNSET
from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)

# Module logger
logger = logging.getLogger(__name__)

# All supported units (flattened from UNIT_OPTIONS)
ALL_SUPPORTED_UNITS = set()
for unit_list in UNIT_OPTIONS.values():
    ALL_SUPPORTED_UNITS.update(unit_list)


@register_processor_validator("convert_units")
def validate_convert_units_param(
    value: Any,
    **kwargs: Any,
) -> bool:
    """
    Validate the parameters provided to the Convert Units Processor.

    This function checks the value provided to the Convert Units Processor and ensures that it
    meets the expected criteria. Will raise a user warning and return false if the value
    is not valid.

    Parameters
    ----------
    value : str or mapping
        A shared target unit, or ``{"by_variable": {variable_id: unit}}``.
        Mapping keys must be variables requested by ``query["variable_id"]``.
        Valid units include temperature units (K, degC, degF), pressure units (Pa, hPa, mb, inHg),
        wind units (m/s, m s-1, mph, knots), precipitation units (mm, mm/d, mm/h, inches, inches/d, inches/h),
        moisture units (kg/kg, kg kg-1, g/kg, g kg-1), flux units (kg m-2 s-1), and relative humidity
        units ([0 to 100], fraction).

    Returns
    -------
    bool
        True if all parameters are valid, False otherwise
    """
    if value is UNSET:
        # UNSET is valid - processor will not perform any conversion
        return True

    if not _check_input_types(value):
        return False

    if not _check_targets_are_selected(value, kwargs.get("query")):
        return False

    if not _check_unit_validity(value):
        return False

    return True


def _check_input_types(value: Any) -> bool:
    """
    Check if the input value has the correct type.

    Parameters
    ----------
    value : Any
        The shared unit string or per-variable unit mapping to check.

    Returns
    -------
    bool
        True if the input type is valid, False otherwise.
    """
    if isinstance(value, str):
        return True

    if isinstance(value, Mapping):
        targets = value.get("by_variable")
        if (
            set(value) != {"by_variable"}
            or not isinstance(targets, Mapping)
            or not targets
        ):
            logger.warning(
                "Convert Units Processor expects a string or a non-empty "
                "{'by_variable': {variable: unit}} mapping."
            )
            return False

        if any(not isinstance(name, str) or not name.strip() for name in targets):
            logger.warning(
                "The 'by_variable' mapping must use non-empty variable names."
            )
            return False
        return True

    logger.warning(
        "Convert Units Processor expects a string or a 'by_variable' mapping. "
        "Received type: %s",
        type(value),
    )
    return False


def _check_targets_are_selected(value: Any, query: Any) -> bool:
    """
    Check that per-variable unit targets name requested variables.

    Parameters
    ----------
    value : Any
        The shared unit string or per-variable unit mapping to check.
    query : Any
        The full user query. Targets are only checked when it provides a
        ``variable_id`` string or sequence.

    Returns
    -------
    bool
        True if every target is a requested variable or no selection is known,
        False otherwise.
    """
    if not isinstance(value, Mapping) or not isinstance(query, Mapping):
        return True

    selected_variables = query.get("variable_id", UNSET)
    if isinstance(selected_variables, str):
        selected_variables = {selected_variables}
    elif isinstance(selected_variables, (list, tuple, set)):
        selected_variables = set(selected_variables)
    else:
        return True

    unknown_variables = set(value["by_variable"]) - selected_variables
    if unknown_variables:
        logger.warning(
            "Unit conversion targets are not present in the data: %s",
            sorted(unknown_variables),
        )
        return False
    return True


def _check_unit_validity(value: Any) -> bool:
    """
    Check if the provided unit(s) are supported.

    Parameters
    ----------
    value : Any
        The unit target or targets to validate.

    Returns
    -------
    bool
        True if all units are supported, False otherwise.
    """
    if isinstance(value, str):
        units_to_check = [value]
    elif isinstance(value, Mapping):
        units_to_check = list(value["by_variable"].values())
    else:
        units_to_check = list(value)

    for unit in units_to_check:
        if not isinstance(unit, str) or unit not in ALL_SUPPORTED_UNITS:
            supported_units_str = ", ".join(sorted(ALL_SUPPORTED_UNITS))
            logger.warning(
                "Unsupported unit: '%s'. Supported units are: %s",
                unit,
                supported_units_str,
            )
            return False

    return True
