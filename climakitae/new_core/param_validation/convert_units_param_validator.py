"""
Validator for parameters provided to Convert Units Processor.
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable

from climakitae.core.constants import UNIT_OPTIONS, UNSET
from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)

# All supported units (flattened from UNIT_OPTIONS)
ALL_SUPPORTED_UNITS = set()
for unit_list in UNIT_OPTIONS.values():
    ALL_SUPPORTED_UNITS.update(unit_list)


@register_processor_validator("convert_units")
def validate_convert_units_param(
    value: str | Iterable[str],
    **kwargs: Any,
) -> bool:
    """
    Validate the parameters provided to the Convert Units Processor.

    This function checks the value provided to the Convert Units Processor and ensures that it
    meets the expected criteria. Will raise a user warning and return false if the value
    is not valid.

    Parameters
    ----------
    value : str | Iterable[str]
        The unit(s) to convert to. Can be a single unit string or an iterable of unit strings.
        Valid units include temperature units (K, degC, degF), pressure units (Pa, hPa, mb, inHg),
        wind units (m/s, m s-1, mph, knots), precipitation units (mm, mm/d, mm/h, inches, inches/d, inches/h),
        moisture units (kg/kg, kg kg-1, g/kg, g kg-1), flux units (kg m-2 s-1), and relative humidity
        units ([0 to 100], fraction).

    Returns
    -------
    bool
        True if all parameters are valid, False otherwise
    """
    # kwargs unused but required for signature compatibility
    del kwargs

    if value is UNSET:
        # UNSET is valid - processor will not perform any conversion
        return True

    if not _check_input_types(value):
        return False

    if not _check_unit_validity(value):
        return False

    return True


def _check_input_types(value: str | Iterable[str]) -> bool:
    """
    Check if the input value has the correct type.

    Parameters
    ----------
    value : str | Iterable[str]
        The value to check.

    Returns
    -------
    bool
        True if the input type is valid, False otherwise.
    """
    if isinstance(value, str):
        return True

    warnings.warn(
        "\n\nConvert Units Processor expects a string. "
        f"\nReceived type: {type(value)}"
    )
    return False


def _check_unit_validity(value: str | Iterable[str]) -> bool:
    """
    Check if the provided unit(s) are supported.

    Parameters
    ----------
    value : str | Iterable[str]
        The unit(s) to validate.

    Returns
    -------
    bool
        True if all units are supported, False otherwise.
    """
    units_to_check = [value] if isinstance(value, str) else list(value)

    for unit in units_to_check:
        if unit not in ALL_SUPPORTED_UNITS:
            supported_units_str = ", ".join(sorted(ALL_SUPPORTED_UNITS))
            warnings.warn(
                f"\n\nUnsupported unit: '{unit}'. "
                f"\nSupported units are: {supported_units_str}"
            )
            return False

    return True
