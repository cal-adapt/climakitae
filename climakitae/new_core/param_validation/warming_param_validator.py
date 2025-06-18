"""
Validator for parameters provided to Warming Level Processor.
"""

from __future__ import annotations

import warnings
from typing import Any

from climakitae.core.constants import UNSET
from climakitae.new_core.param_validation.abc_param_validation import (
    register_processor_validator,
)


@register_processor_validator("warming_level")
def validate_warming_level_param(
    value: dict[str, Any],
    **kwargs: Any,
) -> bool:
    """
    Validate the parameters provided to the Warming Level Processor.

    This function checks the value provided to the Warming Level Processor and ensures that it
    meets the expected criteria. Will raise a user warning and return false if the value
    is not valid.

    Parameters
    ----------
    value : Union[str, list, dict[str, Any]]
        The configuration dictionary to validate. Expected keys:
        - warming_levels : list[float]
            List of global warming levels in degrees C (e.g., [1.5, 2.0])
        - warming_level_months : list[int], optional
            List of months to include (1-12). Default: all months
        - warming_level_window : int, optional
            Number of years before and after the central year. Default: 15

    Returns
    -------
    bool
        True if all parameters are valid, False otherwise
    """
    if not _check_input_types(value):
        return False
    
    # now we have to check some more serious stuff
    query = kwargs.get("query", UNSET)
    if query is UNSET:
        warnings.warn(
            "\n\nWarming Level Processor requires a 'query' parameter. "
            "\nPlease check the configuration."
        )
        return False
    trajectories = get_trajectories(


def _check_input_types(
    value: dict[str, Any],
):
    if not isinstance(value, dict):
        warnings.warn(
            "\n\nWarming Level Processor expects a dictionary of parameters. "
            "\nPlease check the configuration."
        )
        return False

    wl = value.get("warming_levels", UNSET)
    if (
        wl is UNSET
        or not isinstance(wl, list)
        or not all(isinstance(x, (int, float)) for x in wl)
    ):
        warnings.warn(
            "\n\nInvalid 'warming_levels' parameter. "
            "\nExpected a list of global warming levels (e.g., [1.5, 2.0])."
        )
        return False

    wl_months = value.get("warming_level_months", UNSET)
    if wl_months is not UNSET:
        if not isinstance(wl_months, list) or not all(
            isinstance(x, int) and 1 <= x <= 12 for x in wl_months
        ):
            warnings.warn(
                "\n\nInvalid 'warming_level_months' parameter. "
                "\nExpected a list of months (1-12). Default: all months."
            )
            return False

    wl_window = value.get("warming_level_window", UNSET)
    if wl_window is not UNSET:
        if not isinstance(wl_window, int) or wl_window < 0:
            warnings.warn(
                "\n\nInvalid 'warming_level_window' parameter. "
                "\nExpected a non-negative integer (default: 15)."
            )
            return False

    return True
