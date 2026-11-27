"""Parameter validator for derived variables.

This module provides validation for derived variable parameters and
supports the inline registration pattern via the ClimateData interface.

"""

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def validate_derived_variable_params(
    name: str,
    depends_on: List[str],
    func: Callable,
    query_extras: Optional[Dict[str, Any]] = None,
) -> bool:
    """Validate parameters for a derived variable registration.

    Parameters
    ----------
    name : str
        The name of the derived variable.
    depends_on : list of str
        Variable IDs that this function requires.
    func : callable
        Function that computes the derived variable.
    query_extras : dict, optional
        Additional query constraints.

    Returns
    -------
    bool
        True if parameters are valid, False otherwise.

    Warns
    -----
    UserWarning
        If any parameters are invalid.

    """
    valid = True

    # Validate name
    if not name or not isinstance(name, str):
        warnings.warn(
            "Derived variable name must be a non-empty string",
            UserWarning,
            stacklevel=3,
        )
        valid = False

    if name and not name.replace("_", "").isalnum():
        warnings.warn(
            f"Derived variable name '{name}' should only contain alphanumeric "
            "characters and underscores",
            UserWarning,
            stacklevel=3,
        )
        # This is a warning, not a hard failure

    # Validate depends_on
    if not depends_on:
        warnings.warn(
            "depends_on must be a non-empty list of variable IDs",
            UserWarning,
            stacklevel=3,
        )
        valid = False
    elif not isinstance(depends_on, (list, tuple)):
        warnings.warn(
            f"depends_on must be a list or tuple, got {type(depends_on).__name__}",
            UserWarning,
            stacklevel=3,
        )
        valid = False
    else:
        for var in depends_on:
            if not isinstance(var, str) or not var.strip():
                warnings.warn(
                    f"Each variable in depends_on must be a non-empty string, got: {var!r}",
                    UserWarning,
                    stacklevel=3,
                )
                valid = False
                break

    # Validate func
    if not callable(func):
        warnings.warn(
            f"func must be callable, got {type(func).__name__}",
            UserWarning,
            stacklevel=3,
        )
        valid = False

    # Validate query_extras
    if query_extras is not None and not isinstance(query_extras, dict):
        warnings.warn(
            f"query_extras must be a dict or None, got {type(query_extras).__name__}",
            UserWarning,
            stacklevel=3,
        )
        valid = False

    return valid
