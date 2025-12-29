"""Derived Variable Registry for ClimakitAE.

This module provides a singleton registry for derived climate variables that
integrates with intake-esm's DerivedVariableRegistry. It enables users to
define variables that are computed from other variables during data loading.

The registry follows the same patterns as the processor registry in
climakitae.new_core.processors.abc_data_processor.

Classes
-------
DerivedVariableInfo
    Dataclass containing metadata about a registered derived variable.

Functions
---------
get_registry
    Get the global intake-esm DerivedVariableRegistry singleton.
register_derived
    Decorator to register a derived variable function.
register_user_function
    Imperatively register a user-defined derived variable.
list_derived_variables
    List all registered derived variables with their metadata.

"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from intake_esm import DerivedVariableRegistry

# Module logger
logger = logging.getLogger(__name__)

# Singleton registry instance
_DERIVED_REGISTRY: Optional[DerivedVariableRegistry] = None

# Metadata storage for registered variables (intake-esm doesn't expose this well)
_DERIVED_METADATA: Dict[str, "DerivedVariableInfo"] = {}


@dataclass
class DerivedVariableInfo:
    """Metadata about a registered derived variable.

    Attributes
    ----------
    name : str
        The name of the derived variable (what users query).
    depends_on : list of str
        Variable IDs that this derived variable requires.
    description : str
        Human-readable description of what this variable represents.
    units : str
        Expected units of the derived variable.
    func : callable
        The function that computes the derived variable.
    source : str
        Where this variable was registered from ('builtin' or 'user').

    """

    name: str
    depends_on: List[str]
    description: str
    units: str
    func: Callable
    source: str = "builtin"


def get_registry() -> DerivedVariableRegistry:
    """Get the global derived variable registry singleton.

    Returns
    -------
    DerivedVariableRegistry
        The singleton intake-esm DerivedVariableRegistry instance.

    Notes
    -----
    The registry is lazily initialized on first access. This ensures that
    builtin derived variables are registered before the registry is used.

    Examples
    --------
    >>> registry = get_registry()
    >>> print(registry)
    DerivedVariableRegistry({...})

    """
    global _DERIVED_REGISTRY
    if _DERIVED_REGISTRY is None:
        logger.debug("Initializing DerivedVariableRegistry singleton")
        _DERIVED_REGISTRY = DerivedVariableRegistry()
    return _DERIVED_REGISTRY


def register_derived(
    variable: str,
    query: Dict[str, Any],
    description: str = "",
    units: str = "",
    source: str = "builtin",
) -> Callable:
    """Decorator to register a derived variable function.

    This decorator registers a function with the intake-esm DerivedVariableRegistry,
    enabling the variable to be queried directly from catalogs that have the
    registry attached.

    Parameters
    ----------
    variable : str
        The name of the derived variable. This is what users will query.
    query : dict
        Query constraints for finding source variables. Must include 'variable_id'
        with a list of required source variables. May include additional constraints
        like 'table_id', 'experiment_id', etc.
    description : str, optional
        Human-readable description of the derived variable.
    units : str, optional
        Expected units of the derived variable.
    source : str, optional
        Where this variable was registered from. Default is 'builtin'.

    Returns
    -------
    callable
        The decorator function.

    Examples
    --------
    >>> @register_derived(
    ...     variable='wind_speed',
    ...     query={'variable_id': ['u10', 'v10']},
    ...     description='Wind speed at 10m',
    ...     units='m/s'
    ... )
    ... def calc_wind_speed(ds):
    ...     import numpy as np
    ...     ds['wind_speed'] = np.sqrt(ds.u10**2 + ds.v10**2)
    ...     ds['wind_speed'].attrs = {'units': 'm/s', 'long_name': 'Wind Speed'}
    ...     return ds

    Notes
    -----
    The decorated function must:
    - Accept a single xarray.Dataset argument
    - Add the derived variable to the dataset
    - Return the modified dataset
    - Set appropriate attributes (units, long_name) on the new variable

    """

    def decorator(func: Callable) -> Callable:
        registry = get_registry()

        # Extract depends_on from query
        depends_on = query.get("variable_id", [])
        if isinstance(depends_on, str):
            depends_on = [depends_on]

        # Store metadata
        _DERIVED_METADATA[variable] = DerivedVariableInfo(
            name=variable,
            depends_on=depends_on,
            description=description,
            units=units,
            func=func,
            source=source,
        )

        # Register with intake-esm
        logger.info(
            "Registering derived variable '%s' depending on %s", variable, depends_on
        )
        registry.register(variable=variable, query=query)(func)

        return func

    return decorator


def register_user_function(
    name: str,
    depends_on: List[str],
    func: Callable,
    description: str = "",
    units: str = "",
    query_extras: Optional[Dict[str, Any]] = None,
) -> None:
    """Register a user-defined derived variable at runtime.

    This function allows users to register custom derived variables without
    using the decorator syntax. This is useful for dynamic registration or
    when working interactively.

    Parameters
    ----------
    name : str
        The name of the derived variable. This is what users will query.
    depends_on : list of str
        Variable IDs that this function requires (e.g., ['tasmax', 'tasmin']).
    func : callable
        Function that takes an xarray.Dataset and returns a modified Dataset
        with the new variable added.
    description : str, optional
        Human-readable description of the derived variable.
    units : str, optional
        Expected units of the derived variable.
    query_extras : dict, optional
        Additional query constraints beyond variable_id (e.g., table_id, experiment_id).

    Raises
    ------
    ValueError
        If name is empty or depends_on is empty.
    TypeError
        If func is not callable.

    Examples
    --------
    >>> def calc_temp_range(ds):
    ...     ds['temp_range'] = ds.tasmax - ds.tasmin
    ...     ds['temp_range'].attrs = {'units': 'K', 'long_name': 'Diurnal Range'}
    ...     return ds
    ...
    >>> register_user_function(
    ...     name='temp_range',
    ...     depends_on=['tasmax', 'tasmin'],
    ...     func=calc_temp_range,
    ...     description='Daily temperature range',
    ...     units='K'
    ... )
    ...
    >>> # Now query it directly
    >>> data = cd.catalog("cadcat").variable("temp_range").get()

    Notes
    -----
    Registration is permanent for the session. Once registered, the variable
    is available for all subsequent queries until the Python process ends.

    """
    # Validation
    if not name or not isinstance(name, str):
        raise ValueError("name must be a non-empty string")
    if not depends_on:
        raise ValueError("depends_on must be a non-empty list of variable IDs")
    if not callable(func):
        raise TypeError("func must be callable")

    # Build query
    query = {"variable_id": depends_on}
    if query_extras:
        query.update(query_extras)

    # Store metadata
    _DERIVED_METADATA[name] = DerivedVariableInfo(
        name=name,
        depends_on=depends_on,
        description=description,
        units=units,
        func=func,
        source="user",
    )

    # Register with intake-esm
    registry = get_registry()
    logger.info(
        "Registering user-defined derived variable '%s' depending on %s",
        name,
        depends_on,
    )
    registry.register(variable=name, query=query)(func)


def list_derived_variables() -> Dict[str, DerivedVariableInfo]:
    """List all registered derived variables with their metadata.

    Returns
    -------
    dict
        Dictionary mapping variable names to DerivedVariableInfo objects.

    Examples
    --------
    >>> derived_vars = list_derived_variables()
    >>> for name, info in derived_vars.items():
    ...     print(f"{name}: depends on {info.depends_on}")
    wind_speed: depends on ['u10', 'v10']
    relative_humidity: depends on ['t2', 'q2', 'psfc']

    """
    return _DERIVED_METADATA.copy()


def is_derived_variable(variable: str) -> bool:
    """Check if a variable name is a registered derived variable.

    Parameters
    ----------
    variable : str
        The variable name to check.

    Returns
    -------
    bool
        True if the variable is registered as a derived variable.

    """
    return variable in _DERIVED_METADATA


def get_derived_variable_info(variable: str) -> Optional[DerivedVariableInfo]:
    """Get metadata for a derived variable.

    Parameters
    ----------
    variable : str
        The variable name to look up.

    Returns
    -------
    DerivedVariableInfo or None
        Metadata for the variable, or None if not found.

    """
    return _DERIVED_METADATA.get(variable)
