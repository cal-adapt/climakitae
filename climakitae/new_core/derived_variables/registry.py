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
    drop_dependencies : bool
        Whether to remove source variables after computing derived variable.
        Default: True (keep only the derived variable in the output).

    """

    name: str
    depends_on: List[str]
    description: str
    units: str
    func: Callable
    source: str = "builtin"
    drop_dependencies: bool = True


def _wrap_with_metadata_preservation(
    func: Callable,
    derived_var_name: str,
    source_vars: List[str],
    drop_dependencies: bool = True,
) -> Callable:
    """Wrap a derived variable function to preserve metadata and optionally drop dependencies.

    This wrapper detects when a new variable is added to the dataset and
    automatically copies spatial metadata (CRS, spatial_ref, grid_mapping)
    from the first available source variable. Optionally removes source variables
    after computation.

    Parameters
    ----------
    func : callable
        The original derived variable function.
    derived_var_name : str
        Name of the derived variable being computed.
    source_vars : list of str
        List of source variable names that the function depends on.
    drop_dependencies : bool, optional
        Whether to remove source variables after computation. Default: True.

    Returns
    -------
    callable
        Wrapped function that preserves spatial metadata and optionally drops dependencies.

    """
    from functools import wraps

    @wraps(func)
    def wrapped(ds):
        # Call the original function
        result = func(ds)

        # Find a source variable to copy metadata from
        source_var = None
        for var in source_vars:
            if var in result:
                source_var = var
                break

        # Preserve metadata if we found a source and the derived var exists
        if source_var and derived_var_name in result:
            logger.debug(
                "Preserving spatial metadata for '%s' from '%s'",
                derived_var_name,
                source_var,
            )
            preserve_spatial_metadata(result, derived_var_name, source_var)
        else:
            logger.warning(
                "Could not preserve metadata for '%s': source_var=%s, in_result=%s",
                derived_var_name,
                source_var,
                derived_var_name in result if result is not None else "result is None",
            )

        # Drop source variables only if the derived variable was actually added
        # and dropping was requested. This prevents accidentally removing source
        # variables when the derived function did not produce the expected output.
        if drop_dependencies and result is not None and derived_var_name in result.data_vars:
            vars_to_drop = [v for v in source_vars if v in result.data_vars]
            if vars_to_drop:
                logger.debug(
                    "Dropping dependency variables after computing '%s': %s",
                    derived_var_name,
                    vars_to_drop,
                )
                result = result.drop_vars(vars_to_drop)

        return result

    return wrapped


def preserve_spatial_metadata(ds, derived_var_name: str, source_var_name: str) -> None:
    """Copy spatial metadata from a source variable to a derived variable.

    This function ensures that derived variables retain necessary spatial
    metadata (CRS, spatial_ref, coordinates) from their source variables.
    This is critical for downstream operations like clipping that require
    CRS information.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing both source and derived variables.
    derived_var_name : str
        Name of the derived variable to update.
    source_var_name : str
        Name of the source variable to copy metadata from.

    Notes
    -----
    This function modifies the dataset in-place. It copies:
    - CRS via rioxarray if available
    - Lambert_Conformal or spatial_ref coordinate if present
    - grid_mapping attribute if present
    - Any coordinates present on the source but missing from derived

    Examples
    --------
    >>> ds["derived"] = ds["source_a"] - ds["source_b"]
    >>> preserve_spatial_metadata(ds, "derived", "source_a")

    """
    if derived_var_name not in ds or source_var_name not in ds:
        logger.debug(
            "Cannot preserve metadata: '%s' or '%s' not in dataset",
            derived_var_name,
            source_var_name,
        )
        return

    source = ds[source_var_name]

    # Collect attributes to copy (copy grid_mapping first as rioxarray needs it)
    attrs_to_copy = {}
    if "grid_mapping" in source.attrs:
        attrs_to_copy["grid_mapping"] = source.attrs["grid_mapping"]

    # Copy Lambert_Conformal or other CRS coordinate if present
    # WRF data typically uses Lambert_Conformal as the CRS coordinate
    crs_coord_names = ["Lambert_Conformal", "spatial_ref", "crs"]
    coords_to_add = {}
    for crs_coord in crs_coord_names:
        if crs_coord in ds.coords and crs_coord not in ds[derived_var_name].coords:
            coords_to_add[crs_coord] = ds.coords[crs_coord]
            logger.debug(
                "Will copy '%s' coordinate to '%s'", crs_coord, derived_var_name
            )

    # Apply coordinate changes (this creates a new DataArray, losing attrs)
    if coords_to_add:
        ds[derived_var_name] = ds[derived_var_name].assign_coords(coords_to_add)

    # Now copy attributes AFTER coordinate assignment (since assign_coords makes new obj)
    if attrs_to_copy:
        ds[derived_var_name].attrs.update(attrs_to_copy)
        logger.debug(
            "Copied grid_mapping='%s' to '%s'",
            attrs_to_copy.get("grid_mapping"),
            derived_var_name,
        )

    # Try to copy CRS using rioxarray
    try:
        import rioxarray  # noqa: F401

        # Refresh reference after coord assignment
        derived = ds[derived_var_name]

        # Check if source has CRS via rioxarray
        if hasattr(source, "rio") and source.rio.crs is not None:
            ds[derived_var_name].rio.write_crs(source.rio.crs, inplace=True)
            # Re-apply attrs since rio.write_crs wipes them out!
            if attrs_to_copy:
                ds[derived_var_name].attrs.update(attrs_to_copy)
            logger.debug(
                "Copied CRS %s from '%s' to '%s'",
                source.rio.crs,
                source_var_name,
                derived_var_name,
            )
        elif hasattr(derived, "rio"):
            # Try to get CRS from grid_mapping attribute
            grid_mapping_name = derived.attrs.get("grid_mapping")
            if grid_mapping_name and grid_mapping_name in ds.coords:
                # rioxarray should be able to parse this now
                try:
                    crs = derived.rio.crs
                    if crs is not None:
                        logger.debug(
                            "Derived variable '%s' has CRS from grid_mapping: %s",
                            derived_var_name,
                            crs,
                        )
                except Exception:
                    pass
    except ImportError:
        logger.debug("rioxarray not available for CRS handling")
    except Exception as e:
        logger.debug("Could not copy CRS via rioxarray: %s", e)

    # NOTE: We intentionally do NOT copy spatial coordinates (lat, lon, x, y) here.
    # When derived variables are computed via arithmetic (e.g., ds.t2max - ds.t2min),
    # xarray automatically propagates coordinates from the operands. Re-adding them
    # via assign_coords can corrupt the dimension structure, especially for 2D
    # coordinates like lat(y,x) and lon(y,x) in WRF data.
    # Only CRS-related metadata (Lambert_Conformal, grid_mapping attr) needs explicit
    # preservation, which is handled above.


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
    drop_dependencies: bool = True,
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
    drop_dependencies : bool, optional
        Whether to remove source variables from the output after computing the
        derived variable. Default: True (only return the derived variable).
        Set to False to keep source variables alongside the derived variable.

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

    >>> # Keep source variables alongside derived variable
    >>> @register_derived(
    ...     variable='temp_range',
    ...     query={'variable_id': ['tasmax', 'tasmin']},
    ...     description='Daily temperature range',
    ...     drop_dependencies=False  # Keep tasmax and tasmin
    ... )
    ... def calc_temp_range(ds):
    ...     ds['temp_range'] = ds.tasmax - ds.tasmin
    ...     return ds

    Notes
    -----
    The decorated function must:
    - Accept a single xarray.Dataset argument
    - Add the derived variable to the dataset
    - Return the modified dataset
    - Set appropriate attributes (units, long_name) on the new variable

    By default, source variables are dropped to reduce output size. Set
    drop_dependencies=False to keep them for downstream analysis.

    """

    def decorator(func: Callable) -> Callable:
        registry = get_registry()

        # Extract depends_on from query
        depends_on = query.get("variable_id", [])
        if isinstance(depends_on, str):
            depends_on = [depends_on]

        # Wrap function to automatically preserve spatial metadata
        # This must happen BEFORE storing in metadata so both intake-esm
        # and _apply_derived_variable use the same wrapped function
        wrapped_func = _wrap_with_metadata_preservation(
            func, variable, depends_on, drop_dependencies=drop_dependencies
        )

        # Store metadata with the WRAPPED function
        # This ensures _apply_derived_variable() also preserves spatial metadata
        _DERIVED_METADATA[variable] = DerivedVariableInfo(
            name=variable,
            depends_on=depends_on,
            description=description,
            units=units,
            func=wrapped_func,  # Use wrapped function, not original
            source=source,
            drop_dependencies=drop_dependencies,
        )

        # Register with intake-esm
        logger.info(
            "Registering derived variable '%s' depending on %s", variable, depends_on
        )
        registry.register(variable=variable, query=query)(wrapped_func)

        return func

    return decorator


def register_user_function(
    name: str,
    depends_on: List[str],
    func: Callable,
    description: str = "",
    units: str = "",
    query_extras: Optional[Dict[str, Any]] = None,
    drop_dependencies: bool = True,
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
    drop_dependencies : bool, optional
        Whether to remove source variables from the output after computing the
        derived variable. Default: True (only return the derived variable).
        Set to False to keep source variables alongside the derived variable.

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

    # Wrap function to automatically preserve spatial metadata
    # This must happen BEFORE storing in metadata
    wrapped_func = _wrap_with_metadata_preservation(
        func, name, depends_on, drop_dependencies=drop_dependencies
    )

    # Store metadata with the WRAPPED function
    _DERIVED_METADATA[name] = DerivedVariableInfo(
        name=name,
        depends_on=depends_on,
        description=description,
        units=units,
        func=wrapped_func,  # Use wrapped function, not original
        source="user",
        drop_dependencies=drop_dependencies,
    )

    # Register with intake-esm
    registry = get_registry()
    logger.info(
        "Registering user-defined derived variable '%s' depending on %s",
        name,
        depends_on,
    )
    registry.register(variable=name, query=query)(wrapped_func)


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
