"""Derived Variables Module for ClimakitAE.

This module provides a registry system for derived climate variables that integrates
with intake-esm catalogs. Derived variables are computed on-the-fly from source
variables during data loading, enabling users to query variables that don't exist
in the raw data but can be computed from available variables.

Key Features
------------
- **Registry Pattern**: Global singleton registry for derived variable definitions
- **Decorator API**: Simple `@register_derived` decorator for defining new variables
- **Runtime Registration**: Users can register custom functions at runtime
- **Fluent Integration**: Works seamlessly with ClimateData's fluent interface

Architecture
------------
The derived variable system operates at the intake-esm catalog level, meaning
computations happen during `to_dataset_dict()` rather than as post-processing.
This is more efficient for simple derived variables and allows them to be
queried directly by name.

Example Usage
-------------
Using builtin derived variables:

    >>> from climakitae.new_core.user_interface import ClimateData
    >>> cd = ClimateData()
    >>> # Query wind_speed which is derived from u10 and v10
    >>> data = cd.catalog("cadcat").variable("wind_speed").get()

Registering custom derived variables:

    >>> from climakitae.new_core.derived_variables import register_user_function
    >>>
    >>> def calc_temp_range(ds):
    ...     ds['temp_range'] = ds.tasmax - ds.tasmin
    ...     ds['temp_range'].attrs = {'units': 'K', 'long_name': 'Diurnal Temperature Range'}
    ...     return ds
    >>>
    >>> register_user_function('temp_range', ['tasmax', 'tasmin'], calc_temp_range)
    >>>
    >>> # Now you can query it directly
    >>> data = cd.catalog("cadcat").variable("temp_range").get()

Using the decorator:

    >>> from climakitae.new_core.derived_variables import register_derived
    >>>
    >>> @register_derived(variable='my_index', query={'variable_id': ['t2', 'rh']})
    ... def calc_my_index(ds):
    ...     ds['my_index'] = ds.t2 * ds.rh / 100
    ...     return ds

See Also
--------
climakitae.tools.derived_variables : Lower-level functions for computing derived variables
climakitae.new_core.processors : Post-load processing (use for complex transformations)

"""

from climakitae.new_core.derived_variables.registry import (
    DerivedVariableInfo,
    get_registry,
    list_derived_variables,
    register_derived,
    register_user_function,
)

# Import builtin derived variables to register them
from climakitae.new_core.derived_variables import builtin  # noqa: F401

__all__ = [
    "get_registry",
    "register_derived",
    "register_user_function",
    "list_derived_variables",
    "DerivedVariableInfo",
]
