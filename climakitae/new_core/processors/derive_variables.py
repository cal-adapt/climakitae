"""Calculate registered derived variables after spatial subsetting.

Validation plans derived variables as their catalog dependencies, so
``DataCatalog.get_data`` returns only dependency fields. This processor
runs after ``clip`` and calculates the requested derived variables on the
smaller, clipped data.
"""

import logging
from typing import Any, Dict, Iterable, List, Union

import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.derived_variables.registry import (
    DerivedVariableInfo,
    list_derived_variables,
    preserve_spatial_metadata,
)
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)

logger = logging.getLogger(__name__)


def _requested_variables(context: Dict[str, Any]) -> List[str]:
    """Return the user-requested variable names as a list."""
    variable_id = context.get("variable_id", UNSET)
    if variable_id is UNSET or variable_id is None:
        return []
    if isinstance(variable_id, str):
        return [variable_id]
    return list(variable_id)


def requested_derived_variables(variable_id: Any) -> List[str]:
    """Return the registered derived variables in ``variable_id``.

    Parameters
    ----------
    variable_id : str, list of str, or UNSET
        The user-requested variable or variables.

    Returns
    -------
    list of str
        Requested names that are registered derived variables, in request
        order.

    """
    requested = _requested_variables({"variable_id": variable_id})
    if not requested:
        return []
    registry = list_derived_variables()
    return [name for name in requested if name in registry]


@register_processor("derive_variables", priority=3)
class DeriveVariables(DataProcessor):
    """Calculate requested derived variables from their dependency fields.

    This is an internal processor. Validators add it automatically when the
    query requests a registered derived variable. It runs after ``clip``
    (priority 2) so derived functions see only the clipped grid, and before
    unit conversion, warming levels, and concatenation.

    Parameters
    ----------
    value : UNSET, optional
        Not used. The processor reads the requested variables from
        ``context["variable_id"]``.

    Methods
    -------
    execute(result, context)
        Calculate the derived variables for each dataset.
    update_context(context)
        Record the calculated variables in the dataset attributes.
    set_data_accessor(catalog)
        Not used.

    """

    def __init__(self, value: Any = UNSET):
        self.value = value
        self.name = "derive_variables"
        self._derived: List[str] = []

    def execute(
        self,
        result: Union[xr.Dataset, xr.DataArray, Iterable[xr.Dataset]],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[xr.Dataset]]:
        """Calculate the requested derived variables.

        Parameters
        ----------
        result : xr.Dataset, dict, list, or tuple
            Data that contains the dependency fields.
        context : dict
            Processing context. ``context["variable_id"]`` holds the
            user-requested variables.

        Returns
        -------
        xr.Dataset, dict, list, or tuple
            Data with the requested variables first, followed by any
            dependencies kept by functions registered with
            ``drop_dependencies=False``.

        Raises
        ------
        ValueError
            If a dataset lacks a dependency, or a derived function does not
            return its registered output name.

        """
        requested = _requested_variables(context)
        registry = list_derived_variables()
        self._derived = [name for name in requested if name in registry]
        if not self._derived:
            return result

        match result:
            case dict():
                processed = {
                    key: self._derive(value, requested, registry, key)
                    for key, value in result.items()
                }
            case xr.Dataset():
                processed = self._derive(result, requested, registry, "dataset")
            case list() | tuple():
                processed = type(result)(
                    self._derive(value, requested, registry, str(i))
                    for i, value in enumerate(result)
                )
            case _:
                logger.warning(
                    "DeriveVariables expected a Dataset, dict, list, or tuple "
                    "but got %s. Returning data unchanged.",
                    type(result).__name__,
                )
                return result

        self.update_context(context)
        return processed

    def _derive(
        self,
        data: Any,
        requested: List[str],
        registry: Dict[str, DerivedVariableInfo],
        key: str,
    ) -> Any:
        """Calculate the derived variables for one dataset."""
        if not isinstance(data, xr.Dataset):
            return data

        output = data
        kept_dependencies: List[str] = []
        for name in self._derived:
            info = registry[name]
            missing = [dep for dep in info.depends_on if dep not in output.data_vars]
            if missing:
                raise ValueError(
                    f"Dataset {key!r} cannot calculate derived variable {name!r}: "
                    f"missing dependencies {missing}. "
                    f"Available variables: {list(output.data_vars)}."
                )

            logger.info("Calculating derived variable '%s' for %s", name, key)
            # Shallow copy so drop_dependencies cleanup cannot remove fields
            # that other requested outputs still need.
            computed = info.func(output.copy(deep=False))
            if name not in computed.data_vars:
                raise ValueError(
                    f"Derived function for {name!r} did not return its registered "
                    f"output name. Returned {list(computed.data_vars)}."
                )
            output = output.assign(
                {
                    var: computed[var]
                    for var in computed.data_vars
                    if var not in output.data_vars
                }
            )
            self._ensure_spatial_metadata(output, name, info.depends_on)
            if not info.drop_dependencies:
                kept_dependencies.extend(info.depends_on)

        missing_outputs = [var for var in requested if var not in output.data_vars]
        if missing_outputs:
            raise ValueError(
                f"Dataset {key!r} is missing requested output variables "
                f"{missing_outputs}. Returned {list(output.data_vars)}."
            )
        keep = list(dict.fromkeys(requested + kept_dependencies))
        return output[keep]

    @staticmethod
    def _ensure_spatial_metadata(
        ds: xr.Dataset, name: str, depends_on: List[str]
    ) -> None:
        """Copy spatial metadata from a dependency if the output lacks it."""
        da = ds[name]
        if da.attrs.get("grid_mapping"):
            return
        try:
            if da.rio.crs is not None:
                return
        except Exception:  # pylint: disable=broad-except
            pass
        for source in depends_on:
            if source in ds.data_vars:
                try:
                    preserve_spatial_metadata(ds, name, source)
                except Exception:  # pylint: disable=broad-except
                    logger.debug(
                        "Could not copy spatial metadata to '%s' from '%s'",
                        name,
                        source,
                        exc_info=True,
                    )
                return

    def update_context(self, context: Dict[str, Any]) -> None:
        """Record the calculated derived variables in the context.

        Parameters
        ----------
        context : dict
            Processing context. Updated in place.

        """
        context.setdefault(_NEW_ATTRS_KEY, {})
        context[_NEW_ATTRS_KEY][self.name] = (
            "Derived variables calculated after spatial subsetting: "
            f"{', '.join(self._derived)}."
        )

    def set_data_accessor(self, catalog: DataCatalog) -> None:
        """Set the data accessor (not used)."""
