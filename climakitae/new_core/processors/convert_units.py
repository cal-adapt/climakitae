"""
DataProcessor for converting units of data.
"""

import logging
from collections.abc import Mapping
from typing import Any, Dict, Iterable, Union

import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY, UNIT_OPTIONS, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)

logger = logging.getLogger(__name__)

UNIT_CONVERSIONS = {
    # Identity conversion
    ("*", "*"): lambda da: da,
    # Initial hPa to Pa conversion
    ("hPa", "Pa"): lambda da: da / 100.0,
    ("hPa", "mb"): lambda da: da / 100.0,
    ("hPa", "inHg"): lambda da: da * 0.0295300,
    # Precipitation units
    ("mm", "inches"): lambda da: da / 25.4,
    ("mm/d", "inches/d"): lambda da: da / 25.4,
    ("mm/h", "inches/h"): lambda da: da / 25.4,
    ("mm", "kg m-2 s-1"): lambda da, freq=None: _handle_precipitation_to_flux(
        da, frequency=freq
    ),
    ("mm/d", "kg m-2 s-1"): lambda da, freq=None: _handle_precipitation_to_flux(
        da, frequency=freq
    ),
    ("mm/h", "kg m-2 s-1"): lambda da, freq=None: _handle_precipitation_to_flux(
        da, frequency=freq
    ),
    ("kg m-2 s-1", "mm"): lambda da, freq=None: _handle_flux_to_precipitation(
        da, unit="mm", frequency=freq
    ),
    ("kg m-2 s-1", "inches"): lambda da, freq=None: _handle_flux_to_precipitation(
        da, unit="inches", frequency=freq
    ),
    # Moisture ratio units
    ("kg/kg", "g/kg"): lambda da: da * 1000,
    ("kg kg-1", "g kg-1"): lambda da: da * 1000,
    ("g/kg", "kg/kg"): lambda da: da / 1000,
    # Pressure units
    ("Pa", "hPa"): lambda da: da / 100.0,
    ("Pa", "mb"): lambda da: da / 100.0,
    ("Pa", "inHg"): lambda da: da * 0.000295300,
    # Wind units
    ("m/s", "knots"): lambda da: da * 1.9438445,
    ("m/s", "mph"): lambda da: da * 2.236936,
    ("m s-1", "knots"): lambda da: da * 1.9438445,
    ("m s-1", "mph"): lambda da: da * 2.236936,
    # Temperature units
    ("K", "degC"): lambda da: da - 273.15,
    ("K", "degF"): lambda da: (1.8 * (da - 273.15)) + 32,
    ("degC", "K"): lambda da: da + 273.15,
    ("degC", "degF"): lambda da: (1.8 * da) + 32,
    ("degF", "degC"): lambda da: (da - 32) / 1.8,
    ("degF", "K"): lambda da: ((da - 32) / 1.8) + 273.15,
    # Relative humidity
    ("[0 to 100]", "fraction"): lambda da: da / 100.0,
    ("%", "fraction"): lambda da: da / 100.0,
}


def _handle_precipitation_to_flux(da, frequency=None):
    """Convert precipitation (mm) to flux (kg m-2 s-1)"""
    result = da / 86400
    if frequency == "mon":
        da_name = da.name
        result = result / da["time"].dt.days_in_month
        result.name = da_name  # Preserve name
    return result


def _handle_flux_to_precipitation(da, unit="mm", frequency=None):
    """Convert flux (kg m-2 s-1) to precipitation (mm or inches)"""
    if frequency == "mon":
        da_name = da.name
        result = da * da["time"].dt.days_in_month
        result.name = da_name  # Preserve name
    else:
        result = da

    result = result * 86400  # Convert to mm

    if unit == "inches":
        result = result / 25.4  # Convert mm to inches

    return result


@register_processor("convert_units", priority=5)
class ConvertUnits(DataProcessor):
    """
    Convert units of the data.

    This class tries to convert the units of the data to a user specified value.
    If the conversion is not possible, it raises a warning and returns the original data.

    Parameters
    ----------
    value : string or mapping
        A shared target unit, a ``{"by_variable": {name: unit}}`` mapping, or
        UNSET to skip conversion.

    Methods
    -------
    __init__(value: Union[str, Iterable, object] = UNSET)
    execute(
        result: Union[xr.Dataset, xr.DataArray,
        Iterable[Union[xr.Dataset, xr.DataArray]]],
        context: Dict[str, Any]
        ) -> Union[xr.Dataset, xr.DataArray,
        Convert units of the data in result based on the specified value.
    update_context(context: Dict[str, Any])
        Update the context with information about the unit conversion operation.
    set_data_accessor(catalog: DataCatalog)
        Set the data accessor for the processor.
    _convert_units(data: Union[xr.Dataset, xr.DataArray], value: str | Iterable[str])
        Convert the units of the data based on the specified value.
    """

    def __init__(self, value: Union[str, Mapping[str, Any], object] = UNSET):
        """
        Initialize the ConvertUnits processor.

        Parameters
        ----------
        value : string or mapping
            A shared target unit or a ``{"by_variable": {name: unit}}`` mapping.
        """
        self.value = value
        self.name = "convert_units"
        self.success = True
        logger.debug(
            "ConvertUnits processor initialized with target_units=%s",
            value if value is not UNSET else "<not set>",
        )

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        logger.debug(
            "ConvertUnits.execute called with target_units=%s, result_type=%s",
            self.value,
            type(result).__name__,
        )

        self.success = True
        if self.value is UNSET:
            logger.debug("No target units specified, returning data unchanged")
            return result

        if isinstance(self.value, Mapping):
            targets = self.value.get("by_variable")
            if (
                set(self.value) != {"by_variable"}
                or not isinstance(targets, Mapping)
                or not targets
            ):
                logger.warning(
                    "Convert Units Processor expects a string or a non-empty "
                    "{'by_variable': {variable: unit}} mapping."
                )
                self.success = False
                return result

            unknown_variables = set(targets) - self._get_data_variable_names(result)
            if unknown_variables:
                logger.warning(
                    "Unit conversion targets are not present in the data: %s",
                    sorted(unknown_variables),
                )
                self.success = False
                return result

        selected_variables = context.get("variable_id", UNSET)
        if selected_variables is UNSET:
            selected_variables = context.get("_catalog_variable_ids", UNSET)
        if isinstance(selected_variables, str):
            selected_variables = {selected_variables}
        elif isinstance(selected_variables, (list, tuple, set)):
            selected_variables = set(selected_variables)
        else:
            selected_variables = None

        ret = result
        match result:
            case dict():
                # If result is a dictionary, convert each item
                logger.debug(
                    "Converting units for %d datasets in dictionary", len(result)
                )
                ret = {
                    k: self._convert_units(v, self.value, selected_variables)
                    for k, v in result.items()
                }
            case xr.Dataset() | xr.DataArray():
                ret = self._convert_units(result, self.value, selected_variables)
            case list() | tuple():
                # If result is an iterable, convert each item
                logger.debug(
                    "Converting units for %d datasets in iterable", len(result)
                )
                ret = type(result)(
                    [
                        self._convert_units(item, self.value, selected_variables)
                        for item in result
                    ]
                )
            case _:
                logger.warning(
                    "Unexpected result type %s in ConvertUnits processor",
                    type(result).__name__,
                )

        if not self.success:
            return result

        if self.success:
            # In this processor, it doesn't make sense to update unless the conversion was successful
            logger.info("Units successfully converted to '%s'", self.value)
            self.update_context(context)
        return ret

    @staticmethod
    def _get_data_variable_names(data: Any) -> set[str]:
        """Return the names of all data variables in supported result containers.

        Parameters
        ----------
        data : Any
            An xarray object or a nested dictionary, list, or tuple of xarray
            objects.

        Returns
        -------
        set of str
            Names of data variables available for conversion.
        """
        match data:
            case xr.Dataset():
                return set(data.data_vars)
            case xr.DataArray():
                return {data.name} if data.name is not None else set()
            case dict():
                return set().union(
                    *(
                        ConvertUnits._get_data_variable_names(item)
                        for item in data.values()
                    )
                )
            case list() | tuple():
                return set().union(
                    *(ConvertUnits._get_data_variable_names(item) for item in data)
                )
            case _:
                return set()

    def update_context(self, context: Dict[str, Any]) -> None:
        """
        Update the context with information about the unit conversion operation, to be
        stored in the "new_attrs" attribute.

        Parameters
        ----------
        context : dict[str, Any]
            Parameters for processing the data.

        Note
        ----
        The context is updated in place. This method does not return anything.
        """

        if _NEW_ATTRS_KEY not in context:
            context[_NEW_ATTRS_KEY] = {}

        context[_NEW_ATTRS_KEY][
            self.name
        ] = f"""Process '{self.name}' applied to the data. Units were converted to the following: {self.value}."""

    def set_data_accessor(self, catalog: DataCatalog) -> None:
        # Placeholder for setting data accessor
        pass

    def _convert_units(
        self,
        data: Union[xr.Dataset, xr.DataArray],
        value: Union[str, Mapping[str, Any]],
        selected_variables: set[str] | None = None,
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Convert the units of the data.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            The data to convert.
        value : str or mapping
            A shared target unit or a ``{"by_variable": {name: unit}}`` mapping.
        selected_variables : set of str, optional
            Requested measurement variables. Other data variables are treated as
            support fields and left unchanged.

        Returns
        -------
        Union[xr.Dataset, xr.DataArray]
            The converted data.
        """
        if isinstance(value, str):
            if isinstance(data, xr.DataArray):
                targets = {data.name: value} if data.name is not None else {}
            else:
                grid_mapping_variables = {
                    variable.attrs.get("grid_mapping")
                    for variable in data.data_vars.values()
                }
                targets = {
                    name: value
                    for name, variable in data.data_vars.items()
                    if "units" in variable.attrs
                    and name != "landmask"
                    and name not in grid_mapping_variables
                    and (selected_variables is None or name in selected_variables)
                    and not {
                        "grid_mapping_name",
                        "spatial_ref",
                        "crs_wkt",
                    }.intersection(variable.attrs)
                }
        elif (
            isinstance(value, Mapping)
            and set(value) == {"by_variable"}
            and isinstance(value["by_variable"], Mapping)
        ):
            targets = value["by_variable"]
        else:
            logger.warning(
                "The provided value is not the correct type. Expected a string "
                "or a 'by_variable' mapping, but got %s.",
                type(value).__name__,
            )
            self.success = False
            return data

        if isinstance(data, xr.DataArray):
            if data.name not in targets:
                return data
            variables = {data.name: data}
        else:
            variables = {name: data[name] for name in targets if name in data.data_vars}

        if not variables:
            logger.warning(
                "No climate data variables with identifiable units were selected "
                "for conversion."
            )
            self.success = False
            return data

        frequency = data.attrs.get("frequency")
        converted_variables = {}
        for name, variable in variables.items():
            target_unit = targets[name]
            if not isinstance(target_unit, str):
                logger.warning(
                    "Target units for variable '%s' must be a string; "
                    "no variables were converted.",
                    name,
                )
                self.success = False
                return data

            units_from = variable.attrs.get("units")
            if units_from is None:
                logger.warning(
                    "This variable does not have identifiable native units "
                    "(variable '%s'); no variables were converted.",
                    name,
                )
                self.success = False
                return data

            valid_units = UNIT_OPTIONS.get(units_from)
            if valid_units is None:
                logger.warning(
                    "No valid unit conversions implemented for '%s' on variable "
                    "'%s'; no variables were converted.",
                    units_from,
                    name,
                )
                self.success = False
                return data

            if target_unit not in valid_units:
                logger.warning(
                    "The selected units '%s' are not valid for variable '%s' "
                    "with source units '%s'. Valid options: %s; no variables "
                    "were converted.",
                    target_unit,
                    name,
                    units_from,
                    valid_units,
                )
                self.success = False
                return data

            converter = UNIT_CONVERSIONS.get((units_from, target_unit))
            if converter is None and units_from != target_unit:
                logger.warning(
                    "Conversion from '%s' to '%s' is not implemented for "
                    "variable '%s'; no variables were converted.",
                    units_from,
                    target_unit,
                    name,
                )
                self.success = False
                return data

            if converter is None:
                converted_variable = variable.copy(deep=False)
            elif "kg m-2 s-1" in (units_from, target_unit):
                converted_variable = converter(variable, freq=frequency)
            else:
                converted_variable = converter(variable)

            converted_variable = converted_variable.copy(deep=False)
            converted_variable.attrs = dict(variable.attrs)
            converted_variable.attrs["units"] = target_unit
            converted_variables[name] = converted_variable
            logger.debug(
                "Prepared conversion for variable '%s': %s -> %s",
                name,
                units_from,
                target_unit,
            )

        if isinstance(data, xr.DataArray):
            return converted_variables[data.name]
        return data.assign(converted_variables)
