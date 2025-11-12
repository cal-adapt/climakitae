"""
DataProcessor for converting units of data.
"""

import warnings
from typing import Any, Dict, Iterable, Union

import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY, UNIT_OPTIONS, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)

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
    ("mm", "kg m-2 s-1"): lambda da: _handle_precipitation_to_flux(da),
    ("mm/d", "kg m-2 s-1"): lambda da: _handle_precipitation_to_flux(da),
    ("mm/h", "kg m-2 s-1"): lambda da: _handle_precipitation_to_flux(da),
    ("kg m-2 s-1", "mm"): lambda da: _handle_flux_to_precipitation(da, unit="mm"),
    ("kg m-2 s-1", "inches"): lambda da: _handle_flux_to_precipitation(
        da, unit="inches"
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
}


def _handle_precipitation_to_flux(da):
    """Convert precipitation (mm) to flux (kg m-2 s-1)"""
    result = da / 86400
    if da.attrs.get("frequency") == "monthly":
        da_name = da.name
        result = result / da["time"].dt.days_in_month
        result.name = da_name  # Preserve name
    return result


def _handle_flux_to_precipitation(da, unit="mm"):
    """Convert flux (kg m-2 s-1) to precipitation (mm or inches)"""
    if da.attrs.get("frequency") == "monthly":
        da_name = da.name
        result = da * da["time"].dt.days_in_month
        result.name = da_name  # Preserve name
    else:
        result = da

    result = result * 86400  # Convert to mm

    if unit == "inches":
        result = result / 25.4  # Convert mm to inches

    return result


@register_processor("convert_units", priority=750)
class ConvertUnits(DataProcessor):
    """
    Convert units of the data.

    This class tries to convert the units of the data to a user specified value.
    If the conversion is not possible, it raises a warning and returns the original data.

    Parameters
    ----------
    value : string
        The value to convert the units to. If not specified, the processor will
        not perform any conversion.

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

    def __init__(self, value: Union[str, object] = UNSET):
        """
        Initialize the ConvertUnits processor.

        Parameters
        ----------
        value : string
            The value to convert the units to.
        """
        self.value = value
        self.name = "convert_units"
        self.success = True

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        # Placeholder for unit conversion logic
        if self.value is UNSET:
            return result

        ret = result
        match result:
            case dict():
                # If result is a dictionary, convert each item
                ret = {k: self._convert_units(v, self.value) for k, v in result.items()}
            case xr.Dataset() | xr.DataArray():
                ret = self._convert_units(result, self.value)
            case list() | tuple():
                # If result is an iterable, convert each item
                ret = type(result)(
                    [self._convert_units(item, self.value) for item in result]
                )
            case _:
                warnings.warn()

        if self.success:
            # In this processor, it doesn't make sense to update unless the conversion was successful
            self.update_context(context)
        return ret

    def update_context(self, context: Dict[str, Any]):
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

    def set_data_accessor(self, catalog: DataCatalog):
        # Placeholder for setting data accessor
        pass

    def _convert_units(
        self, data: Union[xr.Dataset, xr.DataArray], value: str
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Convert the units of the data.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            The data to convert.
        value : str
            The value to convert the units to.

        Returns
        -------
        Union[xr.Dataset, xr.DataArray]
            The converted data.
        """
        try:
            var = list(data.data_vars.keys())[0]
            units_from = data.data_vars[var].attrs[
                "units"
            ]  # Trying to get an error if the units attribute does not exist
        except (KeyError, IndexError):
            warnings.warn(
                (
                    "WARNING ::: You've encountered a bug in the code. "
                    " This variable does not have identifiable native units. The data"
                    " for this variable will need to have a 'units'"
                    " attribute added in the catalog."
                    " Please report this issue to the developers at"
                    " https://github.com/cal-adapt/climakitae/issues/new/choose "
                )
            )
            self.success = False
            return data

        valid_units = UNIT_OPTIONS.get(units_from, None)
        if valid_units is None:
            warnings.warn(
                (
                    "WARNING ::: You've encountered a bug in the code. "
                    f" There are no valid unit conversions implemented for {units_from}. "
                    " Please report this issue to the developers at"
                    " https://github.com/cal-adapt/climakitae/issues/new/choose "
                )
            )
            self.success = False
            return data

        match value:
            case str():
                if value not in valid_units:
                    warnings.warn(
                        (
                            f"WARNING ::: The selected units {value} are not valid for {units_from}."
                        )
                    )
                    self.success = False
                    return data

                # Perform the actual conversion - Fix: assign to the dataset directly
                converted_var = UNIT_CONVERSIONS.get(
                    (units_from, value), lambda da: da
                )(data.data_vars[var])
                # Update the units attribute
                converted_var.attrs["units"] = value
                # Assign back to the dataset
                data = data.assign({var: converted_var})
            case _:
                warnings.warn(
                    (
                        "WARNING ::: The provided value is not the correct type. "
                        f"Expected str or Iterable[str], but got {type(value)}."
                    )
                )
                self.success = False
                return data

        return data
