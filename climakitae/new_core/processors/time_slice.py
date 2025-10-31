"""
Subset data on time
"""

import warnings
from typing import Any, Dict, Iterable, Union

import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.param_validation.param_validation_tools import _coerce_to_dates
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)


@register_processor("time_slice", priority=100)
class TimeSlice(DataProcessor):
    """
    Slice data based on time.

    Parameters
    ----------
    value : tuple(date-like, date-like)
        The value to subset the data by. This should be a tuple of two
        date-like values.

    Methods
    -------
    _coerce_to_dates(value: tuple) -> tuple[pd.Timestamp, pd.Timestamp]
        Coerce the values to date-like objects.

    """

    def __init__(self, value: Iterable[Any]):
        """
        Initialize the TimeSlice processor.

        Parameters
        ----------
        value : Iterable(date-like, date-like)
            The value to subset the data by.
        """
        self.value = _coerce_to_dates(value)
        self.name = "time_slice"

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        """
        Run the time slicing operation on the data.

        Parameters
        ----------
        result : xr.Dataset | xr.DataArray | Iterable[xr.Dataset | xr.DataArray]
            The data to be sliced.

        context : dict
            The context for the processor. This is not used in this
            implementation but is included for consistency with the
            DataProcessor interface.

        Returns
        -------
        Union[xr.Dataset, xr.DataArray, Iterable[xr.Dataset | xr.DataArray]]
            The sliced data. This can be a single Dataset/DataArray or
            an iterable of them.
        """
        match result:
            case dict():  # most likely case at top
                subset_data = {}
                for key, value in result.items():
                    subset_data[key] = value.sel(
                        time=slice(self.value[0], self.value[1])
                    )
                self.update_context(context)
                return subset_data

            case xr.DataArray() | xr.Dataset():
                self.update_context(context)
                return result.sel(time=slice(self.value[0], self.value[1]))

            case list() | tuple():
                subset_data = []
                for value in result:
                    subset_data.append(
                        value.sel(time=slice(self.value[0], self.value[1]))
                    )
                # return as the same type as the input
                self.update_context(context)
                return type(result)(subset_data)
            case _:
                warnings.warn(
                    f"""Invalid data type for subsetting. 
                    Expected xr.Dataset, dict, list, or tuple but got {type(result)}."""
                )

    def update_context(self, context: Dict[str, Any]):
        """
        Update the context with information about the clipping operation, to be stored
                in the "new_attrs" attribute.

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
        ] = f"""Process '{self.name}' applied to the data. Slicing was done using the following value: {self.value}."""

    def set_data_accessor(self, catalog: DataCatalog):
        # Placeholder for setting data accessor
        pass
