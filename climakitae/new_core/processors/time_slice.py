"""
Subset data on time
"""

import datetime
import os
import warnings
from typing import Any, Dict, Iterable, Union

import pandas as pd
import xarray as xr

from climakitae.new_core.data_access import DataCatalog
from climakitae.new_core.processors.data_processor import (
    _PROCESSOR_REGISTRY,  # looks unused but is used in the decorator
)
from climakitae.new_core.processors.data_processor import (
    DataProcessor,
    register_processor,
)


@register_processor("time_slice")
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

    def __init__(self, value):
        """
        Initialize the TimeSlice processor.

        Parameters
        ----------
        value : tuple(date-like, date-like)
            The value to subset the data by.
        """
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError(
                "Value must be a tuple of two date-like values."
            )  # TODO warning not error
        self.value = self._coerce_to_dates(value)

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
                return subset_data

            case xr.DataArray() | xr.Dataset():
                return result.sel(time=slice(self.value[0], self.value[1]))

            case Iterable():
                subset_data = []
                for value in result:
                    subset_data.append(
                        value.sel(time=slice(self.value[0], self.value[1]))
                    )
                # return as the same type as the input
                return type(result)(subset_data)
            case _:
                raise ValueError(  # TODO warning not error
                    f"""Invalid data type for subsetting. 
                    Expected xr.Dataset, dict, list, or tuple but got {type(result)}."""
                )

    def update_context(self, context: Dict[str, Any]):
        # Placeholder for updating context
        pass

    def set_data_accessor(self, catalog: DataCatalog):
        # Placeholder for setting data accessor
        pass

    @staticmethod
    def _coerce_to_dates(value: tuple) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Coerce the values to date-like objects.

        Parameters
        ----------
        value : tuple
            The value to coerce.

        Returns
        -------
        tuple
            The coerced values.
        """
        ret = []
        for x in value:
            match x:
                case str() | int() | float() | datetime.date() | datetime.datetime():
                    ret.append(pd.to_datetime(x))
                case pd.Timestamp():
                    ret.append(x)
                case pd.DatetimeIndex():
                    ret.append(x[0])
                case _:
                    raise ValueError(  # TODO warning not error
                        f"Invalid type {type(x)} for date coercion. Expected str, pd.Timestamp, pd.DatetimeIndex, datetime.date, or datetime.datetime."
                    )
        return tuple(ret)
