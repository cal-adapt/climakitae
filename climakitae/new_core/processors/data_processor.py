"""Data processing module for climakitae.""" ""
import datetime
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Union

import geopandas as gpd
import pandas as pd
import pyproj
import xarray as xr
from shapely.geometry import mapping

from climakitae.core.constants import UNSET
from climakitae.new_core.data_access import DataCatalog

# Registry to hold all registered processors
_PROCESSOR_REGISTRY = {}


def register_processor(key: str = UNSET) -> callable:
    """
    Decorator to register a processor class.

    Parameters
    ----------
    key : str, optional
        The key to register the processor under. If not provided, a key
        will be generated from the class name.

    Returns
    -------
    callable
        The decorator function that registers the processor class.
    """

    def decorator(cls):
        # If no key is provided, generate one from the class name
        processor_key = (
            key
            if key is not UNSET
            else "".join(
                ["_" + c.lower() if c.isupper() else c for c in cls.__name__]
            ).lstrip("_")
        )
        _PROCESSOR_REGISTRY[processor_key] = cls
        return cls

    return decorator


class DataProcessor(ABC):
    """
    Abstract base class for data processing.

    Each subclass must have an the following methods:
    - `execute`: Process the data.
    - `update_context`: Update the context with additional parameters.
    - `set_data_accessor`: Set the data accessor for the processor.

    Notes on building a processor:
    - The processor should only store the parameters needed for processing.
    - The processor should not store the data itself.
    - The processor should not throw exceptions. Instead, it should return the data
    passed to it and a warning message
    """

    @abstractmethod
    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        """
        Process raw data into the required format.

        Parameters
        ----------
        result : object
            data to be processed.
        context : dict
            Parameters for processing the data.

        Returns
        -------
        DataArray
            Processed data in the form of a DataArray.

        Raises
        ------
        ValueError
            If the data cannot be processed.
        """

    @abstractmethod
    def update_context(self, context: Dict[str, Any]):
        """
        Update the context with additional parameters.

        Parameters
        ----------
        context : dict
            Parameters for processing the data.

        Returns
        -------
        None
            Updates the context in place.
        """

    @abstractmethod
    def set_data_accessor(self, catalog: DataCatalog):
        """
        Set the data accessor for the processor.

        Parameters
        ----------
        catalog : DataCatalog
            Data catalog for accessing datasets.

        Returns
        -------
        None
            Sets the data accessor in place.
        """


@register_processor("get_variable")
class GetVariable(DataProcessor):
    """
    Get a specific variable from the data.

    This class is a placeholder for variable extraction logic.
    """

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        # Placeholder for variable extraction logic
        return result

    def update_context(self, context: Dict[str, Any]):
        # Placeholder for updating context
        pass

    def set_data_accessor(self, catalog: DataCatalog):
        # Placeholder for setting data accessor
        pass


@register_processor("convert_units")
class ConvertUnits(DataProcessor):
    """
    Convert units of the data.

    This class is a placeholder for unit conversion logic.
    """

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        # Placeholder for unit conversion logic
        return result

    def update_context(self, context: Dict[str, Any]):
        # Placeholder for updating context
        pass

    def set_data_accessor(self, catalog: DataCatalog):
        # Placeholder for setting data accessor
        pass


@register_processor("rename_variables")
class RenameVariables(DataProcessor):
    """
    Rename variables in the data to user-friendly names.

    This class is a placeholder for variable renaming logic.
    """

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        # Placeholder for variable renaming logic
        return result

    def update_context(self, context: Dict[str, Any]):
        # Placeholder for updating context
        pass

    def set_data_accessor(self, catalog: DataCatalog):
        # Placeholder for setting data accessor
        pass


@register_processor("apply_bias_correction")
class ApplyBiasCorrection(DataProcessor):
    """
    Apply bias correction to the data.

    This class is a placeholder for bias correction logic.
    """

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        # Placeholder for bias correction logic
        return result

    def update_context(self, context: Dict[str, Any]):
        # Placeholder for updating context
        pass

    def set_data_accessor(self, catalog: DataCatalog):
        # Placeholder for setting data accessor
        pass


@register_processor("filter_data")
class FilterData(DataProcessor):
    """
    Filter data based on certain criteria.

    This class is a placeholder for data filtering logic.
    """

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        # Placeholder for data filtering logic
        return result

    def update_context(self, context: Dict[str, Any]):
        # Placeholder for updating context
        pass

    def set_data_accessor(self, catalog: DataCatalog):
        # Placeholder for setting data accessor
        pass


@register_processor("apply_shape_file")
class ApplyShapeFile(DataProcessor):
    """
    Apply shapefile to the data.

    This class is a placeholder for shapefile application logic.
    """

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        # Placeholder for shapefile application logic
        return result

    def update_context(self, context: Dict[str, Any]):
        # Placeholder for updating context
        pass

    def set_data_accessor(self, catalog: DataCatalog):
        # Placeholder for setting data accessor
        pass


@register_processor("time_slice")
class TimeSlice(DataProcessor):
    """
    Subset data based on certain criteria.

    This class is a placeholder for data subsetting logic.
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
        match result:
            case xr.DataArray() | xr.Dataset():
                # Subset the data using the date range
                return result.sel(time=slice(self.value[0], self.value[1]))
            case dict():
                # Subset the data using the date range
                subset_data = {}
                for key, value in result.items():
                    subset_data[key] = value.sel(
                        time=slice(self.value[0], self.value[1])
                    )
                return subset_data
            case list():
                # Subset the data using the date range
                subset_data = []
                for value in result:
                    subset_data.append(
                        value.sel(time=slice(self.value[0], self.value[1]))
                    )
                return subset_data
            case tuple():
                # Subset the data using the date range
                subset_data = []
                for value in result:
                    subset_data.append(
                        value.sel(time=slice(self.value[0], self.value[1]))
                    )
                # convert to tuple
                return tuple(subset_data)
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


@register_processor("global_warming_level")
class GlobalWarmingLevel(DataProcessor):
    """
    Apply global warming level method to the data.

    This class is a placeholder for global warming level application logic.
    """

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        # Placeholder for global warming level application logic
        return result

    def update_context(self, context: Dict[str, Any]):
        # Placeholder for updating context
        pass

    def set_data_accessor(self, catalog: DataCatalog):
        # Placeholder for setting data accessor
        pass
