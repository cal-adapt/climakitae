"""Data processing module for climakitae.""" ""
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Union

import xarray as xr

from climakitae.core.constants import UNSET
from climakitae.new_core.data_access import DataCatalog

# Registry to hold all registered processors
_PROCESSOR_REGISTRY = {}


def register_processor(key=None):
    """Decorator to register a processor class."""

    def decorator(cls):
        # If no key is provided, generate one from the class name
        processor_key = key or "".join(
            ["_" + c.lower() if c.isupper() else c for c in cls.__name__]
        ).lstrip("_")
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


@register_processor("subset_data_space")
class SubsetDataSpace(DataProcessor):
    """
    Subset data based on certain criteria.

    This class is a placeholder for data subsetting logic.
    """

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        # Placeholder for data subsetting logic
        return result

    def update_context(self, context: Dict[str, Any]):
        # Placeholder for updating context
        pass

    def set_data_accessor(self, catalog: DataCatalog):
        # Placeholder for setting data accessor
        pass


@register_processor("subset_data_time")
class SubsetDataTime(DataProcessor):
    """
    Subset data based on certain criteria.

    This class is a placeholder for data subsetting logic.
    """

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        # Placeholder for data subsetting logic
        return result

    def update_context(self, context: Dict[str, Any]):
        # Placeholder for updating context
        pass

    def set_data_accessor(self, catalog: DataCatalog):
        # Placeholder for setting data accessor
        pass


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
