"""Data processing module for climakitae.

This module defines the abstract base class for data processors, a registry system for processor classes, and example processor implementations. Processors are used to transform, filter, or otherwise process xarray data objects in a modular and extensible way.

Classes
-------
DataProcessor : Abstract base class for all data processors.
RenameVariables : Example processor for renaming variables.
ApplyBiasCorrection : Example processor for bias correction.
FilterData : Example processor for filtering data.

Functions
---------
register_processor : Decorator for registering processor classes.

"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Union

import xarray as xr

from climakitae.core.constants import UNSET
from climakitae.new_core.data_access.data_access import DataCatalog

# Registry to hold all registered processors
_PROCESSOR_REGISTRY = {}


def register_processor(
    key: str | object = UNSET, priority: int | object = UNSET
) -> Callable:
    """Decorator to register a processor class.

    Parameters
    ----------
    key : str, optional
        The key to register the processor under. If not provided, a key
        will be generated from the class name.
    priority : int, optional
        Optional priority for the processor. Lower values indicate higher priority.

    Returns
    -------
    callable
        The decorator function that registers the processor class.

    Examples
    --------
    @register_processor("my_processor")
    class MyProcessor(DataProcessor):
        ...

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
        _PROCESSOR_REGISTRY[processor_key] = (cls, priority)
        return cls

    return decorator


class DataProcessor(ABC):
    """Abstract base class for data processing.

    All data processors should inherit from this class and implement the required methods.

    Notes
    -----
    - Processors should only store parameters needed for processing, not the data itself.
    - Processors should not throw exceptions; instead, they should return the data and a warning message if needed.
    - All processors should update the context with information about how they modified the data.

    Methods
    -------
    execute(result, context)
        Process the data and return the result.
    update_context(context)
        Update the context with additional parameters.
    set_data_accessor(catalog)
        Set the data accessor for the processor.

    """

    @abstractmethod
    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        """Process raw data into the required format.

        Parameters
        ----------
        result : Dataset, DataArray, or iterable of these
            Data to be processed.
        context : dict
            Parameters for processing the data.

        Returns
        -------
        Dataset, DataArray, or iterable of these
            Processed data.

        Raises
        ------
        ValueError
            If the data cannot be processed.

        """

    @abstractmethod
    def update_context(self, context: Dict[str, Any]):
        """Update the context with additional parameters.

        Parameters
        ----------
        context : dict
            Parameters for processing the data.

        Returns
        -------
        None

        """

    @abstractmethod
    def set_data_accessor(self, catalog: DataCatalog):
        """Set the data accessor for the processor.

        Parameters
        ----------
        catalog : DataCatalog
            Data catalog for accessing datasets.

        Returns
        -------
        None

        """
