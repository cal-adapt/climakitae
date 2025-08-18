"""Template for a DataProcessor subclass in climakitae.

This module provides a template for implementing custom data processors that can be registered and used within the climakitae data processing pipeline. Processors are designed to transform, filter, or otherwise process xarray data objects in a modular and extensible way.

Classes
-------
Template : Example processor template for subsetting data.

"""

from typing import Any, Dict, Iterable, Union

import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)


@register_processor("template", priority=50)
class Template(DataProcessor):
    """Template for a DataProcessor.

    This class serves as a template for creating new data processors. It demonstrates the required methods and docstring style for consistency within the climakitae framework.

    Parameters
    ----------
    value : Iterable[Any]
        The value to subset the data by. Typically a tuple of two date-like values.

    Attributes
    ----------
    value : Iterable[Any]
        The value used for subsetting or transformation.
    name : str
        The name of the processor.

    Methods
    -------
    execute(result, context)
        Run the processor on the provided data.
    update_context(context)
        Update the context with information about the transformation.
    set_data_accessor(catalog)
        Set the data accessor for the processor (optional, for advanced use).

    """

    def __init__(self, value: Iterable[Any]):
        """Initialize the processor.

        Parameters
        ----------
        value : Iterable[Any]
            The value to subset the data by. Typically a tuple of two date-like values.

        """
        self.value = value
        self.name = "template"

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        """Run the processor on the provided data.

        Parameters
        ----------
        result : xr.Dataset or xr.DataArray or Iterable of these
            The data to be processed or sliced.
        context : dict
            The context for the processor. This is not used in this implementation but is included for consistency with the DataProcessor interface.

        Returns
        -------
        xr.Dataset, xr.DataArray, or Iterable of these
            The processed or sliced data. This can be a single Dataset/DataArray or an iterable of them.

        """

    def update_context(self, context: Dict[str, Any]):
        """Update the context with information about the transformation.

        Parameters
        ----------
        context : dict[str, Any]
            Parameters for processing the data. The context is updated in place.

        Returns
        -------
        None

        """

        if _NEW_ATTRS_KEY not in context:
            context[_NEW_ATTRS_KEY] = {}

        context[_NEW_ATTRS_KEY][
            self.name
        ] = f"""Process '{self.name}' applied to the data. Transformation was done using the following value: {self.value}."""

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
        # Placeholder for setting data accessor
        pass
