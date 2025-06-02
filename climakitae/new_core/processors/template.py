"""
DataProcessor Template
"""

from typing import Any, Dict, Iterable, Union

import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.new_core.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)


@register_processor("template", priority=50)
class Template(DataProcessor):
    """
    Template for a DataProcessor.

    Parameters
    ----------
    value : tuple(date-like, date-like)
        The value to subset the data by. This should be a tuple of two
        date-like values.

    Methods
    -------


    Notes
    -----

    """

    def __init__(self, value: Iterable[Any]):
        """
        Initialize the processor.

        Parameters
        ----------
        value : Iterable(date-like, date-like)
            The value to subset the data by.
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
        """
        Run the processor

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

    def update_context(self, context: Dict[str, Any]):
        """
        Update the context with information about the transformation.

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
        ] = f"""Process '{self.name}' applied to the data. Transformation was done using the following value: {self.value}."""

    def set_data_accessor(self, catalog: DataCatalog):
        # Placeholder for setting data accessor
        pass
