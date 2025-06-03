"""
UpdateAttributes Processor definition.
"""

from typing import Any, Dict, Iterable, Union

import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)


@register_processor("update_attributes", priority=999)
class UpdateAttributes(DataProcessor):
    """
    Update attributes of the data.

    Adds new attributes to the data that describe the processing steps
    """

    def __init__(self, value: Any = UNSET):
        """
        Initialize the UpdateAttributes processor.

        Parameters
        ----------
        value : Any
            The value to update the attributes with.
        """
        self.value = value
        self.name = "update_attributes"

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        """
        Execute the UpdateAttributes processor.

        This method updates the attributes of the data based on the provided value.
        """
        if self.name not in context:
            self.update_context(context)

        match result:
            case dict():
                for key, item in result.items():
                    result[key].attrs = item.attrs | context[_NEW_ATTRS_KEY]
            case xr.Dataset() | xr.DataArray():
                result.attrs = result.attrs | context[_NEW_ATTRS_KEY]
            case list() | tuple():
                for i, item in enumerate(result):
                    result[i].attrs = item.attrs | context[_NEW_ATTRS_KEY]
            case _:
                raise TypeError(
                    "Result must be an xarray Dataset, DataArray, or iterable of them."
                )

        return result

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
        ] = f"""Process '{self.name}' applied to the data."""

    def set_data_accessor(self, catalog: DataCatalog):
        # Placeholder for setting data accessor
        pass
