"""
Merge DataProcessor
"""

from typing import Any, Dict, Iterable, List, Union

import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.new_core.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)


@register_processor("merge", priority=60)
class Merge(DataProcessor):
    """
    DataProcessor that merges multiple datasets based on their activity_id.

    This processor takes an iterable of xarray datasets or data arrays and merges
    them together, using the activity_id as a common identifier.

    Parameters
    ----------
    value : Any
        Not used in this processor, included for compatibility with the DataProcessor interface.

    Methods
    -------
    execute(result, context)
        Merges the input datasets based on activity_id.
    update_context(context)
        Updates the context with information about the merge operation.

    Notes
    -----
    All input datasets should have the 'activity_id' attribute.
    """

    def __init__(self, value: Any = None):
        """
        Initialize the merge processor.

        Parameters
        ----------
        value : Any, optional
            Not used in this processor, but maintained for compatibility.
        """
        self.value = value
        self.name = "merge"

    def execute(
        self,
        result: Iterable[Union[xr.Dataset, xr.DataArray]],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, List[xr.Dataset]]:
        """
        Merge multiple datasets based on activity_id.

        Parameters
        ----------
        result : Iterable[Union[xr.Dataset, xr.DataArray]]
            The datasets to be merged. Must be an iterable of xarray Datasets or DataArrays.

        context : dict
            The context for the processor.

        Returns
        -------
        Union[xr.Dataset, List[xr.Dataset]]
            If all datasets have the same activity_id, returns a single merged Dataset.
            Otherwise, returns a list of merged datasets grouped by activity_id.
        """
        if not isinstance(result, Iterable):
            # If we receive a single dataset, just return it
            return result

        # Group datasets by activity_id
        source_groups = {}
        for dataset in result:
            if not isinstance(dataset, (xr.Dataset, xr.DataArray)):
                continue

            # Extract activity_id from attributes
            source_id = dataset.attrs.get("source_id", "unknown")

            if source_id not in source_groups:
                source_groups[source_id] = []

            source_groups[source_id].append(dataset)

        # Merge datasets within each activity group
        merged_results = []
        for source_id, datasets in source_groups.items():
            if len(datasets) == 1:
                merged_results.append(datasets[0])
            else:
                # Merge datasets with the same activity_id
                merged = xr.merge(datasets)
                merged.attrs["source_id"] = source_id
                merged_results.append(merged)

        # If there's only one activity group, return just the dataset
        if len(merged_results) == 1:
            return merged_results[0]

        self.update_context(context)
        return merged_results

    def update_context(self, context: Dict[str, Any]):
        """
        Update the context with information about the merge transformation.

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
        ] = f"""Process '{self.name}' applied to the data.
        Datasets were merged based on their activity_id attribute."""

    def set_data_accessor(self, catalog: DataCatalog):
        # Placeholder for setting data accessor
        pass
