"""
Concat DataProcessor
"""

import warnings
from typing import Any, Dict, Iterable, List, Union

import xarray as xr

from climakitae.core.constants import _NEW_ATTRS_KEY, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)


# concatenation processor in the pre-processing chain
@register_processor("concat", priority=50)
class Concat(DataProcessor):
    """
    DataProcessor that concatenates multiple datasets along a new "sim" dimension.

    This processor takes an iterable of xarray datasets or data arrays and concatenates
    them along a new "sim" dimension using their source_id values. This is useful
    for creating ensemble datasets from multiple climate models.

    Parameters
    ----------
    value : Any
        Optional configuration for the concatenation process.
        Can specify a dimension name other than "sim".

    Methods
    -------
    execute(result, context)
        Concatenates the input datasets along a new "sim" dimension.
    update_context(context)
        Updates the context with information about the concatenation operation.

    Notes
    -----
    All input datasets should have the 'source_id' attribute.
    """

    def __init__(self, value: str = "sim"):
        """
        Initialize the concat processor.

        Parameters
        ----------
        value : str, optional
            Optional dimension name to use instead of "sim".
            Defaults to "sim".
        """
        self.dim_name = value if isinstance(value, str) else "sim"
        self.name = "concat"
        self.catalog = None
        self.needs_catalog = True

    def execute(
        self,
        result: xr.Dataset | xr.DataArray | Iterable[Union[xr.Dataset, xr.DataArray]],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        """
        Concatenate multiple datasets along a new dimension named after source_id.

        Parameters
        ----------
        result : Iterable[Union[xr.Dataset, xr.DataArray]]
            The datasets to be concatenated. Must be an iterable of xarray Datasets or DataArrays.

        context : dict
            The context for the processor.

        Returns
        -------
        Union[xr.Dataset, xr.DataArray]
            A single dataset with a new dimension that contains all input datasets.
        """
        if not isinstance(result, Iterable):
            # If we receive a single dataset, just return it
            return result

        datasets_to_concat = []
        concat_attr = (
            [
                "installation",
                "institution_id",
                "source_id",
                "experiment_id",
                "member_id",
            ]
            if self.catalog
            and getattr(self.catalog, "catalog_key", None) == "renewables"
            else [
                "intake_esm_attrs:activity_id",
                "intake_esm_attrs:institution_id",
                "intake_esm_attrs:source_id",
                "intake_esm_attrs:experiment_id",
                "intake_esm_attrs:member_id",
            ]
        )
        unknown_attr = "unknown"
        attr_ids = []

        print(len(result), "datasets to concatenate")

        match result:
            case dict():
                for key, dataset in result.items():
                    if not isinstance(dataset, (xr.Dataset, xr.DataArray)):
                        continue

                    # Extract source_id from attributes
                    attr_id = "_".join(
                        [
                            dataset.attrs.get(concat_attr, unknown_attr)
                            for concat_attr in concat_attr
                            if dataset.attrs.get(concat_attr, unknown_attr)
                            != unknown_attr
                        ]
                    )
                    attr_id = attr_id.replace(
                        " ", ""
                    )  # Replace spaces with empty string
                    attr_id = attr_id.lower()

                    if "member_id" in dataset.dims:
                        # If member_id is present, append it to simulation name if not already there
                        member_ids = dataset.member_id.values
                        datasets_for_member = []

                        for member_id in member_ids:
                            member_str = str(member_id)
                            # Check if member_id is already in the attr_id
                            if member_str not in attr_id:
                                current_attr_id = f"{attr_id}_{member_str}"
                            else:
                                current_attr_id = attr_id

                            # Select this specific member
                            member_dataset = dataset.sel(member_id=member_id).drop_vars(
                                "member_id"
                            )
                            member_dataset = member_dataset.expand_dims(
                                {self.dim_name: [current_attr_id]}
                            )
                            datasets_for_member.append(member_dataset)
                            attr_ids.append(current_attr_id)

                        # Add all member datasets to the list
                        datasets_to_concat.extend(datasets_for_member)
                    elif key.split(".")[-1][0] == "r":
                        # the key comes with a member_id, so we need to handle it
                        attr_ids.append(key.replace(".", "_"))
                        dataset = dataset.expand_dims(
                            {self.dim_name: [key.replace(".", "_")]}
                        )
                        datasets_to_concat.append(dataset)

                    else:
                        attr_ids.append(attr_id)
                        # Add sim dimension to the dataset
                        dataset = dataset.expand_dims({self.dim_name: [attr_id]})
                        datasets_to_concat.append(dataset)

            case _:
                for dataset in result:
                    if not isinstance(dataset, (xr.Dataset, xr.DataArray)):
                        # skip items that are not xarray datasets or data arrays
                        continue

                    # Extract source_id from attributes
                    attr_id = "_".join(
                        [
                            dataset.attrs.get(concat_attr, unknown_attr)
                            for concat_attr in concat_attr
                        ]
                    )
                    attr_id = attr_id.replace(
                        " ", ""
                    )  # Replace spaces with empty string
                    attr_id = attr_id.lower()

                    if "member_id" in dataset.dims:
                        # If member_id is present, append it to simulation name if not already there
                        member_ids = dataset.member_id.values

                        for member_id in member_ids:
                            member_str = str(member_id)
                            # Check if member_id is already in the attr_id
                            if member_str not in attr_id:
                                current_attr_id = f"{attr_id}_{member_str}"
                            else:
                                current_attr_id = attr_id

                            # Select this specific member
                            member_dataset = dataset.sel(member_id=member_id).drop_vars(
                                "member_id"
                            )
                            member_dataset = member_dataset.expand_dims(
                                {self.dim_name: [current_attr_id]}
                            )
                            datasets_to_concat.append(member_dataset)
                            attr_ids.append(current_attr_id)
                    else:
                        attr_ids.append(attr_id)
                        # Add sim dimension to the dataset
                        dataset = dataset.expand_dims({self.dim_name: [attr_id]})
                        datasets_to_concat.append(dataset)

        if not datasets_to_concat:
            return result  # Return original if no valid datasets

        # Concatenate all datasets along the sim dimension
        try:
            concatenated = xr.concat(datasets_to_concat, dim=self.dim_name)
        except ValueError as e:
            warnings.warn(
                f"Failed to concatenate datasets along '{self.dim_name}' dimension: {e}",
            )
            # Print dimensions of each dataset for debugging
            for i, dataset in enumerate(datasets_to_concat):
                print(f"Dataset {i} dimensions: {list(dataset.dims.keys())}")
            return result

        print(f"Concatenated datasets along '{self.dim_name}' dimension.")

        self.update_context(context, attr_ids)
        print(concatenated)
        return concatenated

    def update_context(
        self, context: Dict[str, Any], source_ids: List[str] | object = UNSET
    ):
        """
        Update the context with information about the concatenation transformation.

        Parameters
        ----------
        context : dict[str, Any]
            Parameters for processing the data.
        source_ids : List[str], optional
            List of source_ids that were concatenated

        Note
        ----
        The context is updated in place. This method does not return anything.
        """
        if _NEW_ATTRS_KEY not in context:
            context[_NEW_ATTRS_KEY] = {}

        # If source_ids is not provided, use the default value
        if source_ids is UNSET:
            source_ids = []
        elif not isinstance(source_ids, list):
            source_ids = [source_ids]

        source_info = (
            f"source_ids: {', '.join(str(sid) for sid in source_ids)}"
            if source_ids
            else ""
        )

        context[_NEW_ATTRS_KEY][
            self.name
        ] = f"""Process '{self.name}' applied to the data. Multiple datasets were concatenated along a new '{self.dim_name}' dimension.
        {source_info}"""

    def set_data_accessor(self, catalog: DataCatalog):
        """
        Set the data catalog for this processor.

        Parameters
        ----------
        catalog : DataCatalog
            The data catalog to be used by this processor.
        """
        self.catalog = catalog
