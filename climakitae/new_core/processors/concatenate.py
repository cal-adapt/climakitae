"""Concat DataProcessor"""

import logging
from typing import Any, Dict, Iterable, List, Union

import xarray as xr

from climakitae.core.constants import (_NEW_ATTRS_KEY, CATALOG_REN_ENERGY_GEN,
                                       UNSET)
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor, register_processor)
from climakitae.new_core.processors.processor_utils import extend_time_domain

# Module logger
logger = logging.getLogger(__name__)


# concatenation processor in the pre-processing chain
@register_processor("concat", priority=50)
class Concat(DataProcessor):
    """DataProcessor that concatenates multiple datasets along a new "sim" dimension.

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

    def __init__(self, value: str = "time"):
        """Initialize the concat processor.

        Parameters
        ----------
        value : str, optional
            Optional dimension name to use instead of "sim".
            Defaults to "sim".

        """
        self.dim_name = value if isinstance(value, str) else "time"
        self._original_dim_name = self.dim_name  # Track original dimension name
        self.name = "concat"
        self.catalog = None
        self.needs_catalog = True
        # Log initialization
        logger.debug("Concat processor initialized with dim_name=%s", self.dim_name)

    def execute(
        self,
        result: Union[
            xr.Dataset,
            xr.DataArray,
            Dict[str, Union[xr.Dataset, xr.DataArray]],
            Iterable[Union[xr.Dataset, xr.DataArray]],
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray]:
        """Concatenate multiple datasets along a specified dimension.

        Routes to appropriate concatenation method based on catalog type.

        Parameters
        ----------
        result : Union[xr.Dataset, xr.DataArray, Dict[str, Union[xr.Dataset, xr.DataArray]], Iterable[Union[xr.Dataset, xr.DataArray]]]
            The datasets to be concatenated. Must be an iterable of xarray Datasets or DataArrays.

        context : dict
            The context for the processor.

        Returns
        -------
        Union[xr.Dataset, xr.DataArray]
            A single dataset with concatenated data.

        """
        logger.debug(
            "Concat.execute called with dim_name=%s result_type=%s",
            self.dim_name,
            type(result).__name__,
        )

        if isinstance(result, (xr.Dataset, xr.DataArray)):
            # If we receive a single dataset, just return it
            logger.debug(
                "Concat.execute received single dataset/dataarray, returning as-is"
            )
            return result

        # Route to appropriate concat method based on catalog
        # Check context for catalog type (more reliable than catalog object attribute)
        catalog_type = context.get("catalog", context.get("_catalog_key"))
        if catalog_type == "hdp":
            return self._execute_hdp_concat(result, context)
        else:
            return self._execute_gridded_concat(result, context)

    def _execute_hdp_concat(
        self,
        result: Union[
            Dict[str, Union[xr.Dataset, xr.DataArray]],
            Iterable[Union[xr.Dataset, xr.DataArray]],
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray]:
        """Concatenate HDP station data along station dimension.

        Simpler concatenation for HDP catalog - just concatenates along
        the station dimension without complex attribute extraction.

        Parameters
        ----------
        result : Union[Dict[str, Union[xr.Dataset, xr.DataArray]], Iterable[Union[xr.Dataset, xr.DataArray]]]
            The datasets to be concatenated.
        context : dict
            The context for the processor.

        Returns
        -------
        Union[xr.Dataset, xr.DataArray]
            A single dataset with concatenated station data.

        """
        logger.debug("Using HDP concatenation logic")
        datasets_to_concat = []
        station_ids = []

        # Convert result to list of datasets
        if isinstance(result, dict):
            datasets = list(result.values())
        else:
            datasets = list(result)

        # Simple concatenation along station_id dimension
        # (station already renamed to station_id in data_access.get_data)
        for dataset in datasets:
            if not isinstance(dataset, (xr.Dataset, xr.DataArray)):
                continue

            datasets_to_concat.append(dataset)

            # Collect station IDs for context
            station_vals = dataset.station_id.values
            if hasattr(station_vals, "__iter__") and not isinstance(station_vals, str):
                station_ids.extend([str(s) for s in station_vals])
            else:
                station_ids.append(str(station_vals))

        if not datasets_to_concat:
            raise ValueError("No valid datasets found for HDP concatenation")

        # Simple concatenation along station_id dimension
        concatenated = xr.concat(
            datasets_to_concat,
            dim="station_id",
            join="outer",
        )

        logger.info("Concatenated HDP datasets along station_id dimension.")
        self.update_context(context, station_ids)
        return concatenated

    def _execute_gridded_concat(
        self,
        result: Union[
            Dict[str, Union[xr.Dataset, xr.DataArray]],
            Iterable[Union[xr.Dataset, xr.DataArray]],
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray]:
        """Concatenate gridded climate data with time domain extension.

        Complex concatenation for data/renewables catalogs - handles time domain
        extension, member IDs, and attribute extraction.

        Parameters
        ----------
        result : Union[Dict[str, Union[xr.Dataset, xr.DataArray]], Iterable[Union[xr.Dataset, xr.DataArray]]]
            The datasets to be concatenated.
        context : dict
            The context for the processor.

        Returns
        -------
        Union[xr.Dataset, xr.DataArray]
            A single dataset with concatenated data.

        """
        logger.debug("Using gridded data concatenation logic")

        # Special handling for time dimension concatenation
        if self.dim_name == "time" and isinstance(result, dict):
            # Handle time domain extension for dictionaries
            result = extend_time_domain(result)  # type: ignore
            # After extending time domain, switch to standard sim concatenation
            self.dim_name = "sim"
            logger.info("Time-domain extension applied; switching concat dim to 'sim'")

        datasets_to_concat = []
        concat_attrs = (
            [
                "installation",
                "institution_id",
                "source_id",
                "experiment_id",
                "member_id",
            ]
            if self.catalog
            and getattr(self.catalog, "catalog_key", None) == CATALOG_REN_ENERGY_GEN
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
        match result:
            case dict():
                for key, dataset in result.items():
                    if not isinstance(dataset, (xr.Dataset, xr.DataArray)):
                        continue

                    # Make sure the time dimension is aligned on the same timestamps
                    dataset = self._align_time_dim(dataset, context)

                    # Extract source_id from attributes
                    attr_id = "_".join(
                        [
                            dataset.attrs.get(attr, unknown_attr)
                            for attr in concat_attrs
                            if dataset.attrs.get(attr, unknown_attr) != unknown_attr
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

                    # Make sure the time dimension is aligned on the same timestamps
                    dataset = self._align_time_dim(dataset, context)

                    # Extract source_id from attributes
                    attr_id = "_".join(
                        [dataset.attrs.get(attr, unknown_attr) for attr in concat_attrs]
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
            # If no valid datasets to concatenate, return the first valid dataset
            if isinstance(result, dict):
                for dataset in result.values():
                    if isinstance(dataset, (xr.Dataset, xr.DataArray)):
                        return dataset
            # If still no valid dataset found, raise an error
            logger.debug(
                "No valid datasets found for concatenation. Sample result: %s", result
            )
            raise ValueError("No valid datasets found for concatenation")

        # Concatenate all datasets along the sim dimension
        try:
            # Use optimized settings for faster concatenation
            # compat="override" skips expensive coordinate equality checks
            # coords="minimal" avoids duplicating coordinates
            concatenated = xr.concat(
                datasets_to_concat,
                dim=self.dim_name,
                data_vars="minimal",
                coords="minimal",
                compat="override",
                join="outer",
            )
        except ValueError as e:
            # Log dimensions of each dataset for debugging
            for i, dataset in enumerate(datasets_to_concat):
                logger.debug("Dataset %d dimensions: %s", i, list(dataset.dims.keys()))
                intake_key = dataset.attrs.get("intake_esm_dataset_key", "<unknown>")
                logger.debug("Dataset:\n%s", intake_key)
            msg = (
                f"\nFailed to concatenate datasets along '{self.dim_name}' dimension: {e}"
                "\nThis can happen when mixing datasets with different projections"
                "\nsuch as WRF and LOCA2 or WRF.UCLA and WRF.UCSD."
                "\nIf you're trying to pull dynamically downscaled data, please specify"
                "\n    an institution_id that matches the downscaling method (e.g., "
                "\n'UCLA' or 'UCSD' are two of several options for WRF datasets)."
            )
            logger.warning(msg)
            logger.error(
                "Failed to concatenate datasets along '%s' dimension: %s",
                self.dim_name,
                e,
                exc_info=True,
            )
            raise

        logger.info("Concatenated datasets along '%s' dimension.", self.dim_name)

        self.update_context(context, attr_ids)

        # Set resolution attribute if available
        resolutions = {
            "d01": "45 km",
            "d02": "9 km",
            "d03": "3 km",
        }
        if isinstance(result, dict) and result:
            key = list(result.keys())[0]
            key_parts = key.split(".")
            for k in key_parts:
                if k in resolutions:
                    concatenated.attrs["resolution"] = resolutions[k]
                    break
        return concatenated

    def _align_time_dim(
        self, dataset: Union[xr.Dataset, xr.DataArray], context: Dict[str, Any] = {}
    ) -> Union[xr.Dataset, xr.DataArray]:
        """Align the time dimension of the dataset to ensure consistency.

        Parameters
        ----------
        dataset : Union[xr.Dataset, xr.DataArray]
            The dataset whose time dimension needs to be aligned.

        Returns
        -------
        Union[xr.Dataset, xr.DataArray]
            The dataset with aligned time dimension.
        """
        if (
            self.dim_name == "sim"
            and "table_id" in context
            and context["table_id"] == "day"
            and "time" in dataset.coords
        ):
            floored_time = dataset["time"].dt.floor("D")
            dataset = dataset.assign_coords(time=floored_time)
        return dataset

    def update_context(
        self, context: Dict[str, Any], source_ids: List[str] | object = UNSET
    ):
        """Update the context with information about the concatenation transformation.

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

        # Include information about time domain extension if time dimension was used
        process_info = f"Process '{self.name}' applied to the data."
        if hasattr(self, "_original_dim_name") and self._original_dim_name == "time":
            process_info += " Time domain extension was performed by prepending historical data to SSP scenarios."
        process_info += f" Multiple datasets were concatenated along a new '{self.dim_name}' dimension."

        context[_NEW_ATTRS_KEY][self.name] = f"""{process_info}"""

    def set_data_accessor(self, catalog: DataCatalog):
        """Set the data catalog for this processor.

        Parameters
        ----------
        catalog : DataCatalog
            The data catalog to be used by this processor.

        """
        self.catalog = catalog
