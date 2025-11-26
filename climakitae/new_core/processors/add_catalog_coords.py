"""AddCatalogCoords Processor for adding catalog metadata as coordinates.

This processor is specifically designed for HDP data to add metadata like
network_id and station_id as coordinates to the dataset.
"""

from typing import Any, Dict, Iterable, Union
import logging
import xarray as xr

from climakitae.core.constants import UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.processors.abc_data_processor import (
    DataProcessor,
    register_processor,
)

# Module logger
logger = logging.getLogger(__name__)


# Run after concatenation (priority=26), only for hdp catalog
@register_processor("add_catalog_coords", priority=26, catalogs=["hdp"])
class AddCatalogCoords(DataProcessor):
    """Add catalog metadata as coordinates to the dataset.

    For HDP data, this adds network_id from the query parameters as a coordinate
    to make the dataset more self-describing.

    Parameters
    ----------
    value : Any
        Not used for this processor, but kept for consistency with the interface.

    """

    def __init__(self, value: Any = UNSET):
        """Initialize the AddCatalogCoords processor.

        Parameters
        ----------
        value : Any, optional
            Not used, kept for interface consistency.
        """
        self.value = value
        self.name = "add_catalog_coords"
        self.catalog = None
        self.needs_catalog = True

    def execute(
        self,
        result: Union[
            xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]
        ],
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray, Iterable[Union[xr.Dataset, xr.DataArray]]]:
        """Execute the AddCatalogCoords processor.

        Adds network_id and other catalog metadata as coordinates to the dataset.

        Parameters
        ----------
        result : Dataset, DataArray, or iterable
            The data to process.
        context : dict
            Processing context containing query parameters.

        Returns
        -------
        Dataset, DataArray, or iterable
            Data with catalog metadata added as coordinates.
        """
        logger.debug("AddCatalogCoords.execute called")

        # Always get network_id from catalog using station_ids
        station_ids = context.get("station_id", UNSET)
        if station_ids is UNSET:
            logger.debug("No station_id in context, skipping coordinate addition")
            return result

        subset = self.catalog.hdp.search(station_id=station_ids)
        # Create mapping from station_id to network_id
        station_to_network = dict(zip(subset.df["station_id"], subset.df["network_id"]))
        logger.debug("Retrieved station to network mapping from catalog: %s", station_to_network)

        match result:
            case dict():
                # Process each dataset in the dictionary
                for key, item in result.items():
                    result[key] = self._add_coords_to_dataset(
                        item, station_to_network, context
                    )
            case xr.Dataset():
                result = self._add_coords_to_dataset(result, station_to_network, context)
            case xr.DataArray():
                # Convert to dataset, add coords, convert back
                ds = result.to_dataset()
                ds = self._add_coords_to_dataset(ds, station_to_network, context)
                result = (
                    ds[result.name] if result.name else list(ds.data_vars.values())[0]
                )
            case list() | tuple():
                result = [
                    self._add_coords_to_dataset(item, station_to_network, context)
                    for item in result
                ]
            case _:
                logger.warning(
                    "Unexpected result type: %s, skipping coordinate addition",
                    type(result),
                )
                return result

        logger.info("AddCatalogCoords applied: added network_id coordinate")
        return result

    def _add_coords_to_dataset(
        self,
        ds: Union[xr.Dataset, xr.DataArray],
        station_to_network: dict,
        context: Dict[str, Any],
    ) -> Union[xr.Dataset, xr.DataArray]:
        """Add coordinates to a single dataset or dataarray.

        Parameters
        ----------
        ds : Dataset or DataArray
            The dataset to modify.
        station_to_network : dict
            Mapping from station_id to network_id.
        context : dict
            Processing context.

        Returns
        -------
        Dataset or DataArray
            Dataset with added coordinates.
        """
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset()
            was_dataarray = True
        else:
            was_dataarray = False

        # Add network_id with station_id dimension if it exists
        if "station_id" in ds.dims:
            # Map each station_id in the dataset to its network_id
            station_ids_in_ds = ds["station_id"].values
            logger.debug("Station IDs in dataset: %s", station_ids_in_ds)
            logger.debug("Station to network mapping: %s", station_to_network)
            network_values = [station_to_network.get(str(sid), "unknown") for sid in station_ids_in_ds]
            logger.debug("Network values to assign: %s", network_values)

            ds = ds.assign_coords(
                network_id=("station_id", network_values)
            )
            ds["network_id"].attrs.update(
                {
                    "long_name": "Weather station network identifier",
                    "description": "Network that operates each weather station",
                }
            )
            logger.debug("Successfully added network_id coordinate")
        else:
            # No station_id dimension, add as scalar coordinate
            unique_networks = list(set(station_to_network.values()))
            if len(unique_networks) == 1:
                ds = ds.assign_coords(network_id=unique_networks[0])
                ds["network_id"].attrs.update(
                    {
                        "long_name": "Weather station network identifier",
                        "description": "Network that operates the weather station(s)",
                    }
                )
            else:
                logger.warning("Multiple network_ids but no station_id dimension found")

        if was_dataarray:
            # Convert back to DataArray
            var_name = list(ds.data_vars.keys())[0]
            return ds[var_name]

        return ds

    def update_context(self, context: Dict[str, Any]):
        """Update the context with information about coordinate addition.

        Parameters
        ----------
        context : dict
            Parameters for processing the data.

        Note
        ----
        The context is updated in place.
        """
        from climakitae.core.constants import _NEW_ATTRS_KEY

        if _NEW_ATTRS_KEY not in context:
            context[_NEW_ATTRS_KEY] = {}

        network_id = context.get("network_id", UNSET)
        if network_id is not UNSET:
            context[_NEW_ATTRS_KEY][
                self.name
            ] = f"Added network_id='{network_id}' as coordinate"
            logger.debug(
                "AddCatalogCoords.update_context added entry for network_id=%s",
                network_id,
            )

    def set_data_accessor(self, catalog: DataCatalog):
        """Set the data accessor.

        Parameters
        ----------
        catalog : DataCatalog
            The data catalog instance.
        """
        self.catalog = catalog
