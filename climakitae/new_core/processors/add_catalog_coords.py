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

        Adds network_id as a coordinate to the dataset. Since validation ensures
        only a single network_id is queried, this is obtained directly from context.

        Parameters
        ----------
        result : Dataset, DataArray, or iterable
            The data to process.
        context : dict
            Processing context containing query parameters including network_id.

        Returns
        -------
        Dataset, DataArray, or iterable
            Data with network_id added as a coordinate.
        """
        logger.debug("AddCatalogCoords.execute called")

        # Get network_id from context (required by validator)
        network_id = context.get("network_id", UNSET)
        if network_id is UNSET:
            logger.debug("No network_id in context, skipping coordinate addition")
            return result

        logger.debug("Adding network_id='%s' as coordinate", network_id)

        match result:
            case dict():
                # Process each dataset in the dictionary
                for key, item in result.items():
                    result[key] = self._add_coords_to_dataset(item, network_id)
            case xr.Dataset():
                result = self._add_coords_to_dataset(result, network_id)
            case xr.DataArray():
                # Convert to dataset, add coords, convert back
                ds = result.to_dataset()
                ds = self._add_coords_to_dataset(ds, network_id)
                result = (
                    ds[result.name] if result.name else list(ds.data_vars.values())[0]
                )
            case list() | tuple():
                result = [
                    self._add_coords_to_dataset(item, network_id) for item in result
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
        network_id: str,
    ) -> Union[xr.Dataset, xr.DataArray]:
        """Add network_id coordinate to a single dataset or dataarray.

        Parameters
        ----------
        ds : Dataset or DataArray
            The dataset to modify.
        network_id : str
            The network identifier to add as a coordinate.

        Returns
        -------
        Dataset or DataArray
            Dataset with network_id coordinate added.
        """
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset()
            was_dataarray = True
        else:
            was_dataarray = False

        # Add network_id coordinate
        if "station_id" in ds.dims:
            # Dataset has station_id dimension - broadcast network_id across all stations
            # Since validation ensures all stations are from the same network,
            # we can safely broadcast the single network_id value
            num_stations = ds.dims["station_id"]
            network_values = [network_id] * num_stations
            logger.debug(
                "Broadcasting network_id='%s' across %d stations",
                network_id,
                num_stations,
            )

            ds = ds.assign_coords(network_id=("station_id", network_values))
            ds["network_id"].attrs.update(
                {
                    "long_name": "Weather station network identifier",
                    "description": "Network that operates each weather station",
                }
            )
            logger.debug(
                "Successfully added network_id coordinate with station_id dimension"
            )
        else:
            # No station_id dimension, add as scalar coordinate
            ds = ds.assign_coords(network_id=network_id)
            ds["network_id"].attrs.update(
                {
                    "long_name": "Weather station network identifier",
                    "description": "Network that operates the weather station(s)",
                }
            )
            logger.debug("Successfully added network_id as scalar coordinate")

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
