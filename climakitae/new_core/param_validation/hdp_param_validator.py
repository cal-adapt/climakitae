"""Validator for historical data platform parameters."""

from __future__ import annotations

import logging
from typing import Any, Dict

from climakitae.core.constants import CATALOG_HDP, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.param_validation.abc_param_validation import (
    ParameterValidator,
    register_catalog_validator,
)

# Module logger
logger = logging.getLogger(__name__)


@register_catalog_validator(CATALOG_HDP)
class HDPValidator(ParameterValidator):
    """Validator for historical data platform parameters.

    This validator enforces that queries must specify a single network_id
    to prevent mixing data from different weather station networks, which
    may have different time periods and data characteristics.

    Parameters
    ----------
    catalog : DataCatalog
        Catalog of datasets

    Attributes
    ----------
    catalog : intake_esm.core.esm_datastore
        The HDP catalog for validation
    all_catalog_keys : dict
        Required query parameters with default values

    """

    def __init__(self, catalog: DataCatalog):
        """Initialize with catalog of historical data platform datasets.

        Parameters
        ----------
        catalog : DataCatalog
            Catalog of datasets

        """
        super().__init__()
        self.all_catalog_keys = {
            "network_id": UNSET,
            "station_id": UNSET,
        }
        self.catalog = catalog.hdp
        logger.debug("HDPValidator initialized for hdp catalog")

    def is_valid_query(self, query: Dict[str, Any]) -> Dict[str, Any] | None:
        """Validate query parameters for HDP data.

        Requires network_id and ensures it's a single value. Station_id is optional.

        Parameters
        ----------
        query : dict
            Query parameters to validate

        Returns
        -------
        dict or None
            Validated query if valid, None otherwise

        Notes
        -----
        Validation checks performed:

        1. network_id is required and must be a single value
           (accepts string or single-item list, rejects multi-item lists)
        2. station_id is optional and can be used to filter within the network
        3. If station_id is provided, all requested station IDs must exist in the catalog

        Multiple network_ids are not allowed to prevent mixing data from
        different networks with potentially different time periods and
        data characteristics.

        """
        logger.debug("Validating HDP query: %s", query)

        # Check that network_id is provided and is a single value
        initial_checks = [
            self._check_network_id_required(query),
            self._check_station_ids_exist(query),
        ]
        if not all(initial_checks):
            logger.warning("Initial validation checks failed")
            return None

        result = super()._is_valid_query(query)
        logger.info("HDP query validation result: %s", bool(result))
        return result

    def _check_network_id_required(self, query: Dict[str, Any]) -> bool:
        """Check that network_id is provided and is a single value.

        Accepts network_id as a string or single-item list (which gets converted
        to a string). Rejects multi-item lists to prevent mixing data from
        different weather station networks.

        Parameters
        ----------
        query : Dict[str, Any]
            The query to check. Single-item lists will be converted to strings.

        Returns
        -------
        bool
            True if network_id is valid, False otherwise.

        """
        logger.debug(
            "Checking network_id constraint: %s",
            query.get("network_id"),
        )

        network_id = query.get("network_id", UNSET)

        # Check if network_id is provided
        if network_id is UNSET:
            msg = (
                "network_id is required for HDP data queries. "
                "Please specify which weather station network to query. "
                "station_id is optional and can be used to filter to specific stations."
            )
            logger.warning(msg, stacklevel=999)
            return False

        # Ensure network_id is a single value, not a list
        if isinstance(network_id, (list, tuple)):
            if len(network_id) == 0:
                msg = (
                    "network_id cannot be an empty list. "
                    "Please specify a single weather station network."
                )
                logger.warning(msg, stacklevel=999)
                return False
            elif len(network_id) > 1:
                msg = (
                    f"Only one network_id can be queried at a time. "
                    f"You provided: {network_id}. "
                    "Please specify a single weather station network to avoid "
                    "mixing data from different networks with potentially different "
                    "time periods and data characteristics."
                )
                logger.warning(msg, stacklevel=999)
                return False
            else:
                # Convert single-item list to string
                query["network_id"] = network_id[0]
                logger.debug("Converted single-item network_id list to string")

        return True

    def _check_station_ids_exist(self, query: Dict[str, Any]) -> bool:
        """Check that all requested station_ids exist in the catalog.

        If station_id is provided in the query, validates that all requested
        stations exist in the catalog. Returns an error listing any invalid IDs.

        Parameters
        ----------
        query : Dict[str, Any]
            The query to check.

        Returns
        -------
        bool
            True if all station_ids are valid or station_id not provided,
            False if any station_ids are invalid.

        """
        station_id = query.get("station_id", UNSET)

        # station_id is optional, so if not provided, validation passes
        if station_id is UNSET:
            return True

        # Normalize to list for processing
        if isinstance(station_id, str):
            station_ids_to_check = [station_id]
        else:
            station_ids_to_check = list(station_id)

        # Search catalog for all requested stations
        try:
            result = self.catalog.search(station_id=station_ids_to_check)
            found_station_ids = set(result.df["station_id"].unique())
            requested_station_ids = set(station_ids_to_check)

            # Check if any requested stations are missing
            missing_station_ids = requested_station_ids - found_station_ids

            if missing_station_ids:
                msg = (
                    f"The following station_id(s) were not found in the catalog: "
                    f"{sorted(missing_station_ids)}. "
                    f"Please verify your station IDs and try again."
                )
                logger.warning(msg, stacklevel=999)
                return False

            logger.debug(
                "All requested station_ids are valid: %s", station_ids_to_check
            )
            return True

        except Exception as e:
            msg = f"Error validating station_ids: {e}"
            logger.warning(msg, stacklevel=999)
            return False

      
    def _is_valid_processor(): 
    """Check if processor supplied with variable query is one of time_slice, update_attributes. All else invalid
    """
