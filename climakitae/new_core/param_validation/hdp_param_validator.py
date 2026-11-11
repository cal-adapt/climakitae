"""Validator for historical data platform parameters."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import xarray as xr

from climakitae.core.constants import CATALOG_HDP, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.param_validation.abc_param_validation import (
    ParameterValidator,
    register_catalog_validator,
)

# Module logger
logger = logging.getLogger(__name__)

# Maximum number of station zarrs opened concurrently when checking the
# requested time_slice against each station's record.
_MAX_STATION_READERS = 16

# Earliest start and latest end of any station's record in each network,
# read from the time coordinates of every station zarr. Used when a query
# names a network but no stations.
NETWORK_DATE_RANGES: Dict[str, Tuple[str, str]] = {
    "ASOSAWOS": ("1980-01-01", "2022-08-31"),
    "CAHYDRO": ("2010-05-27", "2022-08-31"),
    "CDEC": ("2010-09-09", "2022-08-31"),
    "CIMIS": ("1982-05-30", "2022-08-31"),
    "CNRFC": ("2001-06-21", "2022-08-31"),
    "CRN": ("2006-11-08", "2022-08-31"),
    "CW3E": ("2019-10-24", "2022-08-31"),
    "CWOP": ("2002-06-20", "2022-08-31"),
    "HADS": ("1997-01-01", "2022-08-31"),
    "HNXWFO": ("2004-03-25", "2022-08-31"),
    "HOLFUY": ("2017-08-21", "2022-08-31"),
    "HPWREN": ("2004-10-03", "2022-08-31"),
    "LOXWFO": ("2006-05-08", "2022-08-31"),
    "MAP": ("2014-03-20", "2022-08-31"),
    "MARITIME": ("1984-01-25", "2022-08-31"),
    "MTRWFO": ("2004-03-25", "2022-08-31"),
    "NCAWOS": ("2002-05-01", "2022-08-31"),
    "NDBC": ("1980-01-01", "2022-08-31"),
    "NOS-NWLON": ("2008-02-09", "2022-08-31"),
    "NOS-PORTS": ("2008-02-09", "2022-08-31"),
    "OtherISD": ("1980-01-01", "2022-08-31"),
    "RAWS": ("1997-01-01", "2022-08-31"),
    "SCAN": ("1986-10-01", "2022-08-31"),
    "SGXWFO": ("2003-09-10", "2022-08-31"),
    "SHASAVAL": ("2002-12-17", "2022-08-31"),
    "SNOTEL": ("1980-01-01", "2022-08-31"),
    "VALLEYWATER": ("1949-09-12", "2025-03-05"),
    "VCAPCD": ("2010-05-27", "2022-08-31"),
}


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
        self.invalid_processors = [
            "localize",
            "clip",
            "convert_units",
            "bias_adjust_model_to_station",
            "filter_unadjusted_models",
            "metric_calc",
            "warming_level",
        ]
        logger.debug("HDPValidator initialized for hdp catalog")

    def get_default_processors(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Get default processors for HDP catalog.

        HDP data uses station_id for concatenation since it's station-based data.

        Parameters
        ----------
        query : Dict[str, Any]
            The current query containing user parameters

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping processor names to their default configurations
        """
        defaults = super().get_default_processors(query)  # Get universal defaults
        defaults.update(
            {
                "concat": "station_id",
            }
        )
        return defaults

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
        4. A time_slice must overlap the record of at least one requested
           station (or, with no station_id, the network's overall record).
           Stations with no data in the slice are reported.

        Multiple network_ids are not allowed to prevent mixing data from
        different networks with potentially different time periods and
        data characteristics.

        """
        logger.debug("Validating HDP query: %s", query)

        # Check that network_id is provided and is a single value
        initial_checks = [
            self._check_network_id_required(query),
            self._check_station_ids_exist(query),
            self._check_query_invalid_processors(query),
            self._check_time_slice_available(query, CATALOG_HDP),
        ]
        if not all(initial_checks):
            logger.warning("Initial validation checks failed")
            return None

        result = super()._is_valid_query(query)
        logger.info("HDP query validation result: %s", bool(result))
        return result

    def _check_query_invalid_processors(self, query: Dict[str, Any]) -> bool:
        """Check if the query contains the invalid processors

        Parameters
        ----------
        query : Dict[str, Any]
            The query to check.

        Returns
        -------
        bool
            True if the query does not contain invalid processors, False otherwise.

        """
        for processor in query.get("processors", []):
            if processor in self.invalid_processors:
                logger.warning("Invalid processor for HDP data: %s", processor)
                return False
        return True

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
            logger.warning(msg)
            return False

        # Ensure network_id is a single value, not a list
        if isinstance(network_id, (list, tuple)):
            if len(network_id) == 0:
                msg = (
                    "network_id cannot be an empty list. "
                    "Please specify a single weather station network."
                )
                logger.warning(msg)
                return False
            elif len(network_id) > 1:
                msg = (
                    f"Only one network_id can be queried at a time. "
                    f"You provided: {network_id}. "
                    "Please specify a single weather station network to avoid "
                    "mixing data from different networks with potentially different "
                    "time periods and data characteristics."
                )
                logger.warning(msg)
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
                logger.warning(msg)
                return False

            logger.debug(
                "All requested station_ids are valid: %s", station_ids_to_check
            )
            return True

        except Exception as e:
            msg = f"Error validating station_ids: {e}"
            logger.warning(msg)
            return False

    def _check_time_slice_available(
        self, query: Dict[str, Any], catalog_name: str
    ) -> bool:
        """Check the requested time_slice against the station records.

        With station_id, each requested station's time coordinate is read
        and the slice must overlap at least one of them; stations without
        data in the slice are reported. With only network_id, the slice is
        checked against the network's overall record.

        Parameters
        ----------
        query : Dict[str, Any]
            The query to check.
        catalog_name : str
            Catalog name used in user-facing messages.

        Returns
        -------
        bool
            False if no requested station (or the network) has data in the
            time slice.
        """
        dates = self._get_time_slice_dates(query)
        if dates is None:
            return True
        start, end = dates

        station_id = query.get("station_id", UNSET)
        if station_id is UNSET:
            network_id = query.get("network_id", UNSET)
            if not isinstance(network_id, str) or network_id not in NETWORK_DATE_RANGES:
                return True
            first, last = NETWORK_DATE_RANGES[network_id]
            return self._check_dates_overlap(
                start,
                end,
                pd.Timestamp(first),
                pd.Timestamp(last),
                label=f"stations in the {network_id} network",
                available=f"{first} to {last}",
            )

        station_ids = [station_id] if isinstance(station_id, str) else list(station_id)
        ranges = self._get_station_date_ranges(station_ids)
        if not ranges:
            return True

        no_data = {
            sid: rng
            for sid, rng in ranges.items()
            if start.normalize() > rng[1] or end.normalize() < rng[0]
        }
        described = {
            sid: f"{rng[0].date()} to {rng[1].date()}" for sid, rng in no_data.items()
        }
        requested = f"{start.date()} to {end.date()}"

        if len(no_data) == len(ranges):
            logger.warning(
                "Time slice (%s) is outside the records of all requested "
                "stations in the '%s' catalog: %s. Please choose a time slice "
                "within a station's record.",
                requested,
                catalog_name,
                described,
            )
            return False

        if no_data:
            logger.warning(
                "Time slice (%s) is outside the records of these stations, "
                "which will have no data: %s.",
                requested,
                described,
            )

        if len(ranges) == 1:
            ((sid, (first, last)),) = ranges.items()
            return self._check_dates_overlap(
                start,
                end,
                first,
                last,
                label=f"station {sid}",
                available=f"{first.date()} to {last.date()}",
            )
        return True

    def _get_station_date_ranges(
        self, station_ids: list[str]
    ) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
        """Read the first and last dates of each station's record.

        Stations that are not in the catalog or cannot be opened are left
        out, so the check is skipped for them.

        Parameters
        ----------
        station_ids : list of str
            Station IDs to look up.

        Returns
        -------
        dict
            Mapping of station_id to (first date, last date).
        """
        try:
            df = self.catalog.search(station_id=station_ids).df
            paths = dict(zip(df["station_id"], df["path"]))
        except Exception as e:  # catalog lookup is best effort here
            logger.debug("Could not look up station paths: %s", e)
            return {}

        def read_range(
            item: Tuple[str, str],
        ) -> Tuple[str, Optional[Tuple[pd.Timestamp, pd.Timestamp]]]:
            sid, path = item
            try:
                ds = xr.open_zarr(
                    path, consolidated=None, storage_options={"anon": True}
                )
                times = ds.indexes["time"]
                return sid, (
                    pd.Timestamp(times.min()).normalize(),
                    pd.Timestamp(times.max()).normalize(),
                )
            except Exception as e:  # unreadable station: skip its check
                logger.debug("Could not read time range for %s: %s", sid, e)
                return sid, None

        with ThreadPoolExecutor(
            max_workers=min(_MAX_STATION_READERS, max(len(paths), 1))
        ) as pool:
            results = pool.map(read_range, paths.items())
        return {sid: rng for sid, rng in results if rng is not None}
