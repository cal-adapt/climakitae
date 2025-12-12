"""Data access module for ClimakitAE.

This module provides a thread-safe singleton DataCatalog class for managing
connections to various climate data catalogs including boundary, renewables,
and general climate datasets. The DataCatalog class offers a unified interface
for accessing and querying multiple intake catalogs with support for dynamic
catalog management.

Thread Safety
-------------
The DataCatalog singleton is thread-safe. Multiple threads can safely:
- Access the singleton instance concurrently
- Call get_data() with different catalog keys simultaneously
- Access catalog properties and boundaries

Classes
-------
DataCatalog
    Thread-safe singleton class that inherits from dict and manages catalog
    connections. Provides properties for accessing specific catalogs and
    methods for querying and retrieving climate datasets.

"""

import difflib
import logging
import threading
from typing import Any, Dict, Optional

import geopandas as gpd
import intake
import intake_esm
import pandas as pd
import xarray as xr

from climakitae.core.constants import (
    CATALOG_BOUNDARY,
    CATALOG_CADCAT,
    CATALOG_REN_ENERGY_GEN,
    UNSET,
)

# Module logger
logger = logging.getLogger(__name__)
from climakitae.core.paths import (
    BOUNDARY_CATALOG_URL,
    DATA_CATALOG_URL,
    RENEWABLES_CATALOG_URL,
    STATIONS_CSV_PATH,
)
from climakitae.new_core.data_access.boundaries import Boundaries
from climakitae.util.utils import read_csv_file


class DataCatalog(dict):
    """Thread-safe singleton for managing catalog connections to climate data sources.

    This class implements a thread-safe singleton pattern and inherits from dict
    to provide a unified interface for accessing multiple climate data catalogs.
    It manages connections to boundary, renewables, and general climate datasets
    through intake and intake-esm catalogs, offering convenient properties and
    methods for data querying and retrieval.

    The class automatically initializes connections to predefined catalogs and
    supports dynamic addition of new catalogs.

    Thread Safety
    -------------
    This class is thread-safe. The singleton instance is protected by a lock
    during creation, and the get_data() method accepts the catalog key as a
    parameter rather than storing it as mutable state, allowing concurrent
    queries from multiple threads.

    Properties
    ----------
    data : intake_esm.core.esm_datastore
        Access to the main climate data catalog.
    boundary : intake.catalog.Catalog
        Access to the boundary conditions catalog.
    boundaries : Boundaries
        Access to the lazy-loading boundaries data manager.
    renewables : intake_esm.core.esm_datastore
        Access to the renewables data catalog.

    Methods
    -------
    set_catalog(name, catalog)
        Add a new catalog to the collection.
    get_data(query, catalog_key)
        Retrieve data from the specified catalog using query parameters.
    resolve_catalog_key(key)
        Resolve and validate a catalog key, returning the closest match if needed.

    Notes
    -----
    This class implements the singleton pattern, ensuring only one instance
    exists throughout the application lifecycle. Multiple calls to DataCatalog()
    will return the same instance.

    The class automatically handles catalog initialization and provides sensible
    defaults when invalid catalog keys are specified.

    Examples
    --------
    Thread-safe concurrent usage:

    >>> from concurrent.futures import ThreadPoolExecutor
    >>> catalog = DataCatalog()
    >>> def fetch_data(params):
    ...     query, catalog_key = params
    ...     return catalog.get_data(query, catalog_key=catalog_key)
    >>> with ThreadPoolExecutor(max_workers=4) as executor:
    ...     results = list(executor.map(fetch_data, queries_and_keys))

    """

    _instance = UNSET
    _lock = threading.Lock()

    def __new__(cls) -> "DataCatalog":
        """Override __new__ to implement thread-safe singleton pattern.

        Uses double-checked locking to ensure thread-safe singleton creation
        while minimizing lock contention after initialization.

        Returns
        -------
        DataCatalog
            The singleton instance of DataCatalog.

        """
        # Fast path: if already initialized, return without lock
        if cls._instance is not UNSET:
            return cls._instance

        # Slow path: acquire lock for initialization
        with cls._lock:
            # Double-check after acquiring lock
            if cls._instance is UNSET:
                cls._instance = super(DataCatalog, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the DataCatalog instance.

        This method sets up the catalog connections and initializes internal
        state. It only runs once due to the singleton pattern implementation.

        """
        if not getattr(self, "_initialized", False):
            super().__init__()
            self[CATALOG_CADCAT] = intake.open_esm_datastore(DATA_CATALOG_URL)
            self[CATALOG_BOUNDARY] = intake.open_catalog(BOUNDARY_CATALOG_URL)
            self[CATALOG_REN_ENERGY_GEN] = intake.open_esm_datastore(
                RENEWABLES_CATALOG_URL
            )
            self.catalog_df = self.merge_catalogs()
            stations_df = read_csv_file(STATIONS_CSV_PATH)
            self["stations"] = gpd.GeoDataFrame(
                stations_df,
                crs="EPSG:4326",
                geometry=gpd.points_from_xy(stations_df.LON_X, stations_df.LAT_Y),
            )

            self._initialized = True
            # Initialize boundaries with lazy loading
            self._boundaries = UNSET
            self._boundaries_lock = threading.Lock()
            self.available_boundaries = UNSET

    @property
    def data(self) -> intake_esm.core.esm_datastore:
        """Access data catalog.

        Returns
        -------
        intake_esm.core.esm_datastore
            The main climate data catalog.

        """
        return self[CATALOG_CADCAT]

    @property
    def boundary(self) -> intake.catalog.Catalog:
        """Access boundary catalog.

        Returns
        -------
        intake.catalog.Catalog
            The boundary conditions catalog.

        """
        return self[CATALOG_BOUNDARY]

    @property
    def renewables(self) -> intake_esm.core.esm_datastore:
        """Access renewables catalog.

        Returns
        -------
        intake_esm.core.esm_datastore
            The renewables data catalog.

        """
        return self[CATALOG_REN_ENERGY_GEN]

    @property
    def boundaries(self) -> Boundaries:
        """Access boundaries data with lazy loading (thread-safe).

        Returns
        -------
        Boundaries
            The lazy-loading boundaries data manager.

        """
        # Fast path: already initialized
        if self._boundaries is not UNSET:
            return self._boundaries

        # Slow path: acquire lock for initialization
        with self._boundaries_lock:
            # Double-check after acquiring lock
            if self._boundaries is UNSET:
                self._boundaries = Boundaries(self.boundary)
        return self._boundaries

    def merge_catalogs(self) -> pd.DataFrame:
        """Merge the intake catalogs for data and renewables into a single DataFrame.

        This method combines the data and renewables catalogs into a unified
        DataFrame for easier searching and querying across all available datasets.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the merged data from both catalogs with an
            additional 'catalog' column identifying the source catalog.

        """
        ren_df = self.renewables.df
        data_df = self.data.df
        ren_df["catalog"] = CATALOG_REN_ENERGY_GEN
        data_df["catalog"] = CATALOG_CADCAT
        ret = pd.concat([ren_df, data_df], ignore_index=True)
        return ret

    def resolve_catalog_key(self, key: str) -> Optional[str]:
        """Resolve and validate a catalog key.

        This method validates the provided catalog key and attempts to find
        the closest match if the exact key is not found. This is a pure function
        that does not modify any instance state, making it thread-safe.

        Parameters
        ----------
        key : str
            Key of the catalog to resolve. Should be one of the available catalog keys.

        Returns
        -------
        str or None
            The resolved catalog key if valid or a close match is found,
            None if no valid key can be determined.

        Warns
        -----
        UserWarning
            If the catalog key is not found and suggestions are provided.

        Examples
        --------
        >>> catalog = DataCatalog()
        >>> resolved = catalog.resolve_catalog_key("cadcat")
        >>> resolved
        'cadcat'

        """
        if key in self:
            return key

        logger.warning(
            f"\n\nCatalog key '{key}' not found."
            f"\nAttempting to find intended catalog key.\n\n",
            stacklevel=999,
        )
        logger.info("Available catalog keys: %s", list(self.keys()))
        closest = _get_closest_options(key, list(self.keys()))
        if not closest:
            logger.warning(
                f"No catalog found for '{key}'. "
                f"Available options: {list(self.keys())}",
                stacklevel=999,
            )
            return None

        match len(closest):
            case 0:
                logger.warning(
                    f"No catalog found for '{key}'. "
                    f"Available options: {list(self.keys())}",
                    stacklevel=999,
                )
                return None
            case 1:
                logger.warning(
                    f"\n\nUsing closest match '{closest[0]}' for catalog '{key}'.",
                    stacklevel=999,
                )
                return closest[0]
            case _:
                logger.warning(
                    f"Multiple closest matches found for '{key}': {closest}. "
                    "Please specify a more precise key.",
                    stacklevel=999,
                )
                return None

    # Backward compatibility alias (deprecated)
    def set_catalog_key(self, key: str) -> "DataCatalog":
        """Set the catalog key (DEPRECATED - use resolve_catalog_key instead).

        .. deprecated:: 1.5.0
            This method stores mutable state on the singleton which is not
            thread-safe. Use :meth:`resolve_catalog_key` and pass the key
            directly to :meth:`get_data` instead.

        Parameters
        ----------
        key : str
            Key of the catalog to set.

        Returns
        -------
        DataCatalog
            The current instance (for backward compatibility).

        """
        import warnings

        warnings.warn(
            "set_catalog_key() is deprecated and not thread-safe. "
            "Use resolve_catalog_key() and pass catalog_key to get_data() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._catalog_key = self.resolve_catalog_key(key)
        return self

    def set_catalog(self, name: str, catalog: str) -> "DataCatalog":
        """Set a named catalog.

        Parameters
        ----------
        name : str
            Name of the catalog to set.
        catalog : str
            URL or path to the catalog file.

        Returns
        -------
        DataCatalog
            The current instance of DataCatalog allowing method chaining.

        """
        self[name] = intake.open_esm_datastore(catalog)
        return self

    def get_data(
        self, query: Dict[str, Any], catalog_key: Optional[str] = None
    ) -> Dict[str, xr.Dataset]:
        """Get data from the specified catalog (thread-safe).

        This method queries the specified catalog using the provided parameters
        and returns the matching datasets as a dictionary. The catalog_key is
        passed as a parameter rather than stored as instance state, making this
        method safe to call from multiple threads simultaneously.

        Parameters
        ----------
        query : dict
            Query parameters for filtering data. The available parameters
            depend on the catalog and may include items like 'variable',
            'scenario', 'model', etc.
        catalog_key : str, optional
            The key identifying which catalog to query. If not provided,
            falls back to the deprecated instance attribute (for backward
            compatibility).

        Returns
        -------
        dict[str, xr.Dataset]
            The requested dataset(s) from the catalog, keyed by dataset identifiers.

        Raises
        ------
        ValueError
            If no catalog_key is provided and no default is available.

        Examples
        --------
        >>> catalog = DataCatalog()
        >>> query = {"variable_id": "tas", "experiment_id": "historical"}
        >>> data = catalog.get_data(query, catalog_key="cadcat")

        """
        # Use provided catalog_key, fall back to deprecated instance attr
        effective_key = catalog_key
        if effective_key is None:
            effective_key = getattr(self, "_catalog_key", None)
        if effective_key is None:
            raise ValueError(
                "catalog_key must be provided. Use resolve_catalog_key() to "
                "validate the key before calling get_data()."
            )

        logger.info("Querying %s catalog", effective_key)
        logger.debug("Query parameters: %s", query)
        logger.debug("Querying %s catalog with query: %s", effective_key, query)
        # if any(isinstance(v, list) for v in query.values()):
        #     # query contains a list, which is not supported by intake
        #     for key, value in query.items():
        #         if isinstance(value, list):
        #             # Convert list to a comma-separated string
        #             query[key] = ",".join(value)

        logger.debug("Executing catalog search")
        # Detailed query log (was printed previously)
        logger.debug("Querying %s catalog with query: %s", effective_key, query)
        result = (
            self[effective_key]
            .search(**query)
            .to_dataset_dict(
                # Use consolidated=None for compatibility with both Zarr v2 and v3.
                # - True: requires consolidated metadata (fails on Zarr v3 without it)
                # - False: always reads metadata from individual arrays
                # - None: uses consolidated if available, falls back to individual reads
                zarr_kwargs={"consolidated": None},
                storage_options={"anon": True},
                progressbar=False,
            )
        )
        logger.info("Retrieved %d dataset(s) from catalog", len(result))
        logger.debug("Retrieved datasets: %s", list(result.keys()))
        return result

    def list_clip_boundaries(self) -> dict[str, list[str]]:
        """List all available boundary options for clipping operations.

        This method populates the `available_boundaries` attribute with a
        dictionary of boundary categories and their available options. It's a
        convenience method that provides direct access to boundary options
        without needing to instantiate a Clip processor.

        Notes
        -----
        After calling this method, the available boundaries can be accessed
        via the `available_boundaries` attribute.

        Examples
        --------
        >>> catalog = DataCatalog()
        >>> catalog.list_clip_boundaries()
        >>> print(catalog.available_boundaries["states"])
        ['AZ', 'CA', 'CO', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY']

        """
        boundary_dict = self.boundaries.boundary_dict()

        # Create a clean dictionary with boundary categories and their available options
        self.available_boundaries = {}

        for category, lookups in boundary_dict.items():
            # Skip special categories that don't represent actual boundary data
            if category in ["none", "lat/lon"]:
                continue

            # Convert keys to a sorted list for better presentation
            boundary_keys = sorted(list(lookups.keys()))
            self.available_boundaries[category] = boundary_keys

        return self.available_boundaries

    def print_clip_boundaries(self) -> None:
        """Print all available boundary options for clipping in a user-friendly format.

        This method provides a nicely formatted output showing all boundary
        categories and their available options for clipping operations. The
        output is formatted to be readable and includes summarized counts for
        categories with many options.

        Examples
        --------
        >>> catalog = DataCatalog()
        >>> catalog.print_clip_boundaries()
        Available Boundary Options for Clipping:
        ========================================

        states:
          - AZ, CA, CO, ID, MT
            ... and 6 more options

        """
        try:
            self.list_clip_boundaries()
        except Exception as e:
            logger.error("Error accessing boundary data: %s", e, exc_info=True)
            return

        logger.info("Available Boundary Options for Clipping:")
        logger.info("%s", "=" * 40)
        logger.info("")

        for category, boundary_list in self.available_boundaries.items():
            logger.info("%s:", category)

            # Format the list nicely - wrap long lists
            if len(boundary_list) <= 5:
                # For short lists, show all on one line
                logger.info("  - %s", ", ".join(boundary_list))
            else:
                # For longer lists, show first few and count
                displayed = boundary_list[:5]
                remaining = len(boundary_list) - 5
                logger.info("  - %s", ", ".join(displayed))
                if remaining > 0:
                    logger.info("    ... and %d more options", remaining)

    def reset(self) -> None:
        """Reset the DataCatalog instance to its initial state.

        This method clears any deprecated mutable state and resets the instance
        to its original state. The catalogs themselves remain loaded and available.

        Note: With thread-safe design, there is minimal mutable state to reset.
        This method is maintained for backward compatibility.

        """
        # Clear deprecated _catalog_key if it exists (backward compatibility)
        if hasattr(self, "_catalog_key"):
            self._catalog_key = None


def _get_closest_options(val, valid_options, cutoff=0.59):
    """If the user inputs a bad option, find the closest option from a list of valid options

    Parameters
    ----------
    val : str
        User input
    valid_options  list
        Valid options for that key from the catalog
    cutoff : a float in the range [0, 1]
        See difflib.get_close_matches
        Possibilities that don't score at least that similar to word are ignored.

    Returns
    -------
    closest_options : list or None
        List of best guesses, or None if nothing close is found

    """

    # Perhaps the user just capitalized it wrong?
    is_it_just_capitalized_wrong = [
        i for i in valid_options if val.lower() == i.lower()
    ]
    if len(is_it_just_capitalized_wrong) > 0:
        return is_it_just_capitalized_wrong

    # Perhaps the input is a substring of a valid option?
    is_it_a_substring = [i for i in valid_options if val.lower() in i.lower()]
    if len(is_it_a_substring) > 0:
        return is_it_a_substring

    # Use difflib package to make a guess for what the input might have been
    # For example, if they input "statistikal" instead of "Statistical", difflib will find "Statistical"
    # Change the cutoff to increase/decrease the flexibility of the function
    maybe_difflib_can_find_something = difflib.get_close_matches(
        val, valid_options, cutoff=cutoff
    )
    if len(maybe_difflib_can_find_something) > 0:
        return maybe_difflib_can_find_something

    return None
