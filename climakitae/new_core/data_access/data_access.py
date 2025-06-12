"""
Data access module for ClimakitAE.

This module provides a singleton DataCatalog class for managing connections to
various climate data catalogs including boundary, renewables, and general climate
datasets. The DataCatalog class offers a unified interface for accessing and
querying multiple intake catalogs with support for method chaining and dynamic
catalog management.

Classes
-------
DataCatalog
    Singleton class that inherits from dict and manages catalog connections.
    Provides properties for accessing specific catalogs and methods for
    querying and retrieving climate datasets.
"""

import warnings

import geopandas as gpd
import intake
import intake_esm
import pandas as pd
import xarray as xr

from climakitae.core.constants import (
    CATALOG_BOUNDARY,
    CATALOG_DATA,
    CATALOG_RENEWABLES,
    UNSET,
)
from climakitae.core.paths import (
    BOUNDARY_CATALOG_URL,
    DATA_CATALOG_URL,
    RENEWABLES_CATALOG_URL,
    STATIONS_CSV_PATH,
)
from climakitae.util.utils import read_csv_file

from .boundaries import Boundaries


class DataCatalog(dict):
    """
    Singleton class for managing catalog connections to climate data sources.

    This class implements the singleton pattern and inherits from dict to provide
    a unified interface for accessing multiple climate data catalogs. It manages
    connections to boundary, renewables, and general climate datasets through
    intake and intake-esm catalogs, offering convenient properties and methods
    for data querying and retrieval.

    The class automatically initializes connections to predefined catalogs and
    supports dynamic addition of new catalogs. Method chaining is supported for
    fluent API usage.

    Attributes
    ----------
    catalog_key : str or UNSET
        The currently selected catalog key for data operations. Defaults to UNSET
        until explicitly set via set_catalog_key().

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
    set_catalog_key(key)
        Set the active catalog for subsequent operations.
    set_catalog(name, catalog)
        Add a new catalog to the collection.
    get_data(query)
        Retrieve data from the active catalog using query parameters.

    Notes
    -----
    This class implements the singleton pattern, ensuring only one instance
    exists throughout the application lifecycle. Multiple calls to DataCatalog()
    will return the same instance.

    The class automatically handles catalog initialization and provides sensible
    defaults when invalid catalog keys are specified.
    """

    _instance = UNSET

    def __new__(cls):
        """Override __new__ to implement singleton pattern."""
        if cls._instance is UNSET:
            cls._instance = super(DataCatalog, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the DataCatalog instance."""
        if not getattr(self, "_initialized", False):
            super().__init__()
            self[CATALOG_DATA] = intake.open_esm_datastore(DATA_CATALOG_URL)
            self[CATALOG_BOUNDARY] = intake.open_catalog(BOUNDARY_CATALOG_URL)
            self[CATALOG_RENEWABLES] = intake.open_esm_datastore(RENEWABLES_CATALOG_URL)
            self.catalog_df = self.merge_catalogs()
            stations_df = read_csv_file(STATIONS_CSV_PATH)
            self["stations"] = gpd.GeoDataFrame(
                stations_df,
                crs="EPSG:4326",
                geometry=gpd.points_from_xy(stations_df.LON_X, stations_df.LAT_Y),
            )

            self._initialized = True
            self.catalog_key = UNSET
            # Initialize boundaries with lazy loading
            self._boundaries = UNSET

    @property
    def data(self) -> intake_esm.core.esm_datastore:
        """Access data catalog."""
        return self[CATALOG_DATA]

    @property
    def boundary(self) -> intake.catalog.Catalog:
        """Access boundary catalog."""
        return self[CATALOG_BOUNDARY]

    @property
    def renewables(self) -> intake_esm.core.esm_datastore:
        """Access renewables catalog."""
        return self[CATALOG_RENEWABLES]

    @property
    def boundaries(self) -> Boundaries:
        """Access boundaries data with lazy loading."""
        if self._boundaries is UNSET:
            self._boundaries = Boundaries(self.boundary)
        return self._boundaries

    def merge_catalogs(self) -> pd.DataFrame:
        """
        Merge the intake catalogs for data and renewables into a single DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the merged data from both catalogs.
        """
        ren_df = self.renewables.df
        data_df = self.data.df
        ren_df["catalog"] = CATALOG_RENEWABLES
        data_df["catalog"] = CATALOG_DATA
        ret = pd.concat([ren_df, data_df], ignore_index=True)
        return ret

    def set_catalog_key(self, key: str) -> "DataCatalog":
        """
        Set the catalog key for accessing a specific catalog.

        Parameters
        ----------
        key : str
            Key of the catalog to set.

        Returns
        -------
        DataCatalog
            The current instance of DataCatalog allowing method chaining.

        Raises
        ------
        ValueError
            If the catalog key is not found in the available catalogs.
        """
        if key not in self:
            warnings.warn(
                f"\n\nCatalog key '{key}' not found."
                f"\nAvailable catalogs keys are: {list(self.keys())}"
                f"\nDefulting to 'data' catalog.\n\n"
            )
            key = "data"
        self.catalog_key = key
        return self

    def set_catalog(self, name: str, catalog: str) -> "DataCatalog":
        """
        Set a named catalog.

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

    def get_data(self, query: dict) -> dict[str, xr.Dataset]:
        """
        Get data from the catalog.

        Parameters
        ----------
        name : str
            Name of the catalog to access.
        query : dict, optional
            Query parameters for filtering data.

        Returns
        -------
        dict[str, xr.Dataset]
            The requested dataset(s) from the catalog.
        """
        print(f"Querying {self.catalog_key} catalog with query: {query}")
        # if any(isinstance(v, list) for v in query.values()):
        #     # query contains a list, which is not supported by intake
        #     for key, value in query.items():
        #         if isinstance(value, list):
        #             # Convert list to a comma-separated string
        #             query[key] = ",".join(value)
        return (
            self[self.catalog_key]
            .search(**query)
            .to_dataset_dict(
                zarr_kwargs={"consolidated": True},
                storage_options={"anon": True},
                progressbar=False,
            )
        )

    def list_clip_boundaries(self) -> dict[str, list]:
        """
        List all available boundary options for clipping operations.

        This is a convenience method that provides direct access to boundary
        options without needing to instantiate a Clip processor.

        Returns
        -------
        Dict[str, list]
            Dictionary with boundary categories as keys and lists of available
            boundary names as values

        Examples
        --------
        >>> catalog = DataCatalog()
        >>> boundaries = catalog.list_clip_boundaries()
        >>> print(boundaries["states"])
        ['AZ', 'CA', 'CO', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY']
        """
        boundary_dict = self.boundaries.boundary_dict()

        # Create a clean dictionary with boundary categories and their available options
        available_boundaries = {}

        for category, lookups in boundary_dict.items():
            # Skip special categories that don't represent actual boundary data
            if category in ["none", "lat/lon"]:
                continue

            # Convert keys to a sorted list for better presentation
            boundary_keys = sorted(list(lookups.keys()))
            available_boundaries[category] = boundary_keys

        return available_boundaries

    def print_clip_boundaries(self) -> None:
        """
        Print all available boundary options for clipping in a user-friendly format.

        This method provides a nicely formatted output showing all boundary
        categories and their available options for clipping operations.

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
            boundaries = self.list_clip_boundaries()
        except Exception as e:
            print(f"Error accessing boundary data: {e}")
            return

        print("Available Boundary Options for Clipping:")
        print("=" * 40)
        print()

        for category, boundary_list in boundaries.items():
            print(f"{category}:")

            # Format the list nicely - wrap long lists
            if len(boundary_list) <= 5:
                # For short lists, show all on one line
                print(f"  - {', '.join(boundary_list)}")
            else:
                # For longer lists, show first few and count
                displayed = boundary_list[:5]
                remaining = len(boundary_list) - 5
                print(f"  - {', '.join(displayed)}")
                if remaining > 0:
                    print(f"    ... and {remaining} more options")
            print()

    def reset(self):
        """
        Reset the DataCatalog instance to its initial state.

        This method clears all catalogs and reinitializes the DataCatalog
        instance, effectively resetting it to its original state.
        """
        self.catalog_key = UNSET
