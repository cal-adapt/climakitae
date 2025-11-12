"""Data access module for ClimakitAE.

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

import difflib
import warnings
from typing import Any, Dict

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
from climakitae.core.paths import (
    BOUNDARY_CATALOG_URL,
    DATA_CATALOG_URL,
    RENEWABLES_CATALOG_URL,
    STATIONS_CSV_PATH,
)
from climakitae.new_core.data_access.boundaries import Boundaries
from climakitae.util.utils import read_csv_file


class DataCatalog(dict):
    """Singleton class for managing catalog connections to climate data sources.

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

    def __new__(cls) -> "DataCatalog":
        """Override __new__ to implement singleton pattern.

        Returns
        -------
        DataCatalog
            The singleton instance of DataCatalog.

        """
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
            self.catalog_key = UNSET
            # Initialize boundaries with lazy loading
            self._boundaries = UNSET
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
        """Access boundaries data with lazy loading.

        Returns
        -------
        Boundaries
            The lazy-loading boundaries data manager.

        """
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

    def set_catalog_key(self, key: str) -> "DataCatalog":
        """Set the catalog key for accessing a specific catalog.

        Parameters
        ----------
        key : str
            Key of the catalog to set. Must be one of the available catalog keys.

        Returns
        -------
        DataCatalog
            The current instance of DataCatalog allowing method chaining.

        Warns
        -----
        UserWarning
            If the catalog key is not found in the available catalogs.
            Defaults to 'data' catalog in this case.

        """
        if key not in self:
            warnings.warn(
                f"\n\nCatalog key '{key}' not found."
                f"\nAttempting to find intended catalog key.\n\n",
                stacklevel=999,
            )
            print(f"Available catalog keys: {list(self.keys())}")
            closest = _get_closest_options(key, list(self.keys()))
            if not closest:
                warnings.warn(
                    f"No validator registered for '{key}'. "
                    f"Available options: {list(self.keys())}",
                    stacklevel=999,
                )
                return None

            match len(closest):
                case 0:
                    warnings.warn(
                        f"No validator registered for '{key}'. "
                        "Available options: {list(self._validator_registry.keys())}",
                        stacklevel=999,
                    )
                    return None  # type: ignore[return-value]
                case 1:
                    warnings.warn(
                        f"\n\nUsing closest match '{closest[0]}' for validator '{key}'.",
                        stacklevel=999,
                    )
                    key = closest[0]
                case _:
                    warnings.warn(
                        f"Multiple closest matches found for '{key}': {closest}. "
                        "Please specify a more precise key.",
                        stacklevel=999,
                    )
                    key = None  # type: ignore[return-value]
        self.catalog_key = key
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

    def get_data(self, query: Dict[str, Any]) -> Dict[str, xr.Dataset]:
        """Get data from the catalog.

        This method queries the active catalog using the provided parameters
        and returns the matching datasets as a dictionary.

        Parameters
        ----------
        query : dict
            Query parameters for filtering data. The available parameters
            depend on the active catalog and may include items like 'variable',
            'scenario', 'model', etc.

        Returns
        -------
        dict[str, xr.Dataset]
            The requested dataset(s) from the catalog, keyed by dataset identifiers.

        Notes
        -----
        The catalog_key must be set before calling this method. If not set,
        this will raise an error.

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
            print(f"Error accessing boundary data: {e}")
            return

        print("Available Boundary Options for Clipping:")
        print("=" * 40)
        print()

        for category, boundary_list in self.available_boundaries.items():
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

    def reset(self) -> None:
        """Reset the DataCatalog instance to its initial state.

        This method clears the catalog key and resets the instance to its
        original state. The catalogs themselves remain loaded and available.

        """
        self.catalog_key = UNSET


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
