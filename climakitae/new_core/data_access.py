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

import intake
import intake_esm
import xarray as xr

from climakitae.core.constants import CATALOG_DATA, CATALOG_RENEWABLES, UNSET
from climakitae.core.paths import (
    BOUNDARY_CATALOG_URL,
    DATA_CATALOG_URL,
    RENEWABLES_CATALOG_URL,
)

CATALOG_BOUNDARY = "boundary"


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
            self._initialized = True
            self.catalog_key = UNSET

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
        return (
            self[self.catalog_key]
            .search(**query)
            .to_dataset_dict(
                zarr_kwargs={"consolidated": True},
                storage_options={"anon": True},
                progressbar=False,
            )
        )
