import warnings

import intake
import intake_esm
import pandas as pd
import xarray as xr

from climakitae.core.constants import UNSET
from climakitae.core.paths import (
    BOUNDARY_CATALOG_URL,
    DATA_CATALOG_URL,
    RENEWABLES_CATALOG_URL,
)


class DataCatalog(dict):
    """
    Singleton class for managing catalog connections.

    This class is a singleton that inherits from dict, allowing direct
    dictionary-style access to catalogs.
    """

    _instance = UNSET

    def __new__(cls):
        if cls._instance is UNSET:
            cls._instance = super(DataCatalog, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not getattr(self, "_initialized", False):
            super().__init__()
            self["data"] = intake.open_esm_datastore(DATA_CATALOG_URL)
            self["boundary"] = intake.open_catalog(BOUNDARY_CATALOG_URL)
            self["renewables"] = intake.open_esm_datastore(RENEWABLES_CATALOG_URL)
            self._initialized = True
            self.catalog_key = UNSET

    @property
    def data(self) -> intake_esm.core.esm_datastore:
        """Access data catalog."""
        return self["data"]

    @property
    def boundary(self) -> intake.catalog.Catalog:
        """Access boundary catalog."""
        return self["boundary"]

    @property
    def renewables(self) -> intake_esm.core.esm_datastore:
        """Access renewables catalog."""
        return self["renewables"]

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
                f"Catalog key '{key}' not found. Available keys are: {list(self.keys())}",
                UserWarning,
            )
            warnings.warn("Defaulting to 'data' catalog.", UserWarning)
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

    def get_data(self, query: dict = UNSET) -> dict[str, xr.Dataset]:
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
        return self[self.catalog_key].search(**query).to_dataset_dict()

