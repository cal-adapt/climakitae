from abc import ABC, abstractmethod

import intake
import pandas as pd

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
    def data(self):
        """Access data catalog."""
        return self["data"]

    @property
    def boundary(self):
        """Access boundary catalog."""
        return self["boundary"]

    @property
    def renewables(self):
        """Access renewables catalog."""
        return self["renewables"]

    def set_catalog_key(self, key: str):
        """Set the catalog key for accessing a specific catalog."""
        if key not in self:
            raise ValueError(f"Catalog '{key}' not found.")
        self.catalog_key = key
        return self

    def set_catalog(self, name: str, catalog: str):
        """Set a named catalog."""
        self[name] = intake.open_esm_datastore(catalog)

    def get_data(self, query: dict = UNSET):
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
        intake.catalog.local.Catalog
            The requested catalog.
        """
        return self[self.catalog_key].search(query)
