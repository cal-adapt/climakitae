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
            self["boundary"] = intake.open_esm_datastore(BOUNDARY_CATALOG_URL)
            self["renewables"] = intake.open_esm_datastore(RENEWABLES_CATALOG_URL)
            self._initialized = True

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

    def set_catalog(self, name: str, catalog: str):
        """Set a named catalog."""
        self[name] = intake.open_esm_datastore(catalog)

    def get_dataset_dict(self, name: str, query: dict = UNSET):
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
        if name not in self:
            raise ValueError(f"Catalog '{name}' not found.")
        return self[name].search(query) if query else self[name]
