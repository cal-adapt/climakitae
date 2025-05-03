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
            self["data"] = intake.open_catalog(DATA_CATALOG_URL)
            self["boundary"] = intake.open_catalog(BOUNDARY_CATALOG_URL)
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

    def set_catalog(self, name, catalog):
        """Set a named catalog."""
        self[name] = catalog


class DataAccessor(ABC):
    """Abstract base class for data access."""

    @abstractmethod
    def get_data(self, parameters):
        """Get data from the source."""
        pass


class IntakeAccessor(DataAccessor):
    """Data accessor using Intake."""

    def __init__(self, catalog_df: pd.DataFrame):
        """
        Initialize with a catalog of datasets.

        Parameters
        ----------
        catalog : pd.DataFrame
            Catalog of datasets
        """
        self.catalog = catalog_df

    def get_data(self, query: dict) -> dict:
        """
        Get data from the source.

        Parameters
        ----------
        parameters : dict
            Parameters for data access

        Returns
        -------

            Data object
        """
        # Implement the logic to access data using the catalog and parameters
        datasets = self.catalog.search(**query).to_dataset_dict(
            xarray_open_kwargs={"consolidated": True},
            storage_options={"anon": True},
        )
