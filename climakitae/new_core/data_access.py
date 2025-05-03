from abc import ABC, abstractmethod

import intake
import pandas as pd

from climakitae.core.constants import UNSET


class DataCatalog(dict):
    """
    Singleton class for managing catalog connections.
    
    This class is a singleton and inhertis from dict.
    """

    _instance = UNSET

    def __new__(cls):
        if cls._instance is UNSET:
            cls._instance = super(DataCatalog, cls).__new__(cls)
            cls._instance.catalog = UNSET
        return cls._instance

    def __init__(self):
        if self.catalog is UNSET:
            self.catalog = None

    def set_catalog(self, catalog: intake.Catalog):
        """Set the catalog."""
        self.catalog = catalog

    def get_catalog(self) -> intake.Catalog:
        """Get the catalog."""
        if self.catalog is None:
            raise ValueError("Catalog not set.")
        return self.catalog


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
