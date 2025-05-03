"""
Parameter validation module.

This module contains classes and functions for validating parameters
used in the climakitae package. It includes a base class for parameter
validation and a specific implementation for various datasets.

The `ParameterValidator` class is an abstract base class that defines
the interface for parameter validation. It requires subclasses to
implement the `is_valid` method, which takes a dictionary of parameters
and returns a boolean indicating whether the parameters are valid.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict

import intake

from climakitae.core.constants import UNSET
from climakitae.new_core.data_access import DataCatalog
from climakitae.new_core.param_validation_tools import _get_closest_options


class ParameterValidator(ABC):
    """Abstract base class for parameter validation."""

    @abstractmethod
    def is_valid(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate parameters and return processed parameters.

        Parameters
        ----------
        parameters : Dict[str, Any]
            Parameters to validate

        Returns
        -------
        Dict[str, Any]
            Processed parameters

        Raises
        ------
        ValueError
            If parameters are invalid
        """

    def _convert_frequency(self, frequency: str) -> str:
        """
        Convert frequency to table_id.

        Parameters
        ----------
        frequency : str
            Frequency to convert

        Returns
        -------
        str
            Converted table_id
        """
        frequency_mapping = {
            "hourly": "1hr",
            "daily": "1day",
            "monthly": "1mon",
            "yearly": "1yr",
        }  # TODO this goes in a constants file
        return frequency_mapping.get(frequency, UNSET)

    def _convert_resolution(self, resolution: str) -> str:
        """
        Convert resolution to grid_label.

        Parameters
        ----------
        resolution : str
            Resolution to convert

        Returns
        -------
        str
            Converted grid_label
        """
        resolution_mapping = {
            "3 km": "d03",
            "9 km": "d02",
            "45 km": "d01",
        }
        return resolution_mapping.get(resolution, UNSET)


class RenewablesValidator(ParameterValidator):
    """
    Validator for renewable energy dataset parameters.

    Parameters
    ----------
    catalog : str
        path to the renewables dataset catalog

    Attributes
    ----------
    """

    def __init__(self, catalog: DataCatalog):
        """
        Initialize with  catalog of renewable energy datasets.

        Parameters
        ----------
        catalog : DataCatalog
            Catalog of datasets
        """
        self.all_catalog_keys = {
            "installation": UNSET,
            "activity_id": UNSET,
            "institution_id": UNSET,
            "source_id": UNSET,
            "experiment_id": UNSET,
            "table_id": UNSET,
            "grid_label": UNSET,
        }
        self.catalog = catalog.renewables

    def populate_catalog_keys(self, query: Dict[str, Any]) -> None:
        """
        Populate the catalog keys with the values from the query.

        Parameters
        ----------
        query : Dict[str, Any]
            query to populate

        Returns
        -------
        None
        """
        self.all_catalog_keys.update(query)
        self.all_catalog_keys["table_id"] = self._convert_frequency(query["frequency"])
        self.all_catalog_keys["grid_label"] = self._convert_resolution(
            query["resolution"]
        )
        # remove any unset values
        self.all_catalog_keys = {
            k: v for k, v in self.all_catalog_keys.items() if v is not UNSET
        }

    def is_valid_query(self, query: Dict[str, Any]) -> bool:
        """
        Validate renewable energy dataset query.

        Parameters
        ----------
        query : Dict[str, Any]
            query to validate

        Returns
        -------
        bool
            True if the query is valid, False otherwise

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        # convert user input to Renewables keys
        self.populate_catalog_keys(query)

        # check if the catalog keys can be found
        subset = self.catalog.search(**self.all_catalog_keys)
        if len(subset) != 0:
            print(
                f"Found {len(subset)} datasets matching the query: {self.all_catalog_keys}"
            )
            return True

        # dataset not found
        # find closest match to each provided key
        for key, value in self.all_catalog_keys.items():
            if value != UNSET:
                # check if the key is in the catalog
                if key not in self.catalog.df.columns:
                    raise ValueError(f"Key {key} not found in catalog")

                # check if the value is in the catalog
                valid_options = self.catalog.df[key].unique()
                closest_options = _get_closest_options(value, valid_options)
                if closest_options:
                    print(
                        f"Did you mean {closest_options} for {key}? "
                        f"Please check your input."
                    )
                else:
                    warnings.warn(f"Invalid value {value} for key {key}", UserWarning)

        return False
