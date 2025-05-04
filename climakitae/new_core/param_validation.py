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

_VALIDATOR_REGISTRY = {}


def register_validator(name: str):
    """
    Decorator to register a validator class.

    Parameters
    ----------
    name : str
        Name of the validator

    Returns
    -------
    function
        Decorated class
    """

    def decorator(cls):
        _VALIDATOR_REGISTRY[name] = cls
        return cls

    return decorator


class ParameterValidator(ABC):
    """Abstract base class for parameter validation."""

    @abstractmethod
    def is_valid_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters and return processed parameters.

        Parameters
        ----------
        parameters : Dict[str, Any]
            Parameters to validate

        Returns
        -------
        Dict[str, Any]
            Processed parameters if valid, otherwise None and print warning messages

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
            "daily": "day",
            "day": "day",
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


@register_validator("renewables")
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
        super().__init__()
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
        # populate catalog keys with the values from the query
        for key in self.all_catalog_keys.keys():
            self.all_catalog_keys[key] = query.get(key, UNSET)
        self.all_catalog_keys["table_id"] = self._convert_frequency(query["timescale"])
        self.all_catalog_keys["grid_label"] = self._convert_resolution(
            query["resolution"]
        )
        # remove any unset values
        self.all_catalog_keys = {
            k: v for k, v in self.all_catalog_keys.items() if v is not UNSET
        }

    def is_valid_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate renewable energy dataset query.

        Parameters
        ----------
        query : Dict[str, Any]
            query to validate

        Returns
        -------
        Dict[str, Any]
            Processed parameters if valid, otherwise None and print warning messages

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        # convert user input to Renewables keys
        print("populating catalog keys")
        self.populate_catalog_keys(query)
        print(self.all_catalog_keys)
        # check if the catalog keys can be found

        try:
            subset = self.catalog.search(**self.all_catalog_keys)
        except ValueError as e:
            # no datasets found or invalid query
            warnings.warn(
                f"Query did not match any datasets: {e}\n\nSearching for close matches...",
                UserWarning,
            )
        if len(subset) != 0:
            print(
                f"Found {len(subset)} datasets matching the query: {self.all_catalog_keys}"
            )
            return self.all_catalog_keys

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

        return None
