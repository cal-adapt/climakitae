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

from climakitae.core.constants import PROC_KEY, UNSET
from climakitae.new_core.param_validation.param_validation_tools import (
    _get_closest_options,
)

_CATALOG_VALIDATOR_REGISTRY = {}
_PROCESSOR_VALIDATOR_REGISTRY = {}


def register_catalog_validator(name: str):
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
        _CATALOG_VALIDATOR_REGISTRY[name] = cls
        return cls

    return decorator


def register_processor_validator(name: str):
    """
    Decorator to register a processor validator class.

    Parameters
    ----------
    name : str
        Name of the processor validator

    Returns
    -------
    function
        Decorated class
    """

    def decorator(cls):
        _PROCESSOR_VALIDATOR_REGISTRY[name] = cls
        return cls

    return decorator


class ParameterValidator(ABC):
    """
    Abstract base class for parameter validation.

    The user query contains the parameters to be validated.
    These parameters fall under the following categories:
    - processing variables: variables that are used to process the dataset
        - the logic for this is rag tag at best
    """

    def __init__(self):
        self.catalog_path = "climakitae/data/catalogs.csv"
        self.catalog = UNSET
        self.all_catalog_keys = UNSET
        self.load_catalog_df()

    @abstractmethod
    def is_valid_query(self, query: Dict[str, Any]) -> Dict[str, Any] | None:
        """
        Validate the query parameters.

        Parameters
        ----------
        query : Dict[str, Any]
            Query parameters to validate

        Returns
        -------
        Dict[str, Any] | None
            Validated query parameters or None if invalid
        """
        pass

    def _is_valid_query(self, query: Dict[str, Any]) -> Dict[str, Any] | None:
        """
        Validate parameters and return processed parameters.

        Parameters
        ----------
        parameters : Dict[str, Any]
            Parameters to validate

        Returns
        -------
        Dict[str, Any] | None
            Processed parameters if valid, otherwise None and print warning messages

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        # convert user input to keys
        self.populate_catalog_keys(query)
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
            print(f"Found {len(subset)} datasets matching your query.")
            print(f"Checking processes ...")
            return self.all_catalog_keys if self._has_valid_processes(query) else None

        # dataset not found
        # find closest match to each provided key
        print("No datasets found matching your query. Checking for valid options...")
        df = self.catalog.df.copy()
        last_key = None
        for key, value in self.all_catalog_keys.items():
            # don't check unset values
            if value is UNSET:
                continue

            # check if the key is in the catalog
            if key not in df.columns:
                warnings.warn(
                    f"Key {key} not found in catalog. Did you specify the correct catalog?"
                )
                continue  # skip to the next key
            # subset the dataframe to the key
            remaining_key_values = df[key].unique()
            df = df[df[key] == value]
            if df.empty:
                # this means no datasets were found for this key.
                # this can happen for two reasons:
                # 1. the user made a typo in the key or value
                # 2. the requested dataset does not exist in the catalog

                # start by checking if the value is in the catalog
                if value not in self.catalog.df[key].unique():
                    # the value is not in the catalog, check for closest options
                    print(f"Could not find any datasets with {key} = {value}.")
                    closest_options = _get_closest_options(
                        value, self.catalog.df[key].unique()
                    )
                    if closest_options is not None:
                        # probably a typo in the value
                        warnings.warn(
                            f"\n\nDid you mean one of these options for {key}: {closest_options}?"
                        )
                    else:
                        # no close matches found
                        warnings.warn(
                            f"\n\nNo close matches found for {key} = {value}. "
                            "\nBased on your query, the available options for this key are: "
                            f"{remaining_key_values}."
                        )
                else:
                    # the value is in the catalog, but no datasets were found
                    warnings.warn(
                        f"\n\nNo datasets found for {key} = {value}. "
                        f"\n\nMost likely, this is because the dataset you requested "
                        f"\n    does not exist in the catalog. "
                        f"\nThis most often happens when searching for conflicting time"
                        f"\n    or spatial resolutions"
                        f"\n\nIn this case, it appears that there is a conflict between {key} and {last_key}. "
                        f"\nYour options for {key} are: {remaining_key_values}. "
                        f"\nThis is constrained by the earlier key {last_key} = {self.all_catalog_keys[last_key]}. "
                        f"\nPlease check your query and try again. "
                        f"\n\nTo explore available options, "
                        f"\n please use the `show_*_options()` methods."
                    )
                break
            else:
                print(f"Found {len(df)} datasets with {key} = {value}.")
            last_key = key
            # check if the value is in the catalog
        return None

    def _has_valid_processes(self, query: Dict[str, Any]) -> Dict[str, Any] | None:
        """
        Validate the processor parameters.

        Loop through keys in query['processes'] and check if they are in the processor
        validator registry. If they are, call the processor validator and provide the
        value. Otherwise, warn the user that the processor input has not been validated.

        Parameters
        ----------
        query : Dict[str, Any]
            Query parameters to validate

        Returns
        -------
        Dict[str, Any] | None
            Processed parameters if valid, otherwise None and print warning messages
        """

        # loop through keys in query['processes']
        # and check if they are in the processor validator registry
        # if they are, call the processor validator
        # otherwise warn the user that the processor input has not been validated
        for key, value in query.get(PROC_KEY, {}).items():
            if key in _PROCESSOR_VALIDATOR_REGISTRY:
                valid_value_for_processor = _PROCESSOR_VALIDATOR_REGISTRY[key](value)
                if not valid_value_for_processor:
                    warnings.warn(
                        f"\n\nProcessor {key} with value {value} is not valid. "
                        "\nPlease check the processor documentation for valid options."
                    )
                    return False
            else:
                warnings.warn(
                    f"\n\nProcessor {key} is not registered. "
                    "\nThis processor input has not been validated."
                )

        return True

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

        # remove any unset values
        self.all_catalog_keys = {
            k: v for k, v in self.all_catalog_keys.items() if v is not UNSET
        }

    def load_catalog_df(self):
        """
        load the catalog dataframe and assign to self.catalog_df
        """
        self.catalog_df = intake.open_csv(self.catalog_path).read()

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
