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

from abc import ABC, abstractmethod
from typing import Any, Dict

import intake

from climakitae.core.constants import UNSET

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
    """
    Abstract base class for parameter validation.

    The user query contains the parameters to be validated.
    These parameters fall under the following categories:
    - catalog variables: variables that are used to identify the dataset in the catalog
        - The logic for this is implemented in `get_data_options` I think
        - The motivation is to make the interface match the GUI
        - !!! IS THIS NECESSARY? !!!
            - the GUI is for folks not technically inclined
            - we don't need this logic in the dev/scientist interface
            -
    - processing variables: variables that are used to process the dataset
        - the logic for this is rag tag at best
    """

    def __init__(self):
        self.catalog_path = "climakitae/data/catalogs.csv"
        self.catalog = None
        self.load_catalog_df()

    @abstractmethod
    def is_valid_query(self, query: Dict[str, Any]) -> Dict[str, Any] | None:
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
