"""
Validator for data catalog parameters.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

from climakitae.core.constants import CATALOG_DATA, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.param_validation.abc_param_validation import (
    ParameterValidator,
    register_catalog_validator,
)


@register_catalog_validator(CATALOG_DATA)
class DataValidator(ParameterValidator):
    """
    Validator for data catalog parameters.

    Parameters
    ----------
    catalog : DataCatalog
        the DataCatalog object to validate against

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
            "activity_id": UNSET,
            "institution_id": UNSET,
            "source_id": UNSET,
            "experiment_id": UNSET,
            "table_id": UNSET,
            "grid_label": UNSET,
            "variable_id": UNSET,
        }
        self.catalog = catalog.data

    def is_valid_query(self, query: Dict[str, Any]) -> Dict[str, Any] | None:
        return super()._is_valid_query(query)

    def _check_user_input(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        This is where the a lot of validation logic goes for user inputs like:
        - Station Data

        Parameters
        ----------
        user_input : Dict[str, Any]
            User input to validate.

        Returns
        -------
        Dict[str, Any]
            Validated user input.
        """

        checks = [
            (self._contains_station_data(), self._check_valid_station),
        ]

    def _contains_station_data(self) -> bool:
        """
        Check if the query contains station data.

        Returns
        -------
        bool
            True if station data is present, False otherwise.
        """
        return "localize" in self.query

    def _check_valid_station(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the station data in the query.

        Parameters
        ----------
        query : Dict[str, Any]
            Query to validate.

        Returns
        -------
        Dict[str, Any]
            Validated query.
        """
        if "station_data" not in query:
            warnings.warn("No station data provided in the query.")
            return query

        station_data = query["station_data"]
        if not isinstance(station_data, dict):
            raise ValueError("Station data must be a dictionary.")

        # Additional validation logic can be added here

        return query
