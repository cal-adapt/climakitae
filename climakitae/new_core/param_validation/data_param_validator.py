"""
Validator for data catalog parameters.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

from climakitae.core.constants import UNSET
from climakitae.new_core.data_access import DataCatalog
from climakitae.new_core.param_validation.abc_param_validation import (
    ParameterValidator,
    register_validator,
)
from climakitae.new_core.param_validation.param_validation_tools import (
    _get_closest_options,
)


@register_validator("data")
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

    def is_valid_query(self, query: Dict[str, Any]) -> Dict[str, Any] | None:
        """
        Validate data catalog query.

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
                    warnings.warn(
                        f"Key {key} not found in catalog. Did you specify the correct catalog?"
                    )
                    continue  # skip to the next key

                # check if the value is in the catalog
                valid_options = self.catalog.df[key].unique()
                closest_options = _get_closest_options(value, valid_options)
                if closest_options is not None:
                    print(
                        f"Did you mean {closest_options} for {key}? "
                        f"Please check your input."
                    )
                else:
                    warnings.warn(f"Invalid value {value} for key {key}")
                    print(
                        f"Valid options for {key} are: {valid_options}. "
                        f"Please check your input."
                    )
        return None
