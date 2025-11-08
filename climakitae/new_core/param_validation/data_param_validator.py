"""Validator for data catalog parameters."""

from __future__ import annotations

import warnings
from typing import Any, Dict

from climakitae.core.constants import CATALOG_CADCAT, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.param_validation.abc_param_validation import (
    ParameterValidator,
    register_catalog_validator,
)


@register_catalog_validator(CATALOG_CADCAT)
class DataValidator(ParameterValidator):
    """Validator for data catalog parameters.

    Parameters
    ----------
    catalog : DataCatalog
        the DataCatalog object to validate against

    Attributes
    ----------

    """

    def __init__(self, catalog: DataCatalog):
        """Initialize with  catalog of renewable energy datasets.

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
        """Catalog specific validation for the query.

        Parameters
        ----------
        query : Dict[str, Any]
            The query to validate.

        Returns
        -------
        Dict[str, Any] | None
            The validated query if valid, None otherwise.

        Notes
        -----
        A list of checks that are performed on the query:

        1. Check if the query contains the localize processor.
            Localize is not supported for LOCA2 datasets.

        """
        initial_checks = [self._check_query_for_wrf_and_localize(query)]
        if not all(initial_checks):
            return None
        return super()._is_valid_query(query)

    def _check_query_for_wrf_and_localize(self, query: Dict[str, Any]) -> bool:
        """Check if the query contains the localize processor.

        Localize is not supported for LOCA2 datasets.

        Parameters
        ----------
        query : Dict[str, Any]
            The query to check.

        Returns
        -------
        bool
            True if the query does not contain localize processor, False otherwise.

        """
        if "localize" in query.get("processes", {}).keys():
            if "WRF" not in query.get("activity_id", ""):
                warnings.warn(
                    "\n\nLocalize processor is not supported for LOCA2 datasets."
                    "\nPlease specify '.activity_id(WRF)' in your query.",
                    UserWarning,
                    stacklevel=999,
                )
                return False
            if query.get("variable_id", "") != "t2":
                warnings.warn(
                    f"\n\nLocalize processor is not supported for any variable other than 't2' (Air Temperature at 2m)."
                    f"\nPlease specify '.variable_id('t2')' in your query.",
                    UserWarning,
                    stacklevel=999,
                )
                return False
        return True
