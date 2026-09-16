"""Validator for data catalog parameters."""

from __future__ import annotations

import logging
from typing import Any, Dict

from climakitae.core.constants import CATALOG_SUP3RCC, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.param_validation.abc_param_validation import (
    ParameterValidator,
    register_catalog_validator,
)

# Module logger
logger = logging.getLogger(__name__)


@register_catalog_validator(CATALOG_SUP3RCC)
class Sup3rCCValidator(ParameterValidator):
    """Validator for data catalog parameters.

    Parameters
    ----------
    catalog : DataCatalog
        the DataCatalog object to validate against

    """

    def __init__(self, catalog: DataCatalog):
        """Initialize with  catalog of Sup3rCC energy datasets.

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
        self.catalog = catalog.sup3rcc
        self.invalid_processors = [
            "bias_adjust_model_to_station"
            "filter_unadjusted_models",
            "warming_level",
            "concatenate",
        ]
        logger.debug(
            "DataValidator initialized for catalog with keys: %s",
            list(self.catalog.keys()) if hasattr(self.catalog, "keys") else "unknown",
        )

    def get_default_processors(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Get default processors for SUP3RCC catalog.

        Climate model data gets smart concatenation
        based on experiment_id.

        Parameters
        ----------
        query : Dict[str, Any]
            The current query containing user parameters

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping processor names to their default configurations
        """
        defaults = super().get_default_processors(query)

        # Drop leap days by default
        defaults["drop_leap_days"] = "yes"

        # Set default concatenation
        concat_dim = "time"

        # if experiment_id is a string, check if it contains "historical"
        experiment_id = query.get("experiment_id", UNSET)
        match experiment_id:
            case str():
                if (
                    "historical" in experiment_id.lower()
                    or "reanalysis" in experiment_id.lower()
                ):
                    # if it does, we can use "sim" as the default concat dimension
                    concat_dim = "sim"
            case list() | tuple():
                # if experiment_id is a list or tuple, check each element
                # if there are no elements with "ssp" in them then we use the sim approach
                if not any("ssp" in str(item).lower() for item in experiment_id):
                    concat_dim = "sim"

        defaults["concat"] = concat_dim
        return defaults

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
        logger.debug("Validating query: %s", query)
        initial_checks = [
            self._check_query_for_required_keys(query),
        ]
        if not all(initial_checks):
            logger.warning("Initial validation checks failed: %s", initial_checks)
            return None
        result = super()._is_valid_query(query)
        logger.info("Query validation result: %s", bool(result))
        return result

    def _check_query_for_required_keys(self, query: Dict[str, Any]) -> bool:
        """Check if the query contains all required keys.

        Parameters
        ----------
        query : Dict[str, Any]
            The query to check.

        Returns
        -------
        bool
            True if the query contains all required keys, False otherwise.

        """
        logger.debug("Checking for required keys in query: %s", query)
        required_keys = ("table_id", "grid_label", "variable_id")
        unset_keys = [key for key in required_keys if query.get(key, UNSET) is UNSET]
        if unset_keys:
            logger.warning(
                "Query is missing the following required keys: %s",
                ", ".join(unset_keys),
            )
            return False
        return True

