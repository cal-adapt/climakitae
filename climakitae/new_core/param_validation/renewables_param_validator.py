"""Validator for renewable energy dataset parameters."""

from __future__ import annotations

import logging
from typing import Any, Dict

from climakitae.core.constants import CATALOG_REN_ENERGY_GEN, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.param_validation.abc_param_validation import (
    ParameterValidator,
    register_catalog_validator,
)

# Module logger
logger = logging.getLogger(__name__)


@register_catalog_validator(CATALOG_REN_ENERGY_GEN)
class RenewablesValidator(ParameterValidator):
    """Validator for renewable energy dataset parameters.

    Parameters
    ----------
    catalog : str
        path to the renewables dataset catalog

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
            "installation": UNSET,
            "activity_id": UNSET,
            "institution_id": UNSET,
            "source_id": UNSET,
            "experiment_id": UNSET,
            "table_id": UNSET,
            "grid_label": UNSET,
            "variable_id": UNSET,
        }
        self.catalog = catalog.renewables
        self.invalid_processors = []

        logger.debug("RenewablesValidator initialized for renewables catalog")

    def get_default_processors(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Get default processors for renewables catalog.

        Renewables data uses the same defaults as CADCAT since it's also
        climate model data.

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

        # Add default filtering for climate model data
        defaults["filter_unadjusted_models"] = "yes"

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

        1. Check if the query contains the warming_level processor.
            Warming level is not supported for renewables datasets.

        """
        logger.debug("Validating renewables query: %s", query)
        initial_checks = [
            self._check_query_for_warming_level_processor(query),
        ]
        if not all(initial_checks):
            logger.warning("Initial validation checks failed: %s", initial_checks)
            return None
        result = super()._is_valid_query(query)
        logger.info("Renewables query validation result: %s", bool(result))
        return result

    def _check_query_for_warming_level_processor(self, query: Dict[str, Any]) -> bool:
        """Check if the query contains the warming_level processor.

        Warming level subsetting is not supported for renewable energy datasets
        since renewables data is not necessarily aligned with the same warming
        level trajectories as climate model data.

        Parameters
        ----------
        query : Dict[str, Any]
            The query to check.

        Returns
        -------
        bool
            True if the query does not contain warming_level processor,
            False otherwise.

        """
        logger.debug(
            "Checking for warming_level processor in query: %s",
            query.get("processes"),
        )
        if "warming_level" in query.get("processes", {}).keys():
            msg = (
                "Warming level processor is not supported for renewable energy datasets. "
                "Please use time_slice processor instead to specify a time range."
            )
            logger.warning(msg)
            return False
        return True
