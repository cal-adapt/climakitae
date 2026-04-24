"""Validator for cadcat_preview (internal/pre-release) catalog parameters.

The ``cadcat_preview`` catalog is an umbrella for internal or pre-release
datasets that have not yet landed in the public cadcat collection. The first
member is sup3rcc (NREL's super-resolved 4km CONUS data). Users opt in by
passing ``catalog="cadcat_preview"`` to their query; if the catalog failed to
load (e.g. missing credentials in Phase 2), the validator will not be
instantiated because ``DatasetFactory`` checks catalog availability first.

Notes
-----
Phase 1 of this work disables several processors that assume the cadcat
schema or WRF/LOCA2-specific metadata:

- ``warming_level`` — relies on GWL lookup tables keyed on WRF/LOCA2 source
  IDs; sup3r models are not in those tables yet.
- ``bias_adjust_model_to_station`` — hardcoded to WRF + t2.
- ``filter_unadjusted_models`` — hardcoded lists of WRF-BA-adjusted and
  non-adjusted models; meaningless for sup3r.

These can be re-enabled in a later phase once sup3r-aware support is added.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from climakitae.core.constants import CATALOG_CADCAT_PREVIEW, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.param_validation.abc_param_validation import (
    ParameterValidator,
    register_catalog_validator,
)

# Module logger
logger = logging.getLogger(__name__)


@register_catalog_validator(CATALOG_CADCAT_PREVIEW)
class CadcatPreviewValidator(ParameterValidator):
    """Validator for the cadcat_preview catalog.

    Mirrors the ``DataValidator`` (cadcat) shape but opts out of processors
    that assume cadcat/WRF/LOCA2 schema.

    Parameters
    ----------
    catalog : DataCatalog
        The DataCatalog singleton to validate against.

    """

    def __init__(self, catalog: DataCatalog):
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
        self.catalog = catalog.preview
        self.invalid_processors = [
            "warming_level",
            "bias_adjust_model_to_station",
            "filter_unadjusted_models",
        ]
        logger.debug(
            "CadcatPreviewValidator initialized for catalog_preview with keys: %s",
            list(self.catalog.df.columns) if hasattr(self.catalog, "df") else "unknown",
        )

    def get_default_processors(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Get default processors for the cadcat_preview catalog.

        Unlike cadcat, preview data does not default to
        ``filter_unadjusted_models`` (sup3r has no adjusted/unadjusted
        variants). Leap-day dropping and time-based concatenation are kept
        because they apply to any gridded time series.

        Parameters
        ----------
        query : Dict[str, Any]
            The current query containing user parameters.

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping processor names to their default configurations.

        """
        defaults = super().get_default_processors(query)
        defaults["drop_leap_days"] = "yes"

        concat_dim = "time"
        experiment_id = query.get("experiment_id", UNSET)
        match experiment_id:
            case str():
                if (
                    "historical" in experiment_id.lower()
                    or "reanalysis" in experiment_id.lower()
                ):
                    concat_dim = "sim"
            case list() | tuple():
                if not any("ssp" in str(item).lower() for item in experiment_id):
                    concat_dim = "sim"
        defaults["concat"] = concat_dim
        return defaults

    def is_valid_query(self, query: Dict[str, Any]) -> Dict[str, Any] | None:
        """Validate a cadcat_preview query.

        Parameters
        ----------
        query : Dict[str, Any]
            The query to validate.

        Returns
        -------
        Dict[str, Any] | None
            The validated query if valid, ``None`` otherwise.

        """
        logger.debug("Validating cadcat_preview query: %s", query)
        initial_checks = [
            self._check_query_for_required_keys(query),
            self._check_query_invalid_processors(query),
        ]
        if not all(initial_checks):
            logger.warning("Initial cadcat_preview validation checks failed")
            return None
        result = super()._is_valid_query(query)
        logger.info("cadcat_preview query validation result: %s", bool(result))
        return result

    def _check_query_for_required_keys(self, query: Dict[str, Any]) -> bool:
        """Check the query contains the required keys.

        Required: ``activity_id``, ``table_id``, ``grid_label``, ``variable_id``.

        Parameters
        ----------
        query : Dict[str, Any]
            The query to check.

        Returns
        -------
        bool
            True if all required keys are set.

        """
        required_keys = ("activity_id", "table_id", "grid_label", "variable_id")
        unset_keys = [key for key in required_keys if query.get(key, UNSET) is UNSET]
        if unset_keys:
            logger.warning(
                "cadcat_preview query is missing required keys: %s",
                ", ".join(unset_keys),
            )
            return False
        return True

    def _check_query_invalid_processors(self, query: Dict[str, Any]) -> bool:
        """Reject processors that are not supported for preview data.

        Parameters
        ----------
        query : Dict[str, Any]
            The query to check.

        Returns
        -------
        bool
            True if no invalid processors were requested.

        """
        processes = query.get("processes", {}) or {}
        # ``processes`` is usually a dict in new_core but defensively accept iterable
        requested = processes.keys() if hasattr(processes, "keys") else processes
        for processor in requested:
            if processor in self.invalid_processors:
                logger.warning(
                    "Processor '%s' is not supported for cadcat_preview data. "
                    "See validator docstring for the current list of unsupported "
                    "processors; support may be added in a future phase.",
                    processor,
                )
                return False
        return True
