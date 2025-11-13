"""Validator for renewable energy dataset parameters."""

from __future__ import annotations

from typing import Any, Dict

from climakitae.core.constants import CATALOG_HDP, UNSET
from climakitae.new_core.data_access.data_access import DataCatalog
from climakitae.new_core.param_validation.abc_param_validation import (
    ParameterValidator,
    register_catalog_validator,
)


@register_catalog_validator(CATALOG_HDP)
class HDPValidator(ParameterValidator):
    """Validator for historical data platform parameters.

    Parameters
    ----------
    catalog : str
        path to the hdp dataset catalog

    Attributes
    ----------

    """

    def __init__(self, catalog: DataCatalog):
        """Initialize with  catalog of historical data platform datasets.

        Parameters
        ----------
        catalog : DataCatalog
            Catalog of datasets

        """
        super().__init__()
        self.all_catalog_keys = {
            "network_id": UNSET,
            "station_id": UNSET,
        }
        self.catalog = catalog.hdp

    def is_valid_query(self, query: Dict[str, Any]) -> Dict[str, Any] | None:
        return super()._is_valid_query(query)
