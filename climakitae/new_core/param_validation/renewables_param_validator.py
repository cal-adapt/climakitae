"""
Validator for renewable energy dataset parameters.
"""

from __future__ import annotations

from typing import Any, Dict

from climakitae.core.constants import CATALOG_RENEWABLES, UNSET
from climakitae.new_core.data_access import DataCatalog
from climakitae.new_core.param_validation.abc_param_validation import (
    ParameterValidator,
    register_validator,
)


@register_validator(CATALOG_RENEWABLES)
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
            "variable_id": UNSET,
        }
        self.catalog = catalog.renewables

    def is_valid_query(self, query: Dict[str, Any]) -> Dict[str, Any] | None:
        return super()._is_valid_query(query)
