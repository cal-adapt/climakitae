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
