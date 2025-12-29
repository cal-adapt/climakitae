from .data_access.data_access import DataCatalog
from .derived_variables import (
    get_registry,
    list_derived_variables,
    register_derived,
    register_user_function,
)

CATALOG = DataCatalog()
CATALOG.list_clip_boundaries()

__all__ = [
    "DataCatalog",
    "CATALOG",
    "get_registry",
    "register_derived",
    "register_user_function",
    "list_derived_variables",
]
