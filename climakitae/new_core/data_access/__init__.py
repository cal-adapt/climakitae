"""Data access module for the new core functionality.

This subpackage provides components for accessing climate data and boundary
information from S3-hosted catalogs.

Key Components
--------------
- ``DataCatalog``: Thread-safe singleton for managing intake-esm catalog access
- ``Boundaries``: Lazy-loading manager for geospatial boundary data

See Also
--------
climakitae.new_core.user_interface : Main user interface module
climakitae.new_core.dataset_factory : Factory for creating datasets
"""

from .boundaries import Boundaries
from .data_access import DataCatalog

__all__ = ["DataCatalog", "Boundaries"]
