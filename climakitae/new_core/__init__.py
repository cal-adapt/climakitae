"""New core module for ClimakitAE climate data access and processing.

This module provides the modern interface for accessing and processing climate
data from the Cal-Adapt Analytics Engine. It includes:

- User interface for fluent query building (ClimateData class)
- Factory for creating configured Dataset objects
- Data access layer with catalog management
- Parameter validation framework
- Extensible data processing pipeline

Key Entry Points
----------------
- ``ClimateData``: Main user-facing class for building climate data queries
- ``DataCatalog``: Singleton for managing data catalog access
- ``DatasetFactory``: Factory for creating Dataset objects with validation

Example
-------
>>> from climakitae.new_core.user_interface import ClimateData
>>> data = (ClimateData()
...         .catalog("cadcat")
...         .variable("tasmax")
...         .experiment("ssp245")
...         .get())

Notes
-----
This module provides lazy-loaded access to boundary data. The DataCatalog
singleton is created on first access, and boundary data is only loaded
when explicitly requested (e.g., via ``show_boundary_options()``).
"""

from .data_access.data_access import DataCatalog

# Lazy singleton access - DataCatalog is created on first use
# Boundary data is NOT loaded at import time for performance
CATALOG = DataCatalog()
