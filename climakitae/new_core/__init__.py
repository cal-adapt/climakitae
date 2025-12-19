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
This module initializes boundary data lookups on import to enable the
``show_boundaries()`` functionality. This may add a small delay to the
initial import.
"""

from .data_access.data_access import DataCatalog

CATALOG = DataCatalog()
CATALOG.list_clip_boundaries()
