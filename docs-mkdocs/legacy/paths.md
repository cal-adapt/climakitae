# Legacy Paths

The **`climakitae.core.paths` module** holds the S3 catalog URLs and file path
constants used by the legacy loaders to locate climate data and supporting
catalogs.

!!! warning
    This page documents a legacy support module. It is kept for backward
    compatibility. New code should use
    [`climakitae.new_core.user_interface.ClimateData`](../climate-data-interface/index.md).

## What this module does

- Defines the S3 endpoints for the main climate-data catalog and the boundary
  catalog.
- Provides the file path constants for the bundled CSV data (variable
  descriptions, warming-level lookup tables, station metadata).

These constants are consumed by the legacy [Data Loading](data-load.md) helpers
and the [Boundaries](boundaries.md) loader.

---

## Public API

::: climakitae.core.paths
    options:
      docstring_style: numpy
      show_source: true

---

## Related legacy modules

- [Legacy API Overview](index.md)
- [Data Loading](data-load.md)
- [Constants](constants.md)
