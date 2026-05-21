# Legacy Core Module

The legacy `climakitae.core` module provides the original function-based interface for climate data access. It is maintained for backward compatibility; new code should use [`climakitae.new_core.user_interface.ClimateData`](climate-data.md) instead. See [Legacy API status](../legacy/status.md) for the deprecation timeline.

## Submodule reference

The legacy core is split across several modules. Each has its own dedicated, auto-generated reference page:

| Module | What's in it | Reference |
|--------|--------------|-----------|
| `climakitae.core.data_interface` | `DataParameters` (param-based query class) and `get_data` (top-level entrypoint) | [Data Interface (Detailed)](core-data-interface.md) |
| `climakitae.core.boundaries` | `Boundaries` singleton — counties, watersheds, utilities, etc. | [Boundaries (Detailed)](core-boundaries.md) |
| `climakitae.core.data_load` | Internal data-loading helpers used by `get_data` | rendered inline below |
| `climakitae.core.data_export` | Multi-format export (NetCDF, CSV, Zarr, GeoTIFF) | rendered inline below |
| `climakitae.core.constants` | `UNSET`, `WARMING_LEVELS`, `SSPS`, `_NEW_ATTRS_KEY`, model lists | rendered inline below |
| `climakitae.core.paths` | S3 catalog URLs and file path constants | rendered inline below |

## Data Loading

::: climakitae.core.data_load
    options:
      docstring_style: numpy
      show_source: true

## Data Export

::: climakitae.core.data_export
    options:
      docstring_style: numpy
      show_source: true

## Constants

::: climakitae.core.constants
    options:
      docstring_style: numpy
      show_source: true

## Paths

::: climakitae.core.paths
    options:
      docstring_style: numpy
      show_source: true

## Migration Note

For new code, use the modern `climakitae.new_core` interface:

```python
from climakitae.new_core.user_interface import ClimateData

# Note: WRF uses 't2max'; LOCA2 uses 'tasmax'.
data = (ClimateData()
    .catalog("cadcat")
    .activity_id("LOCA2")
    .grid_label("d03")
    .table_id("day")
    .variable("tasmax")
    .get())
```

See the [Legacy → ClimateData migration guide](../migration/legacy-to-climate-data.md).
