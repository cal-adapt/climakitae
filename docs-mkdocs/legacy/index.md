# Legacy API

The **legacy `climakitae.core` interface** is the original function-based API for climakitae. It is maintained for backward compatibility only — new work should use [`climakitae.new_core.user_interface.ClimateData`](../climate-data-interface/index.md).

!!! warning
    `climakitae.core` is maintained for backward compatibility only. It still powers older notebooks and internal workflows, but it is no longer the home for new features. New processors, validators, catalogs, and analysis features land in `climakitae.new_core`.

---

## What belongs here

The legacy API is centered on a small set of modules that work together. Each has a dedicated reference page:

| Module | Purpose | Reference |
|--------|---------|-----------|
| `climakitae.core.data_interface` | The main legacy entry point. Defines `DataParameters` and `get_data()`. | [Data Interface](data-interface.md) |
| `climakitae.core.boundaries` | Legacy boundary loader used for named geographic clipping. | [Boundaries](boundaries.md) |
| `climakitae.core.data_load` | Internal helpers that assemble and load legacy datasets. | [Data Loading](data-load.md) |
| `climakitae.core.data_export` | Legacy export helpers for NetCDF, CSV, Zarr, and GeoTIFF. | [Data Export](data-export.md) |
| `climakitae.core.constants` | Shared sentinels and lookup constants used across the legacy stack. | [Constants](constants.md) |
| `climakitae.core.paths` | File and catalog path constants used by the legacy loaders. | [Paths](paths.md) |

---

## How to read this section

<div class="grid cards" markdown>

-   **[Core Concepts](concepts.md)**

    ---

    The mental model — the `get_data()` entry point, GUI-style field
    names, the legacy → modern mapping, and the query flow.

-   **Reference**

    ---

    Per-module API references with auto-generated docstrings:
    [Data Interface](data-interface.md) ·
    [Boundaries](boundaries.md) ·
    [Data Loading](data-load.md) ·
    [Data Export](data-export.md) ·
    [Constants](constants.md) ·
    [Paths](paths.md)

-   **[Migration Guide](../migration/legacy-to-climate-data.md)**

    ---

    Step-by-step instructions for porting legacy code to the modern
    `ClimateData` interface.

</div>

---

## How the legacy workflow fits together

The primary entry point is `get_data()`. It accepts keyword arguments, builds
its own `DataParameters` object internally, and returns an xarray object:

1. Call `get_data()` with GUI-style keyword arguments.
2. `get_data()` constructs and validates a `DataParameters` object for you.
3. It executes the query and returns an `xarray.DataArray` (or `None` on bad input).
4. Use `load()` or `export()` when you need to materialize or persist the result.

```python
from climakitae.core.data_interface import get_data

data = get_data(
    variable="Air Temperature at 2m",
    resolution="9 km",
    timescale="hourly",
    downscaling_method="Dynamical",
    scenario=["Historical Climate", "SSP 3-7.0"],
    area_subset="CA counties",
    cached_area=["Los Angeles County"],
)
```

For GUI-style workflows you can still build a `DataParameters` object directly
and call `.retrieve()`, but `get_data()` with keyword arguments is the
recommended path. See [Core Concepts](concepts.md) for the field-name
conventions and the legacy → modern mapping.

---

## Status and roadmap

Legacy support remains available for backward compatibility, but the project
direction is clear:

- Existing code continues to work.
- New documentation and tutorials use `ClimateData`.
- New feature work lands in `climakitae.new_core` only.

If you are starting new code, use the modern interface and treat this section as
a compatibility reference.

---

## Where to read next

- [Core Concepts](concepts.md)
- [Data Interface reference](data-interface.md)
- [Boundaries reference](boundaries.md)
- [Legacy → ClimateData migration guide](../migration/legacy-to-climate-data.md)
- [ClimateData interface overview](../climate-data-interface/index.md)
