# Legacy Boundaries

The **`climakitae.core.boundaries` module** provides the geographic lookup layer used by the legacy API to subset by named regions, stations, and bounding boxes.

!!! warning
    This page documents the legacy `climakitae.core.boundaries` module. It is kept for backward compatibility. New code should use [`climakitae.new_core.data_access.boundaries`](../climate-data-interface/processors/clip.md).

## On this page

- [What this module does](#what-this-module-does)
- [Available boundary groups](#available-boundary-groups)
- [Usage example](#usage-example)
- [Public API](#public-api)
- [Notes on behavior](#notes-on-behavior)

---

## What this module does

- loads the legacy boundary parquet catalog
- caches the available geometries in memory
- exposes lookup dictionaries used by `DataParameters.cached_area`
- normalizes a small set of boundary categories used by the GUI

---

## Available boundary groups

| Boundary key | Meaning |
|---|---|
| `states` | Western US states |
| `CA counties` | California counties |
| `CA watersheds` | California HUC8 watersheds |
| `CA Electric Load Serving Entities (IOU & POU)` | Investor-owned and publicly-owned utilities |
| `CA Electricity Demand Forecast Zones` | Forecast zones used for demand planning |
| `CA Electric Balancing Authority Areas` | Balancing authority areas |
| `lat/lon` | Coordinate-based selection |
| `none` | Entire-domain selection |

---

## Usage example

```python
from climakitae.core.data_interface import DataInterface

boundaries = DataInterface().geographies
county_lookup = boundaries.boundary_dict()["CA counties"]
```

---

## Public API

::: climakitae.core.boundaries.Boundaries
    options:
      docstring_style: numpy
      show_source: true
      merge_init_into_class: true

---

## Notes on behavior

- The county and watershed tables are sorted for stable option ordering.
- The balancing authority table drops the tiny CALISO polygon and keeps the larger geometry.
- The forecast-zone catalog renames entries marked `Other` to the county definition they belong to.

---

## Related legacy modules

- [Legacy API Overview](index.md)
- [Core Concepts](concepts.md)
- [Legacy Data Interface](data-interface.md)
- [Legacy → ClimateData migration guide](../migration/legacy-to-climate-data.md)