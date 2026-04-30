# Climakitae Interface Overview

The **`ClimateData` interface** is the primary, actively developed interface for `climakitae`. It exposes a fluent / builder API rooted in the `ClimateData` class and a registry-based processor pipeline.

```python
from climakitae.new_core.user_interface import ClimateData

data = (ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .table_id("mon")
    .grid_label("d03")
    .variable("t2max")
    .processes({
        "time_slice": ("2015-01-01", "2015-12-31"),
        "clip": "Los Angeles County",
    })
    .get())
```

## Design goals

The `ClimateData` interface is designed with with the following goals in mind:  

- **Fluent method chaining** for readable, top-to-bottom query construction and a familiarity to other packages like `xarray` and `pandas`.
- **Explicit, ordered processors** for ease of use across transformations (clipping, time slicing, warming-level subsetting, unit conversion, export, …). See the [Processors index](processors/index.md).
- **Lazy execution** backed by `xarray` + `dask`. Queries return xarray objects with lazy arrays; data only streams from S3 when calling `.compute()` / `.values` / `.plot()`.
- **Registry-based extensibility.** New catalogs, validators, and processors register themselves at import time — no central wiring required.

## How to read the documentation sections

| Page | Read it when… |
|------|---------------|
| [Concepts](concepts.md) | You need the mental model — catalog hierarchy, lazy evaluation, available boundaries. |
| [How-To Guides](howto.md) | You have a concrete task (clip to a county, request hourly data, export to Zarr, …). |
| [Architecture](architecture.md) | You want to extend climakitae with a new processor / validator / catalog, or you're debugging the pipeline. |
| [Processors](processors/index.md) | You want the registry, priorities, and per-processor parameter shapes. |
| [User Interface API](../api/climate-data.md) | You want the auto-generated `ClimateData` reference. |

## Status

The ClimateData interface has reached feature parity with the legacy `climakitae.core` interface for the catalogs that ship with `climakitae 1.4.x` (`cadcat`, `renewable energy generation`, `hdp`). All new features land here. See [Legacy API status](../legacy/status.md) for the migration timeline.

## See also

- [Cal-Adapt Analytics Engine — Methods](https://analytics.cal-adapt.org/analytics/methods)
- [Cal-Adapt Analytics Engine — Glossary](https://analytics.cal-adapt.org/guidance/glossary)
- [Notebook Gallery](../notebook-gallery.md)
