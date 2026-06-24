# Legacy Core Concepts

The mental model behind the legacy `climakitae.core` interface — how queries are
configured, why the field names differ from the modern interface, and how a
query flows from configuration to data.

!!! note "Audience"
    This page is for readers maintaining or reading **existing** legacy code. If
    you are writing new code, start with the
    [ClimateData interface](../climate-data-interface/index.md) instead.

!!! warning
    `climakitae.core` is maintained for backward compatibility only. New work
    should use [`climakitae.new_core.user_interface.ClimateData`](../climate-data-interface/index.md).

## On this page

- [The configuration object](#the-configuration-object)
- [GUI-style field names](#gui-style-field-names)
- [Legacy → modern mapping](#legacy-to-modern-mapping)
- [Query flow](#query-flow)
- [Named boundaries](#named-boundaries)

---

## The configuration object

Internally, the legacy interface is built around a **configuration object**,
`DataParameters`. It was originally designed to back a GUI, so it carries
observers that keep dependent fields (resolution, timescale, scenario, cached
area) in sync as values change.

You rarely need to build one by hand. The recommended entry point is
`get_data()`, which accepts GUI-style **keyword arguments**, constructs and
validates a `DataParameters` object for you, and returns the data:

```python
from climakitae.core.data_interface import get_data

data = get_data(
    variable="Air Temperature at 2m",
    resolution="9 km",
    timescale="hourly",
)
```

Driving a `DataParameters` instance directly and calling `.retrieve()` is still
supported for GUI-style workflows, but `get_data()` is the preferred path for
new and maintained code.

---

## GUI-style field names

The most important thing to understand about the legacy interface is that its
field names are **human-readable GUI labels**, not the catalog-native names used
by `ClimateData`. For example, the legacy interface uses `"9 km"` where the
modern interface uses the grid label `"d02"`, and `"Statistical"` where the
modern interface uses the activity id `"LOCA2"`.

This means legacy code reads more like the old web tool and less like the
underlying intake-esm catalog.

---

## Legacy → modern mapping { #legacy-to-modern-mapping }

When porting legacy code, this table maps the common GUI-style fields to their
modern `ClimateData` equivalents:

| Legacy field | Meaning | Modern equivalent |
|--------------|---------|-------------------|
| `downscaling_method` | Dynamical, Statistical, or both | `activity_id` (`"WRF"` / `"LOCA2"`) |
| `resolution` | 3 km, 9 km, or 45 km | `grid_label` (`"d03"` / `"d02"` / `"d01"`) |
| `timescale` | hourly, daily, monthly | `table_id` (`"1hr"` / `"day"` / `"mon"`) |
| `scenario_ssp` / `scenario_historical` | Scenario selection buckets | `experiment_id` |
| `area_subset` / `cached_area` | Named boundary selection | `clip` processor |
| `time_slice` | Year-range tuple | `time_slice` processor |
| `variable` | GUI display name | `variable` (catalog id, e.g. `t2max` / `tasmax`) |

See the [migration guide](../migration/legacy-to-climate-data.md) for a complete
walkthrough.

---

## Query flow

A legacy query moves through four stages:

1. `get_data()` builds a `DataParameters` object from your keyword arguments
   and loads the singleton `DataInterface` with the available options.
2. Option observers validate and keep fields like `resolution`, `timescale`,
   `scenario_ssp`, and `cached_area` in sync.
3. `get_data()` calls the catalog loader (the same loader used by
   `DataParameters.retrieve()`).
4. The loader returns an `xarray.DataArray`, `xarray.Dataset`, or a list of
   `DataArray` objects depending on the request.

Always prefer calling `get_data()` with keyword arguments — it handles the
`DataParameters` construction and validation for you.

Like the modern interface, the result is **lazily loaded** — data streams from
S3 only when you compute, plot, or export it.

---

## Named boundaries

Spatial subsetting in the legacy interface is driven by the
[`Boundaries`](boundaries.md) loader. The GUI exposes a small set of boundary
categories (`area_subset`) and, within each, a list of named regions
(`cached_area`):

```python
from climakitae.core.data_interface import DataInterface

boundaries = DataInterface().geographies
county_lookup = boundaries.boundary_dict()["CA counties"]
```

In the modern interface this is replaced by the much more flexible
[`clip` processor](../climate-data-interface/processors/clip.md).

---

## Where to read next

- [Data Interface reference](data-interface.md)
- [Boundaries reference](boundaries.md)
- [Legacy → ClimateData migration guide](../migration/legacy-to-climate-data.md)
- [ClimateData interface overview](../climate-data-interface/index.md)
