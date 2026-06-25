# Legacy Data Interface

The **`climakitae.core.data_interface` module** is the main compatibility layer for the original climakitae API. It exposes the legacy parameter object, the data retrieval entry points, and the discovery helpers that powered the old GUI.

!!! warning
    This page documents the legacy `climakitae.core.data_interface` module. It is kept for backward compatibility. New code should use [`climakitae.new_core.user_interface.ClimateData`](../climate-data-interface/index.md).

## On this page

- [What this module does](#what-this-module-does)
- [Core concepts](#core-concepts)
- [Query flow](#query-flow)
- [Legacy field names](#legacy-field-names)
- [Examples](#examples)
- [Public API](#public-api)
- [Notes on behavior](#notes-on-behavior)

---

## What this module does

The legacy data interface is responsible for:

- mapping human-readable GUI values to catalog values
- validating combinations of resolution, timescale, scenario, and spatial subset
- exposing data and subsetting option lookup helpers
- loading the cached data catalogs, variable metadata, station metadata, and boundary catalogs used by `DataParameters`

---

## Core concepts

| Concept | Legacy symbol | Role |
|---|---|---|
| Variable metadata | `VariableDescriptions` | Loads `variable_descriptions.csv` once and keeps it available for option lookups |
| Shared data connections | `DataInterface` | Singleton cache for catalogs, stations, boundaries, and warming-level tables |
| Query state | `DataParameters` | Param-based configuration object used by the GUI and direct code paths |
| Data discovery | `get_data_options()` | Returns the valid query combinations in legacy GUI language |
| Spatial discovery | `get_subsetting_options()` | Returns valid boundaries and station geometry options |
| Data retrieval | `get_data()` / `DataParameters.retrieve()` | Executes a legacy query and returns lazily loaded xarray data |

---

## Query flow

1. `DataParameters` loads the singleton `DataInterface` and populates the available options.
2. Option observers keep fields like `resolution`, `timescale`, `scenario_ssp`, and `cached_area` in sync.
3. `retrieve()` or `get_data()` calls the catalog loader.
4. The loader returns an `xarray.DataArray`, `xarray.Dataset`, or a list of `DataArray` objects depending on the request.

---

## Legacy field names

The legacy module uses GUI-style names instead of catalog-native names. Common examples:

| Legacy field | Meaning | Modern equivalent |
|---|---|---|
| `downscaling_method` | Dynamical, Statistical, or both | `activity_id` |
| `resolution` | 3 km, 9 km, or 45 km | `grid_label` |
| `timescale` | hourly, daily, monthly | `table_id` |
| `scenario_ssp` / `scenario_historical` | Scenario selection buckets | `experiment_id` |
| `area_subset` / `cached_area` | Named boundary selection | `clip` processor |
| `time_slice` | Year range tuple | `time_slice` processor |

See [Core Concepts](concepts.md#legacy-to-modern-mapping) for the full mapping.

---

## Examples

### Direct query with `DataParameters`

```python
from climakitae.core.data_interface import DataParameters

params = DataParameters()
params.variable = "Air Temperature at 2m"
params.downscaling_method = "Dynamical"
params.resolution = "9 km"
params.timescale = "hourly"
params.scenario_historical = ["Historical Climate"]
params.scenario_ssp = ["SSP 3-7.0"]
params.area_subset = "CA counties"
params.cached_area = ["Los Angeles County"]

data = params.retrieve()
```

### Direct query with `get_data`

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

---

## Public API

### VariableDescriptions

::: climakitae.core.data_interface.VariableDescriptions
    options:
      docstring_style: numpy
      show_source: true

### DataInterface

::: climakitae.core.data_interface.DataInterface
    options:
      docstring_style: numpy
      show_source: true
      merge_init_into_class: true

### DataParameters

::: climakitae.core.data_interface.DataParameters
    options:
      docstring_style: numpy
      show_source: true
      merge_init_into_class: true
      members_order: source

### get_data_options

::: climakitae.core.data_interface.get_data_options
    options:
      docstring_style: numpy
      show_source: true

### get_subsetting_options

::: climakitae.core.data_interface.get_subsetting_options
    options:
      docstring_style: numpy
      show_source: true

### get_data

::: climakitae.core.data_interface.get_data
    options:
      docstring_style: numpy
      show_source: true

---

## Notes on behavior

- `DataParameters.retrieve()` is the closest analogue to the old GUI workflow.
- `get_data_options()` and `get_subsetting_options()` are useful when you need to discover valid values programmatically.
- The module does not raise on every bad input. In several cases it prints a diagnostic message and returns `None` to match the original notebook behavior.
- `get_data()` is the lower-level direct entry point and accepts the same legacy naming conventions as the GUI.

---

## Related legacy modules

- [Legacy API Overview](index.md)
- [Core Concepts](concepts.md)
- [Legacy Boundaries](boundaries.md)
- [Legacy → ClimateData migration guide](../migration/legacy-to-climate-data.md)