# Migration: Legacy to New Core

This guide helps you migrate from the legacy `climakitae.core` API to the modern `new_core` fluent interface.

## Migration Policy

- **New work**: Always use `new_core` (fluent API)
- **Existing code**: Legacy APIs remain available and functional during transition
- **Timeline**: No hard deadline; migrate at your own pace
- **Support**: Both interfaces are maintained and fully featured

## Quick Pattern: Basic Query

### ❌ Legacy Approach

```python
from climakitae.core.data_interface import DataParameters, get_data
import pandas as pd

# Build parameter dictionary
params = DataParameters()
params.catalog = "cadcat"
params.activity_id = "WRF"
params.table_id = "mon"
params.grid_label = "d03"
params.variable = "t2max"
params.start_date = "2015-01-01"
params.end_date = "2015-12-31"
params.area_average = ["Los Angeles"]

# Get data
data = get_data(params)
```

### ✅ New Core Approach

```python
from climakitae.new_core.user_interface import ClimateData

# Fluent/builder pattern
data = (ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .table_id("mon")
    .grid_label("d03")
    .variable("t2max")
    .processes({
        "time_slice": ("2015-01-01", "2015-12-31"),
        "clip": "Los Angeles"
    })
    .get())
```

**Benefits of new core:**
- Cleaner syntax with method chaining
- Query state visible in code flow (no hidden dict state)
- Processors replace ad-hoc parameters (time_slice, clip, export)
- Better IDE autocomplete and type hints

---

## Pattern: Discovering Options

### ❌ Legacy Approach

```python
from climakitae.core.data_interface import DataParameters

params = DataParameters()

# Manual inspection or reading docs
available_catalogs = ["cadcat", "renewables"]  # Hardcoded
available_variables = ["tasmax", "tasmin", "pr", ...]  # Check docs

# Trial-and-error or print statements
print(params.grid_label)  # Check what's set
```

### ✅ New Core Approach

```python
from climakitae.new_core.user_interface import ClimateData

cd = ClimateData()

# Programmatic discovery
cd.show_catalog_options()           # All available catalogs
cd.show_variable_options()          # All available variables for current catalog
cd.show_grid_label_options()        # All available resolutions
cd.show_boundary_options()          # All clipping regions

# Filtered discovery (options update as you chain)
cd.catalog("cadcat").show_activity_id_options()  # Only activity IDs for cadcat
```

---

## Pattern: Clipping / Spatial Subsetting

### ❌ Legacy Approach

```python
params = DataParameters()
params.variable = "t2max"

# Named area
params.area_average = ["Los Angeles"]

# OR for points, had limited options
params.latitude = 34.05
params.longitude = -118.25
```

### ✅ New Core Approach

```python
# Named region
data = (cd.variable("t2max")
    .processes({"clip": "Los Angeles"})
    .get())

# Single point (lat, lon)
data = (cd.variable("t2max")
    .processes({"clip": (34.05, -118.25)})
    .get())

# Multiple points
data = (cd.variable("t2max")
    .processes({"clip": [(34.05, -118.25), (37.77, -122.42)]})
    .get())

# Bounding box ((lat_min, lat_max), (lon_min, lon_max))
data = (cd.variable("t2max")
    .processes({"clip": ((34.0, 36.0), (-121.0, -119.0))})
    .get())
```

---

## Pattern: Exporting Data

### ❌ Legacy Approach

```python
from climakitae.core import data_export

data = get_data(params)

# Export to specific format
data_export.export(data, "output.nc", format="netcdf")
data_export.export(data, "output.zarr", format="zarr")
```

### ✅ New Core Approach

```python
# Export via processor (integrated into query)
data = (cd
    .variable("t2max")
    .processes({
        "time_slice": ("2015-01-01", "2015-12-31"),
        "export": {
            "filename": "la_temps_2015",
            "file_format": "NetCDF"
        }
    })
    .get())

# Or export after retrieval
data.to_netcdf("output.nc")
data.to_zarr("output.zarr")
```

---

## Pattern: Multiple Scenarios

### ❌ Legacy Approach

```python
from climakitae.core.data_interface import DataParameters, get_data

scenarios = ["ssp245", "ssp370", "ssp585"]
results = {}

for scenario in scenarios:
    params = DataParameters()
    params.scenario = scenario
    params.variable = "tasmax"
    # ... set many other params ...
    results[scenario] = get_data(params)
```

### ✅ New Core Approach

```python
from climakitae.new_core.user_interface import ClimateData

scenarios = ["ssp245", "ssp370", "ssp585"]
results = {}

for scenario in scenarios:
    results[scenario] = (ClimateData()
        .experiment_id(scenario)
        .variable("tasmax")
        # ... chain other params ...
        .get())
```

---

## Pattern: Warming Level Analysis

### ❌ Legacy Approach

```python
# Not directly supported in legacy interface
# Workaround: query multiple years manually
params = DataParameters()
params.start_year = 2050  # Approximate when warming reaches target
params.end_year = 2060
```

### ✅ New Core Approach

```python
# Direct warming level support
data = (ClimateData()
    .variable("t2min")
    .processes({
        "warming_level": {
            "warming_levels": [1.5, 2.0, 3.0]
        }
    })
    .get())
```

---

## Common Gotchas

| Issue | Legacy | New Core |
|-------|--------|----------|
| **Query reset after execution** | Set params once, call `get_data()` multiple times | Create new `ClimateData()` instance per query |
| **Discovering options** | Manual, docstring-based | `show_*_options()` methods (programmatic) |
| **Chaining operations** | Limited; multiple `get_data()` calls | Native `.processes()` dict with all operations |
| **Lazy evaluation** | Partial (xarray but not all ops) | Full lazy evaluation with dask |
| **Export** | Separate `data_export` module | Integrated `"export"` processor |
| **Type hints** | Minimal | Full type hints (Python 3.10+) |

---

## Migration Checklist

- [ ] Replace `from climakitae.core.data_interface import DataParameters, get_data` with `from climakitae.new_core.user_interface import ClimateData`
- [ ] Convert `params.variable = "x"` to `.variable("x")`
- [ ] Convert date ranges: `params.start_date` → `.processes({"time_slice": (...)})`
- [ ] Convert clipping: `params.area_average` → `.processes({"clip": ...})`
- [ ] Replace `get_data(params)` with `.get()`
- [ ] Update option discovery to use `show_*_options()`
- [ ] Move exports to processor-based pattern or `.to_netcdf()` / `.to_zarr()`
- [ ] Test with real climate data to verify results match

---

## Still Using Legacy? Questions?

- Check [API reference](api/index.md) for complete new_core docs
- Legacy interface remains fully supported in `climakitae.core`
- See [Legacy API status](../legacy/status.md) for deprecation timeline
