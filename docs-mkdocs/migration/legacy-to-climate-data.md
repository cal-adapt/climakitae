# Migration: Legacy to ClimateData

This guide helps you migrate from the legacy `climakitae.core` API to the modern `ClimateData` fluent interface.

## Migration Policy

- **New work**: Always use `new_core` (fluent API)
- **Existing code**: Legacy APIs remain available and functional during transition
- **Timeline**: A multi-phase deprecation plan is published in [Legacy API status](../legacy/status.md). No breaking changes in the current minor-release series.
- **Support**: Both interfaces are maintained, but `new_core` is where all new features land.

!!! note "A note on field names"
    The legacy `DataParameters` class is built on the `param` library and uses
    *human-readable* field names that mirror its GUI controls
    (`downscaling_method`, `resolution`, `timescale`, `area_subset`,
    `cached_area`, `scenario_ssp`, `time_slice` as a year-range tuple, etc.).
    The `new_core` interface uses *catalog-native* names (`activity_id`,
    `grid_label`, `table_id`, `experiment_id`, etc.). Watch for this when
    porting a script.

## Quick Pattern: Basic Query

### ❌ Legacy Approach

```python
from climakitae.core.data_interface import DataParameters, get_data

# Build parameters using the legacy field names
params = DataParameters()
params.downscaling_method = "Dynamical"           # WRF
params.resolution = "3 km"                         # ≈ grid_label d03
params.timescale = "monthly"                       # ≈ table_id 'mon'
params.variable = "Maximum air temperature at 2m"  # display name, not 't2max'
params.scenario_historical = ["Historical Climate"]
params.scenario_ssp = []                           # historical-only example
params.time_slice = (2015, 2015)                   # year range, NOT date strings
params.area_subset = "CA counties"
params.cached_area = ["Los Angeles County"]

data = get_data(params)
```

### ✅ ClimateData Approach

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

**Benefits of ClimateData:**
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

### ✅ ClimateData Approach

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
params.variable = "Maximum air temperature at 2m"

# Named area (boundary layer + cached entry)
params.area_subset = "CA counties"
params.cached_area = ["Los Angeles County"]

# OR for a single point, set lat/lon ranges to a degenerate range
params.latitude = (34.05, 34.05)
params.longitude = (-118.25, -118.25)
```

### ✅ ClimateData Approach

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

### ✅ ClimateData Approach

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

scenarios = ["SSP 2-4.5", "SSP 3-7.0", "SSP 5-8.5"]
results = {}

for scenario in scenarios:
    params = DataParameters()
    params.scenario_ssp = [scenario]
    params.variable = "Maximum air temperature at 2m"
    # ... set downscaling_method, resolution, timescale, area_subset, etc. ...
    results[scenario] = get_data(params)
```

### ✅ ClimateData Approach

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
# Legacy supports warming-level workflows via the GWL approach toggle.
params = DataParameters()
params.approach = "Warming Level"
params.warming_level = [1.5, 2.0, 3.0]
params.warming_level_window = 15  # ±15-year window around GWL crossing
params.variable = "Minimum air temperature at 2m"
data = get_data(params)
```

The Cal-Adapt [Methods page](https://analytics.cal-adapt.org/analytics/methods)
describes the warming-level fetching algorithm in more detail (and is itself
still written against the legacy API).

### ✅ ClimateData Approach

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

| Issue | Legacy | ClimateData |
|-------|--------|----------|
| **Query reset after execution** | Set params once, call `get_data()` multiple times | Create new `ClimateData()` instance per query |
| **Discovering options** | Manual, docstring-based | `show_*_options()` methods (programmatic) |
| **Chaining operations** | Limited; multiple `get_data()` calls | Native `.processes()` dict with all operations |
| **Lazy evaluation** | Partial (xarray but not all ops) | Full lazy evaluation with dask |
| **Export** | Separate `data_export` module | Integrated `"export"` processor |
| **Type hints** | Minimal | Full type hints (Python 3.10+) |

---

## Migration Checklist  

- Replace `from climakitae.core.data_interface import DataParameters, get_data` with `from climakitae.new_core.user_interface import ClimateData`.  
- Translate field names:  
    - `downscaling_method` ("Dynamical" / "Statistical") → `.activity_id("WRF" | "LOCA2")`  
    - `resolution` ("3 km" / "9 km" / "45 km") → `.grid_label("d03" | "d02" | "d01")`  
    - `timescale` ("hourly" / "daily" / "monthly") → `.table_id("1hr" | "day" | "mon")`  
    - `scenario_ssp` / `scenario_historical` → `.experiment_id(...)` (a list of `"ssp245"` / `"ssp370"` / `"ssp585"` / `"historical"`)  
    - Display variable name (e.g. "Maximum air temperature at 2m") → `.variable("<variable_id>")` such as `"t2max"` for WRF or `"tasmax"` for LOCA2.  
- Convert year ranges: `params.time_slice = (2015, 2050)` → `.processes({"time_slice": (2015, 2050)})` (or ISO date strings).  
- Convert clipping: `params.area_subset` + `params.cached_area` → `.processes({"clip": "<boundary name>"})` or a `(lat, lon)` tuple.  
- Replace `get_data(params)` with `.get()`.  
- Update option discovery to use `show_*_options()` methods.  
- Move exports to the `"export"` processor or call `.to_netcdf()` / `.to_zarr()` on the result.  
- Run side-by-side comparison on a small slice to verify equivalent results before swapping production code.  

---

## Still Using Legacy? Questions?

- Legacy interface remains fully supported in `climakitae.core`  
- For new_core API details, see the [API Reference section](../api/climate-data.md)  
- See [Legacy API status](../legacy/status.md) for deprecation timeline  
