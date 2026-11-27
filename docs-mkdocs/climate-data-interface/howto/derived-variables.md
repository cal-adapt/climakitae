# Derived Variables & Climate Indices

There are two ways to get a derived quantity:

- **Registered derived variables** — request them with `variable_id`, like a
  catalog variable. `ClimateData` retrieves the dependencies, clips them, and
  calculates the variable in the processing pipeline.
- **`climakitae.tools` functions** — call them yourself on data that you
  already retrieved.

## Query a registered derived variable

```python
from climakitae.new_core.user_interface import ClimateData

dtr = (ClimateData()
    .catalog("cadcat")
    .activity_id("LOCA2")
    .institution_id("UCSD")
    .source_id("ACCESS-CM2")
    .experiment_id("historical")
    .table_id("day")
    .grid_label("d03")
    .variable_id("diurnal_temperature_range_loca")
    .processes({"clip": "Los Angeles"})
    .get())
```

Retrieval opens only the dependencies (`tasmax` and `tasmin`). `clip` subsets
them first, then the internal
[`derive_variables`](../processors/derive_variables.md) processor calculates
`diurnal_temperature_range_loca` on the clipped data. The dependencies are not
in the result. Derived variables can also be mixed with catalog variables in a
[multi-variable query](multi-variable-queries.md).

List the registered derived variables and their dependencies:

```python
ClimateData().show_derived_variables()
```

## Register your own derived variable

```python
from climakitae.new_core.derived_variables import register_user_function

def calc_temp_range(ds):
    ds["temp_range"] = ds.tasmax - ds.tasmin
    ds["temp_range"].attrs = {"units": "K", "long_name": "Daily Temperature Range"}
    return ds

register_user_function(
    name="temp_range",
    depends_on=["tasmax", "tasmin"],
    func=calc_temp_range,
    units="K",
)

data = (ClimateData()
    .catalog("cadcat")
    .activity_id("LOCA2")
    .table_id("day")
    .grid_label("d03")
    .variable_id("temp_range")
    .processes({"clip": "Los Angeles"})
    .get())
```

`ClimateData.derived_variable(name, depends_on, func, ...)` registers the
function and sets `variable_id` in one call.

The function must follow these rules:

- It receives an `xr.Dataset` that contains the `depends_on` fields, and
  returns a Dataset that contains a variable with the registered name. If the
  name is missing, the query fails.
- Only new variables are kept. Changes to the input fields are ignored.
- By default the dependencies are removed from the result. Pass
  `drop_dependencies=False` to `register_user_function` to keep them; they
  follow the requested variables.

### What the function receives

The function runs at priority 3, after `clip` and before the other
processors. Thus:

- **Clipped grid only.** A function that uses nearby cells, for example a
  spatial gradient, sees only the clipped extent.
- **Native catalog units.** `convert_units` runs later. Convert units inside
  the function if it needs specific units.
- **Full time series.** `warming_level` and `time_slice` run later, so time
  shifts such as `shift(time=...)` have the complete record.
- **One simulation at a time.** `concat` runs later.
- **Point clips use one common cell.** If the dependencies include two or more
  fields, a point clip uses the nearest grid cell for all of them, with no
  search for valid neighbor cells. If that cell is NaN, the derived output is
  NaN. See [Clip](../processors/clip.md#multi-variable-point-selection).

## Use `climakitae.tools` functions

These functions operate on data that you already retrieved.

```python
from climakitae.new_core.user_interface import ClimateData
from climakitae.tools.derived_variables import compute_hdd_cdd
from climakitae.tools.indices import effective_temp

data = (ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .institution_id("UCLA")
    .table_id("day")
    .grid_label("d03")
    .variable_id("t2")
    .processes({
        "time_slice": ("2030-01-01", "2060-12-31"),
        "clip": "Los Angeles",
        "convert_units": "degF",  # compute_hdd_cdd thresholds are in degF
    })
    .get())

# compute_hdd_cdd requires a DataArray named "t2"
hdd, cdd = compute_hdd_cdd(data["t2"], hdd_threshold=65, cdd_threshold=65)

# Exponentially smoothed temperature for energy demand
eff_temp = effective_temp(data["t2"])
```

Available functions include:

- `climakitae.tools.derived_variables.compute_hdd_cdd()`: heating and cooling degree days (thresholds in degF)
- `climakitae.tools.indices.effective_temp()`: exponentially smoothed temperature for demand forecasting
- `climakitae.tools.indices.noaa_heat_index()`: heat index from temperature (degF) and relative humidity (%)

For the full list, see [Tools](../../api/tools.md). For the registry of
derived variables, see [Derived Variables Module](../../api/derived-variables.md).

---
