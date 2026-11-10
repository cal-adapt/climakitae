# Query Multiple Variables

Use one `ClimateData` query when several climate fields need the same catalog
selection, time period, and spatial selection. Each field remains a separate
named variable in the returned `xarray.Dataset`, with its own units and
metadata.

## Select variables

Pass one variable as a string, or multiple variables as an ordered list or
tuple. Names are trimmed, and empty, blank, non-string, or duplicate entries
are rejected. There is no fixed variable-count limit; larger requests can still
require more time and memory.

```python
from climakitae.new_core.user_interface import ClimateData

single = (
    ClimateData()
    .catalog("cadcat")
    .activity_id("LOCA2")
    .table_id("day")
    .grid_label("d03")
    .variable_id("tasmax")
    .get()
)

multiple = (
    ClimateData()
    .catalog("cadcat")
    .activity_id("LOCA2")
    .institution_id("UCSD")
    .source_id("ACCESS-CM2")
    .experiment_id("historical")
    .table_id("day")
    .grid_label("d03")
    .variable_id(("tasmax", "pr"))
    .get()
)
```

Strings and one-item lists or tuples retain the existing single-variable
behavior and return structure. A multi-variable request returns one Dataset;
it does not combine differently scaled quantities into a single DataArray.
Use `variable_id`, not the removed `variable()` alias. Each `.get()` resets
the query, so start a fresh chain for another independent request.

## Catalogs and derived fields

Multi-variable selection is supported for CADCAT (WRF and LOCA2), renewable
energy generation, Sup3rCC, and HDP. Shared filters must be valid for every
requested field. For example, WRF and LOCA2 use different variable names.

| Catalog | Example native selection |
|---|---|
| CADCAT WRF | `["t2max", "t2min"]` |
| CADCAT LOCA2 | `["tasmax", "pr"]` |
| Renewable energy generation | `["cf", "gen"]` |
| Sup3rCC | `["t2", "ws10"]` |
| HDP | `["tas", "pr"]` |

Use `show_variable_id_options()` after setting the catalog and relevant
selections to check availability. HDP variable IDs are observation fields
found in station assets, not intake catalog search columns. For example,
station `ASOSAWOS_72019300117` has fields `tas` and `pr`.

Registered derived variables can be mixed with native variables where that
derived calculation is supported. Dependencies are retrieved as needed but
are omitted from the result unless you requested them. Each derived function
must produce its registered output name; otherwise the whole request fails
with a naming diagnostic. HDP does not support derived variables.

```python
wind_and_precip = (
    ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .institution_id("UCLA")
    .source_id("EC-Earth3")
    .experiment_id("historical")
    .table_id("1hr")
    .grid_label("d03")
    .variable_id(["wind_speed_10m", "prec"])
    .processes({"time_slice": ("2000-01-01", "2000-01-02")})
    .get()
)
# wind_speed_10m uses catalog fields u10 and v10; the result contains only
# wind_speed_10m and prec, in the requested order.
```

For HDP, choose a single network and request observation field names:

```python
station_data = (
    ClimateData()
    .catalog("hdp")
    .network_id("ASOSAWOS")
    .station_id("ASOSAWOS_72019300117")
    .variable_id(["tas", "pr"])
    .get()
)
```

## Coverage and alignment

Every requested field must be available under the shared query constraints.
An invalid or unavailable field fails the whole request; `get()` returns
`None` and logs the reason instead of returning a successful partial result.

For gridded catalogs, a simulation/member is retained only when it contains all
requested source fields. For HDP, a station must contain every selected
observation field. Incomplete members or stations are excluded and reported.
If none remain, retrieval fails.

Within each eligible simulation or station, fields are trimmed to their shared
timestamps. The returned data keeps the overlap and diagnostics record the
trimmed counts. If even one otherwise eligible record has no common timestamps,
the entire query fails. Existing NaN measurements are preserved; they are not
treated as proof that a field or timestamp is absent. Spatial coordinates and
calendars must align exactly. The query does not interpolate, regrid, or
silently coerce incompatible fields.

## Process variables independently

### Units

A shared target unit is applied to every selected field with identifiable
units. Use `by_variable` when fields need different targets or when only some
fields should be converted. Targets are validated before conversion, so an
invalid target does not leave a partially converted Dataset.

```python
.processes({
    "convert_units": {
        "by_variable": {
            "tasmax": "degC",
            "pr": "inches/d",
        }
    }
})
```

Variables omitted from `by_variable` remain unchanged. Unit conversions must
be registered for each field's native units. `convert_units` is not supported
for HDP.

### Basic metrics and percentiles

Basic reductions and percentiles apply to each selected field. Shared settings
are convenient when they apply to every field; use `by_variable` for
field-specific metrics. Variables omitted from a targeted configuration pass
through unchanged.

```python
.processes({
    "metric_calc": {
        "percentiles_only": True,
        "by_variable": {
            "tasmax": {"percentiles": [5, 95]},
            "pr": {"percentiles": [50, 99]},
        },
    }
})
```

The result contains `tasmax_percentiles` on the `tasmax_percentile` dimension
and `pr_percentiles` on `pr_percentile`. Each dimension has only that
variable's requested levels; the processor does not pad to a shared percentile
coordinate. Other reduction names keep their existing convention, and
single-variable metric output is unchanged.

Threshold-exceedance and `one_in_x` return-period modes are not supported for
multi-variable queries. `metric_calc` is also unavailable for HDP.

## Point clipping

For a multi-variable query, point clipping chooses one common geographic cell
for all selected fields. It honors an applicable spatial mask, preserves
missing measurements at that cell, and does not search for a cell with
non-missing values or average neighboring cells to fill gaps. Reversing the
requested variable order does not change the selected cell. Existing
single-variable point behavior is unchanged.

```python
point_data = (
    ClimateData()
    .catalog("cadcat")
    .activity_id("LOCA2")
    .table_id("day")
    .grid_label("d03")
    .variable_id(["tasmax", "pr"])
    .processes({"clip": (37.7749, -122.4194)})
    .get()
)
```

## Export

NetCDF and Zarr retain the Dataset's separate variables and metadata.
Multi-variable CSV uses long format with `variable`, `value`, `units`, and
coordinate columns. Each variable contributes rows over its own dimensions;
variables with independent percentile dimensions do not expand into a
Cartesian product. Missing measurements remain rows, and coordinates that do
not apply to a variable are blank. A metadata sidecar is written with the CSV.

```python
data = (
    ClimateData()
    .catalog("cadcat")
    .activity_id("LOCA2")
    .table_id("day")
    .grid_label("d03")
    .variable_id(["tasmax", "pr"])
    .processes({
        "export": {
            "filename": "temperature_and_precipitation",
            "file_format": "CSV",
        }
    })
    .get()
)
```

String and one-item selections retain their existing CSV format, including
when processing creates multiple output fields. S3 export is supported for
Zarr, not CSV.

## Read diagnostics

Successful multi-variable results store a JSON string in the
`multi_variable_diagnostics` Dataset attribute. Gridded results record
simulation identity and per-record status, missing catalog variables, common
timestamp counts, and per-variable trimmed timestamp counts. HDP diagnostics
record station status, missing fields, and timestamp counts. Warnings are also
emitted when records are excluded or timestamps are trimmed.

```python
import json

diagnostics = json.loads(data.attrs["multi_variable_diagnostics"])
```

An unavailable requested field, incompatible coordinates, or zero timestamp
overlap is a query failure, not a successful result with an exclusion
diagnostic. Inspect the log message and the `None` return from `.get()` for
those cases.

## Catalog and processor restrictions

- One query uses one catalog and shared selection constraints. HDP permits one
  network per query.
- HDP does not support `clip`, `convert_units`, `metric_calc`,
  `warming_level`, `localize`, station bias adjustment, or model filtering.
- Sup3rCC does not support `warming_level`, `filter_unadjusted_models`, or
  station bias adjustment.
- Multi-variable station bias adjustment is unsupported.
- Multi-variable `metric_calc` supports basic reductions and percentiles, not
  threshold or return-period modes.
- Other processor restrictions still apply. Check the catalog's available
  options and validator diagnostics before adding a processor to a query.

See the [processor reference](../processors/index.md) for individual
parameters and the [Export Data guide](export.md) for export options.
