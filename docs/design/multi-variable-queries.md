# Multi-variable queries

## Problem

`ClimateData.variable_id` used to accept one variable. Users who needed, say,
`tasmax` and `pr` under the same catalog selection ran two queries and joined
the results themselves, with no guarantee that both covered the same members,
timestamps, or grid. A multi-variable query returns one lazy `xarray.Dataset`
with each requested variable as a separate named field, keeping its own units
and attributes. There is no fixed variable-count cap.

## Contract

| Concern | Decision |
| --- | --- |
| Catalogs | CADCAT (WRF and LOCA2), renewable energy generation, Sup3rCC, and HDP. One catalog and one set of shared constraints per query; HDP keeps one network per query. |
| Input | `variable_id` takes a string or a list/tuple of strings. Names are trimmed and kept in request order; empty, blank, non-string, and duplicate entries raise `ValueError`. |
| Single-variable compatibility | A string or one-element collection keeps the existing single-variable path and return shape. |
| Atomic availability | If any requested variable (or derived dependency) is unavailable under the shared constraints, the whole query fails. No partial results. |
| Output | Exactly the requested variables, in request order. Dependency-only fields are dropped. Mixed-unit fields are never stacked into one DataArray. |
| Derived variables | Native/derived mixtures and several derived variables are supported wherever derived variables already work. Each derived function must produce its registered name, or the query fails. HDP has no derived variables. |
| Failures | `get()` returns `None` and logs the reason. Setter misuse still raises `ValueError`. |
| State | Query metadata (`_requested_variables`, `_catalog_variable_ids`, ...) travels in the query dict. Nothing is stored on the `DataCatalog` singleton. |

Validation (`ParameterValidator._build_multi_variable_plan`) expands derived
variables into catalog dependencies and checks that every dependency exists
under the other constraints before anything is opened.

## Retrieval

### Gridded catalogs: intake-esm aggregation

CADCAT, renewables, and Sup3rCC go through
`DataCatalog._open_complete_multi_variable_datasets`. The catalog's
`aggregation_control` is the source of truth: all three use `union` aggregation
on `variable_id` and leave it out of `groupby_attrs`, so a multi-variable
search already yields one intake key per simulation group. CADCAT adds a
`join_new` on `member_id`.

1. `require_all_on` is the catalog's `groupby_attrs` plus `member_id` (when
   the catalog has that column). intake-esm keeps only the simulation/member
   rows that have every requested catalog variable.
2. Excluded members are logged at info level along with their missing
   variables. If no rows survive, the query fails.
3. `to_dataset_dict` combines assets with
   `xarray_combine_by_coords_kwargs={"join": "inner"}` and
   `zarr_kwargs={"consolidated": None}` (works on Zarr v2 and v3).
4. Derived variables are computed with their inputs kept until all outputs
   exist. The result is then subset to the requested variables.

Why an inner join: intake-esm's default outer combine pads a missing variable,
member, or timestamp with NaN. After that, you can't tell a real missing
measurement from padding. With the inner join, every value in the result came
from a source asset.

The trade-off is that the join intersects coordinates across *every member in
a key*, not per member. One short member shortens the whole key. To make this
visible without computing data, a `preprocess` hook records each asset's
dimension sizes as it opens. Afterward:

- a dimension that ends up empty raises `ValueError`;
- a dimension shorter than some contributing asset logs a warning showing
  `original -> joined` lengths.

Warming-level processing keeps the same keys and `member_id` as the
single-variable path. A shortened key narrows every member's window, and the
warning above is how users find out.

### HDP: separate unaggregated path

HDP catalog rows are stations; `station_id` is its intake variable column, and
observation fields (`tas`, `pr`, ...) live inside each station asset. Field
names therefore can't go to intake search. HDP field queries open assets with
`aggregate=False`. `_assemble_hdp_station_fields` then:

- excludes stations missing a requested field (with a warning);
- intersects each station's timestamps across fields, warning on trimming and
  failing if any included station has no shared timestamps;
- merges with `join="exact"`.

Per-station status and trim counts go to the `multi_variable_diagnostics`
Dataset attribute as JSON. `get_hdp_observation_fields` discovers available
fields from representative station assets.

## Processor support

Unsupported combinations are rejected by validators, before retrieval. No
processor handles only the first variable silently. Catalog-level processor
bans (for example, HDP disallows `clip`, `convert_units`, and `metric_calc`)
still apply.

| Processor | Multi-variable behavior |
| --- | --- |
| `convert_units` | A string target applies to every variable. A plain `{variable: unit}` mapping converts only the named variables; unknown keys fail validation. All conversions are checked before any are applied. |
| `metric_calc` | Basic reductions and percentiles run per variable. `by_variable` gives per-variable metric settings; omitted variables pass through. Percentiles become `{var}_percentiles` on a per-variable `{var}_percentile` dimension, so different levels never get union-padded. |
| `metric_calc` thresholds / `one_in_x` | Rejected by the `metric_calc` validator. |
| `bias_adjust_model_to_station` | Rejected by its validator. |
| `clip` (points) | One common cell for all variables via `Clip._get_common_grid_index` (nearest coordinate, honoring an explicit land mask). NaNs don't move the cell or trigger neighbor averaging. Single-variable behavior is unchanged. |
| `export` | NetCDF/Zarr write the Dataset as is. CSV for multi-variable queries is long format (`variable`, `value`, `units`, plus applicable coordinates) with a metadata sidecar, so independent dimensions don't form a Cartesian product. |
| Others (`concat`, `time_slice`, `drop_leap_days`, `warming_level`, `convert_to_local_time`, `filter_unadjusted_models`, `update_attributes`) | Already operate on whole Datasets and pass multiple variables through. |

## Known behavior changes and limitations

- **Mismatched spatial grids shrink.** With the inner join, variables on
  different grids are cut to their overlapping coordinates, and a warning is
  logged. Nothing is interpolated or regridded.
- **Mismatched calendars raise.** Combining assets with incompatible time
  calendars fails the query.
- **The shortening guard assumes one file per member/variable/experiment.** It
  compares each asset's length against the joined result. If a catalog splits
  one variable across several time files, each file is shorter than the
  result, so real shortening can go unreported.
- **Gridded queries have no per-record diagnostics.** Only HDP results carry
  `multi_variable_diagnostics`. Gridded exclusions and shortening appear only
  in the logs.
- **No resource guarantee.** No cap doesn't mean large requests fit in memory
  or finish quickly.
