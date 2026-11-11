# Multi-variable query scope

Status: implementation and acceptance testing are complete as of 2026-11-10.
User-facing documentation is complete in ticket `climakitae-n44.11`; the
feature scope remains tracked under epic `climakitae-n44`.

## Goal

Allow one request to retrieve any number of distinct climate variables greater
than one, without imposing a fixed variable-count cap. Return one lazy
`xarray.Dataset`, preserving variable-specific units and metadata.

This is a cross-catalog feature, not just a list-valued setter change. Work is
split into staged implementation and test tickets. The maintainer approved the
per-asset retrieval boundary below for multi-variable queries; existing
single-variable retrieval remains unchanged.

## Agreed contract

| Concern | Decision |
| --- | --- |
| Catalogs | CADCAT (WRF and LOCA2), renewable energy generation, Sup3rCC, and HDP. |
| Query boundary | One catalog and shared selection constraints per request. HDP retains one network per request. |
| Interface | Extend `variable_id` to accept a string or list/tuple of strings. Dictionary queries use the same key. |
| Input normalization | Trim names, preserve request order, reject empty collections, blank/non-string elements, and duplicates after trimming. |
| Single-variable compatibility | Strings and one-element collections retain existing single-variable behavior. Do not standardize existing return types as part of this feature. |
| Multi-variable result | One lazy Dataset, with separate named variables. No mixed-unit variable stacking into a DataArray. |
| Variable availability | An invalid ID or an ID unavailable under the shared constraints invalidates the entire request. Never return a successful subset of requested variables. |
| Simulation/station coverage | Retain only simulations or stations containing all requested variables. Report exclusions; fail if none remain. Check coverage per member, not just per intake group. |
| Time alignment | Within each otherwise eligible simulation/station, keep common timestamps and report trimming. If any such record has zero overlap, fail the whole query even when other records have overlap; do not silently exclude that record. Existing alignment across simulations/stations remains unchanged. |
| Spatial alignment | Require matching spatial coordinates within a simulation/station. No interpolation or regridding. |
| Point selection | For multi-variable queries, select one common cell by geographic location, honoring an applicable explicit spatial mask. Preserve missing measurements; do not relocate or average cells based on variable values or request order. Preserve existing single-variable behavior. |
| Missing measurements | Existing NaN measurements remain data; do not confuse them with absent fields or absent timestamp coverage. |
| Derived variables | Support native/derived mixtures and multiple derived variables wherever derived variables already work. HDP does not gain derived-variable support. |
| Custom derived output identity | In a multi-variable query, each custom derived function must produce its registered field name. If absent, fail the whole query with an actionable naming diagnostic; do not infer an alias or rename a newly created field. Preserve existing single-variable behavior. |
| Percentile output identity | Multi-variable percentile results use a `{variable}_percentiles` data variable with its own `{variable}_percentile` dimension and coordinate. Each variable can request different percentile levels without union padding. Preserve existing single-variable output formatting. |
| Multi-variable CSV | Every multi-variable query uses one long-format CSV with `variable`, `value`, `units`, and coordinate columns. Each output variable contributes rows over its own dimensions, not a Cartesian product with other variables' dimensions. Retain missing measurements as rows with missing values; leave inapplicable coordinates blank. Preserve existing CSV formatting for strings and one-element variable selections. |
| Dependency visibility | Exclude dependency-only fields from retrieval results; retain dependencies that were explicitly requested. Preserve required coordinate/CRS support metadata. |
| Failures | Preserve `get()` returning `None` with actionable logging for invalid queries. Invalid setter input continues to raise `ValueError`. |
| Successful diagnostics | Report excluded simulations/stations and trimmed timestamps through warnings and export-safe result metadata. |
| State and concurrency | Preserve query reset after `get()` and pass catalog keys explicitly; do not add mutable query state to the catalog singleton. |
| Resources | No fixed variable-count cap. This is not a promise that arbitrary-size requests fit memory or finish within a fixed time. |

The contract refers to the variables selected for retrieval. Tests must
distinguish retrieval names from processed output names. Multi-variable
percentile output follows the explicit naming contract above; other metric
outputs retain their existing naming conventions.

## Original proposed usage (contract now implemented)

This example records the API shape proposed during initial design. The
multi-variable and targeted unit-conversion behavior is now implemented.

```python
from climakitae.new_core.user_interface import ClimateData

result = (
    ClimateData()
    .catalog("cadcat")
    .activity_id("LOCA2")
    .experiment_id("historical")
    .table_id("day")
    .grid_label("d03")
    .variable_id(["tasmax", "pr"])
    .processes({
        "convert_units": {
            "by_variable": {"tasmax": "degC"}
        }
    })
    .get()
)
```

Here, precipitation remains in native units. Unit conversion uses an explicit
`by_variable` wrapper, not positional lists. Unknown variable targets invalidate
the request. Existing scalar processor settings retain single-variable behavior;
on a multi-variable request, a shared setting is accepted only if valid for
every affected variable.

## First-release processor boundary

Retain each catalog's existing restrictions. Inclusion in this table does not
enable a processor on a catalog that currently prohibits it.

| Operation | Multi-variable scope |
| --- | --- |
| Defaults and selectors | Audit model filtering, leap-day removal, concatenation, historical extension, time slicing, local-time conversion, clipping, and warming levels wherever currently supported. |
| Units | Support shared settings valid for all variables and `by_variable` targeting; omitted targets remain unchanged. |
| Basic metrics | Support reductions and percentiles independently per variable, including targeted settings where necessary. Multi-variable percentiles use variable-specific fields and percentile coordinates; other metric names remain unchanged. |
| Advanced metrics | Reject threshold-event and return-period modes on multi-variable queries in this release. |
| Station bias adjustment | Out of scope for this feature. Do not change this processor or add multi-variable-specific rejection behavior here. |
| Attributes | Preserve per-variable units, provenance, CRS/grid mapping, and shared query metadata without copying one variable's units onto another. |
| Export | Support existing NetCDF and Zarr paths, subject to their existing constraints. Multi-variable CSV uses the agreed long format; single-variable CSV remains unchanged. Verify every requested output is represented. |

Reject unsupported combinations during validation, before a processor can
silently handle only the first variable. Do not rewrite unrelated exception
handling or unsupported-processor behavior for existing single-variable queries.

## Evidence from the initial implementation assessment (2026-11-06)

- [ClimateData](../../climakitae/new_core/user_interface.py) now accepts strings,
  lists, and tuples in `variable_id`; it trims IDs, rejects empty, blank,
  non-string, and duplicate selections, preserves scalar storage for one-item
  selections, and deep-copies query snapshots. Dictionary loading and `.get()`
  use the same normalization. Main was merged on 2026-11-06 and removed the
  deprecated `variable` alias; do not reintroduce it.
- [ParameterValidator](../../climakitae/new_core/param_validation/abc_param_validation.py)
  currently recognizes one derived-variable request and records scalar
  `_derived_variable` metadata.
- [DataCatalog](../../climakitae/new_core/data_access/data_access.py) already uses
  intake `.search(...).to_dataset_dict(...)` and returns keyed Datasets.
  Internal query metadata is stripped before intake search.
- [DatasetFactory](../../climakitae/new_core/dataset_factory.py) injects
  catalog-specific default processors. [Dataset](../../climakitae/new_core/dataset.py)
  orchestrates validation, retrieval, and processing.
- [Concat](../../climakitae/new_core/processors/concatenate.py) uses outer
  alignment with compatibility overrides. Those defaults are not sufficient
  evidence that variable-level alignment and coverage are correct.
- [Historical extension](../../climakitae/new_core/processors/processor_utils.py)
  pairs historical and future data, and preserves variable attributes. Tests
  must cover multiple variables through this path.
- [ConvertUnits](../../climakitae/new_core/processors/convert_units.py) selects the
  first data variable.
- [HDPValidator](../../climakitae/new_core/param_validation/hdp_param_validator.py)
  accepts catalog keys `network_id` and `station_id`, not observation field names.

Public intake JSON schemas were inspected on 2026-11-06:

- [CADCAT](https://cadcat.s3.amazonaws.com/cae-collection.json),
  [renewable generation](https://wfclimres.s3.amazonaws.com/era/era-ren-collection.json),
  and [Sup3rCC](https://cadcat.s3.amazonaws.com/sup3r/nlr-sup3rcc-collection.json)
  use `variable_id` union aggregation and omit it from grouping attributes.
  CADCAT additionally aggregates `member_id` with `join_new`.
- [HDP](https://cadcat.s3.amazonaws.com/histwxstns/era-hdp-collection.json)
  indexes networks and stations and uses `station_id` as its intake variable
  column. Observation field selection needs a different path.

The installed version is intake-esm `2023.11.10`. The following local fixtures
use that actual package version and its real `open_esm_datastore` behavior; no
remote climate model data is used by the tests.

- CADCAT-style `join_new(member_id)` plus `union(variable_id)` produces a
  combined Dataset on the union of member and time coordinates. A variable
  absent for a member and a timestamp absent from an asset both appear as NaNs
  after this merge. A genuine NaN observation is indistinguishable from either
  kind of padding in the merged measurements.
- `to_dataset_dict(aggregate=False)` returns separate per-asset datasets.
  Their `member_id`, `variable_id`, and original time coordinates retain the
  source coverage needed to distinguish a genuine NaN from a missing variable
  or timestamp. It also retains a length-one member dimension for CADCAT.
- With distinct assets, the same derived dependencies remain separate under
  `aggregate=False`, so intake-esm does not compute the derived output at that
  stage. Under normal aggregation, the pinned registry loop computes multiple
  derived variables from the same original Dataset and retains only the last
  result. Neither direct intake behavior satisfies the required multi-derived
  contract.
- Renewables- and Sup3rCC-style `union(variable_id)` returns the requested
  native fields, while missing timestamps are outer-aligned. Different spatial
  coordinates are also outer-aligned: the combined latitude becomes the union
  of both grids, with NaN padding where a field had no source value. This does
  not satisfy the agreed spatial-coordinate equality requirement.
- Chunked local assets remain Dask-backed through aggregation. A live, read-only
  open of HDP station `ASOSAWOS_72019300117` returned a station/time Dataset
  with lazy Zarr arrays and these observation fields: `tas`, `pr`,
  `hurs_derived`, `ps`, `ps_altimeter`, `sfcWind`, `sfcWind_dir`, and `tdps`.
  The asset also contains `elevation`, `lat`, `lon`, and `_eraqc` quality
  fields for applicable observations. HDP catalog rows identify `network_id`
  and `station_id`, not observation variables; field availability is determined
  from each opened station asset.
- CADCAT's grouping includes `experiment_id`, so historical and SSP assets are
  returned in separate intake datasets. Existing historical extension is a
  subsequent ClimakitAE processor and must keep that segment distinction while
  assembling multiple variables.

The executable fixture characterization is in
[test_multivariable_intake_behavior.py](../../tests/new_core/data_access/test_multivariable_intake_behavior.py).
The public HDP check did not compute measurement arrays. It verified asset
metadata, field names, and storage-backed array types only.

### Reviewed retrieval boundary — approved for implementation

Multi-variable CADCAT, renewables, and Sup3rCC queries use intake-esm
aggregation directly (ticket `climakitae-wm2.2`, approved 2026-11-11). The
search passes `require_all_on` built from the catalog's
`aggregation_control.groupby_attrs` plus `member_id` when the catalog has that
column, so intake-esm drops any simulation/member missing a requested field.
`to_dataset_dict` then combines assets with
`xarray_combine_by_coords_kwargs={"join": "inner"}`, which leaves no NaN
padding from mismatched coordinates. Derived outputs are computed by their
registered names without dropping inputs other outputs need, and only the
requested variables are returned, in request order. Arrays stay lazy.
Single-variable queries keep the existing aggregated retrieval path.

Two behaviors differ from the earlier per-asset assembly. First, the inner
join intersects coordinates across every member in a key, not per member.
Second, per-record trimming diagnostics are gone. Members dropped by
`require_all_on` are logged at info level with their missing fields. A cheap
guard records each asset's dimension sizes while it opens (coordinates only)
and logs a warning when the join shortens a key's time or spatial dimension
below an asset's length. An empty dimension after the join fails the query.
Warming-level processing is unaffected because keys and `member_id` match
the single-variable path, but a shortened key would narrow every member's
window, which is what the warning flags.

HDP station-field selection remains a separate route in ticket
`climakitae-n44.5`; processor behavior is tracked under `.6` through `.10`.

### Branch assessment: 2026-11-06

Main was merged as `3a98538b`. At the maintainer's request, branch-only
bias-adjustment work was removed from the multi-variable feature scope. Do not
make changes to that processor as part of these tickets. The installed
intake-esm version is `2023.11.10`.

### Implementation progress snapshot: 2026-11-09

The status table below records progress at the time of this snapshot. The
final verification and remaining production-scale question are recorded in
[Implementation completion](#implementation-completion).

Ticket `climakitae-n44.2` is complete. `ClimateData.variable_id` now accepts
strings and ordered lists/tuples, trims IDs, rejects empty/blank/non-string or
duplicate selections, preserves scalar storage for one-item selections, and
returns a detached query copy. Dictionary loading and `.get()` snapshots use
the same setter path. Its interface suite passes (97 tests). Ticket
`climakitae-n44.1` is complete; its local fixture evidence informed the approved
retrieval boundary above. Tickets `.3`, `.4`, and `.5` are complete.

| Surface | Current evidence and remaining gap | Ticket |
| --- | --- | --- |
| Public and dictionary input | Setter normalization, scalar compatibility, detached query copies, dictionary loading, and `.get()` snapshots are implemented and covered. Removed `.variable()` remains absent. | `climakitae-n44.2` |
| Validation | Requested native fields and derived dependencies are checked under shared constraints. Query-local plans distinguish output variables from catalog dependencies. | `climakitae-n44.3` |
| Retrieval and derived registry | CADCAT, renewable energy generation, and Sup3rCC use intake-esm `require_all_on` plus an inner join. Results preserve requested order, derived dependencies, and lazy arrays; exclusions are logged. | `.3` and `.4` complete |
| HDP | Explicit fields are discovered from representative station assets and selected after unaggregated retrieval. Incomplete stations are excluded with JSON diagnostics; empty time intersections fail the query. No-selector and single-field paths are covered, and prohibited processors are checked through `processes`. | `.5` complete |
| Factory and pipeline | Default injection and explicit catalog-key routing already exist. Processor-wide output identity, diagnostics, and rejection gates remain. | `climakitae-n44.6` |

All 12 registered concrete processors were inspected. A Dataset-level operation
is evidence of a useful existing path, not proof that the full new contract
passes.

HDP discovery inspects one representative station per network unless the user
has selected station IDs, in which case those assets are inspected. It lists
time-varying data variables and excludes underscore-prefixed quality fields.
Observation arrays remain lazy, and real NaNs are preserved.

| Processor | Assessment | Completion work |
| --- | --- | --- |
| `filter_unadjusted_models` | Operates on entire results and metadata, not one climate field. | Verify identical member filtering across fields and empty-result failure. |
| `drop_leap_days` | Slices whole Datasets. | Multi-field leap-day, dispatch, and metadata regression tests. |
| `convert_units` | Converts only the first Dataset variable; a local probe left the second temperature field in Kelvin. | Shared all-variable conversion and validated `by_variable` targeting. |
| `warming_level` | Transforms whole Datasets before concat and invokes historical extension. | Verify aligned fields, member splitting, complete windows, center years, and preserved restrictions. |
| `concat` | Uses outer alignment and compatibility overrides; historical extension also uses outer joins. | Establish joint variable coverage before ensemble assembly; preserve historical/SSP identity and diagnostics. |
| `bias_adjust_model_to_station` | Explicitly out of scope for this feature. | No feature-specific changes; preserve current behavior. |
| `clip` | Single-point and active point-mask paths inspect the first field; the latter computes a sample slice and may average neighbors. | Implement common-cell selection across reachable point paths; verify boundary/CRS behavior. |
| `convert_to_local_time` | Changes shared time coordinates and writes timezone metadata per variable. | Verify all fields through DST handling and HDP/gridded dispatch. |
| `time_slice` | Applies shared selections to whole Datasets. | Multi-field time/season and empty-result tests. |
| `metric_calc` | Basic Dataset reductions already support several fields, but `by_variable` is accepted and ignored. A local probe returned a shared mean instead of requested percentiles. | Targeted settings, independent percentile fields/coordinates, and unsupported-mode rejection. |
| `update_attributes` | Updates Dataset-level context and cleans some attribute types. | Preserve variable metadata and serialize coverage diagnostics without unit contamination. |
| `export` | NetCDF/Zarr already accept Datasets; CSV delegates to legacy whole-Dataset flattening. | Long-format multi-variable CSV, metadata/round-trip tests, and unchanged single-variable export. |

The branch also removes a boundary disclaimer from `README.md` and changes a
blank line in `dataset.py`. Neither is part of this feature; both were left
untouched.

## Junior-engineer work breakdown

### 1. Prove retrieval and coverage behavior

Build small local intake/Zarr fixtures matching each catalog's aggregation
schema. Record how native lists, derived mixtures, and member aggregation behave.
Inspect an HDP station dataset to determine available observation field names
and whether intake preserves all fields when opening a station asset.

Verify whether intake's union aggregation pads or overrides timestamps before
ClimakitAE can inspect them. The design must preserve source-level field and
timestamp availability; a merged Dataset with NaNs is not enough to reconstruct
why values are missing.

Deliver an executable fixture-based demonstration and record the approved
assembly boundary. Define how historical/SSP pairs preserve their existing
semantics. The Dataset return contract applies to public ClimateData queries;
low-level raw-access dictionary returns need not change. Do not change external
catalog files or upgrade dependencies without identifying a demonstrated need.

### 2. Normalize input consistently

Reuse one normalization path for the fluent setter and dictionary queries.
Do not reintroduce the removed `variable` alias.
Preserve string behavior, sentinel handling, request order,
query snapshots, and state reset. Extend exploration/documentation so users can
discover HDP observation fields without treating them as intake search keys.

### 3. Validate availability and plan dependencies

Separate requested variables from dependency variables. Validate every native
ID and derived dependency against the full shared selection. Determine joint
coverage per simulation/member or station. Reject unsupported processor
combinations and unknown `by_variable` targets before retrieval where possible.
For HDP, field validation may require opening lazy station metadata first.

### 4. Retrieve and assemble multi-variable results

Implement catalog-appropriate retrieval using the reviewed boundary. HDP field
names must not be sent as nonexistent intake catalog columns. Preserve
source-level coverage information until fields are aligned within each
simulation/station, then apply existing ensemble/station assembly.

Compute each requested derived variable without prematurely dropping shared
dependencies. Return exactly the requested retrieval variables in request order,
plus necessary coordinates/support metadata. Preserve laziness and per-variable
attributes. Fail atomically when no jointly available records remain.
For multi-variable queries, verify that each derived function produces its
registered field name. Existing fallback assumptions that a function succeeded
because dependencies disappeared do not establish that the requested output exists.

### 5. Wire supported processing

Add the explicit `by_variable` configuration where needed. Use existing
processor utilities and dispatch patterns rather than a parallel processing
framework. Verify each operation in the support table, especially clipping
paths that use a first variable to select valid cells. Multi-variable single-point
and multi-point selection must use the same common-cell policy, without inferring
a spatial mask from missing measurements or averaging neighboring measurements.
Omitted targeted variables
remain unchanged. Reject unsupported combinations through catalog/processor
validators with repository-standard diagnostics.
For multi-variable percentile output, create `{variable}_percentiles` fields
with `{variable}_percentile` coordinates rather than outer-aligning independent
percentile lists. Apply this format whether percentiles are requested alone or
alongside a basic reduction; retain existing single-variable output behavior.

### 6. Verify exports and diagnostics

Round-trip multi-variable NetCDF and Zarr output, and verify CSV contains every
requested output field. Verify separated/location-based exports where already
supported. Ensure exclusions and trimming metadata survive attribute cleanup
and serialization. Keep export as the final processing step.
For every multi-variable query, assemble long-format CSV rows per variable.
Do not flatten the full Dataset first: independent percentile axes would multiply
rows and repeat measurements. Keep this new behavior in the `new_core` export
path rather than changing the legacy export interface.

### 7. Finish acceptance coverage and user documentation

Add local, network-independent acceptance tests across all four catalogs.
Add small network-backed smoke tests marked `advanced` for actual catalog
integration. Update API docstrings and runnable examples for each catalog;
use station field names verified in task 1, not guessed names.

Documentation changes in cae-notebooks and cae-website are separate follow-ups.
The implementation belongs entirely in `new_core`; legacy `core` gains no new
features.

### Task dependencies and tracker status

The seven work areas above are split into one P2 epic, `climakitae-n44`, with
18 actionable child tickets. The initial proof and input-normalization work can
start independently. Units, metrics, and clipping can proceed in parallel once
the normalized request contract is available.

#### Implementation and investigation

| Ticket | Deliverable | Prerequisites |
| --- | --- | --- |
| `climakitae-n44.1` | Prove catalog aggregation and source-coverage behavior | None |
| `climakitae-n44.2` | Normalize public and dictionary variable selections | None |
| `climakitae-n44.3` | Validate all requested variables and plan joint coverage | `.1`, `.2` |
| `climakitae-n44.4` | Assemble gridded native/derived results | `.3` |
| `climakitae-n44.5` | Select and discover HDP observation fields | `.3` |
| `climakitae-n44.6` | Preserve contracts through defaults, selectors, and metadata updates | `.4`, `.5` |
| `climakitae-n44.7` | Implement common-cell point clipping | `.2` |
| `climakitae-n44.8` | Add all-variable and targeted unit conversion | `.2` |
| `climakitae-n44.9` | Add targeted metrics and independent percentile outputs | `.2` |
| `climakitae-n44.10` | Implement long CSV and verify multi-variable export metadata | `.6`, `.9` |
| `climakitae-n44.11` | Update user documentation and examples | `.6`, `.7`, `.8`, `.9`, `.10` |

#### Tests

| Ticket | Coverage | Prerequisites |
| --- | --- | --- |
| `climakitae-n44.12` | Input normalization and atomic selection validation | `.3` |
| `climakitae-n44.13` | Source coverage, derived dependencies, and HDP assembly | `.4`, `.5` |
| `climakitae-n44.14` | Default processors, pipeline context, and rejection gates | `.6` |
| `climakitae-n44.15` | Common-cell clipping, order independence, and spatial metadata | `.7` |
| `climakitae-n44.16` | Units and exact metric values/output shapes | `.8`, `.9` |
| `climakitae-n44.17` | Export round trips and exact long-CSV row counts | `.10` |
| `climakitae-n44.18` | Public cross-catalog acceptance, laziness, and regressions | `.12` through `.17` |

Prerequisites are linked in bd, not inferred from ticket numbering. Each
implementation task still adds its own focused regression tests; the dedicated
test tickets complete the acceptance matrix rather than deferring all testing.

The initially selected tracker had no prefix and reported zero issues.
After explicit maintainer approval, the complete `.beads` directory was copied
to session storage before local reinitialization with the `climakitae` prefix.
The configured tracker then exposed 123 existing issues. Duplicate searches
found no matching multi-variable work before this epic was filed. No remote
tracker history was discarded or pushed. Auto-export reports that Git staging
failed because the repository ignores `.beads`; that ignore rule was not changed.
All 19 new issues were reopened and verified, including the 18 parent links and
30 blocking dependencies. The JSONL export contains all 142 issues, including
the new epic and children. `bd lint` passed for all 19 new issues, and the
dependency graph has no cycles.

## Acceptance matrix

| Scenario | Required observation |
| --- | --- |
| Two, three, and ten native variables | All requested fields returned in request order; no hard-coded two-variable branch or fixed cap. |
| String versus one-element collection | Same behavior and return structure as the existing string query. |
| Empty/duplicate/malformed selection | Setter raises `ValueError`; dictionary-query validation logs and fails consistently with its existing public route. |
| One valid and one invalid ID | No partial result; diagnostic names the invalid variable and relevant constraints. |
| Unequal member coverage within one intake group | Keep only complete members; no false success from group-level availability. |
| Unequal station coverage | Keep only complete stations and record exclusions; no survivors means failure. |
| Same timestamps with NaN measurements | Do not drop records merely because measurements are missing. |
| Unequal timestamps within a record | Keep common timestamps and record trimming. One otherwise eligible record with empty overlap fails the entire query, even if another record has usable overlap. |
| Incompatible spatial coordinates or calendars | Clear failure rather than interpolation, positional alignment, or silent coercion. Preserve existing intentional daily timestamp normalization. |
| Native plus derived variable | Return both requested variables with their own metadata. |
| Custom derived output name | A missing registered output field fails a multi-variable query with a diagnostic naming the expected field and actual outputs. No inferred renaming; existing single-variable behavior is unchanged. |
| Two derived variables sharing dependencies | Both outputs correct; shared sources remain available until both computations finish. |
| Explicitly requested source plus derived output | Do not delete the requested source during dependency cleanup. |
| Dependency-only fields | Not exposed as unrequested retrieval data variables. |
| Historical plus SSP | Preserve historical extension and scenario/member identity without confusing segments with simultaneous variables. |
| Shared and targeted units/basic metrics | Every affected variable is processed correctly; omitted targets unchanged; unknown targets fail. |
| Different percentile levels per variable | For temperature levels `[5, 95]` and precipitation levels `[50, 99]`, return `tasmax_percentiles` on `tasmax_percentile` and `pr_percentiles` on `pr_percentile`, with exactly their requested levels and no union padding. Cover percentile-only and percentile-plus-reduction modes, and preserve single-variable output formatting. |
| Unsupported processor/catalog combination | `get()` returns `None` with a useful diagnostic, rather than an unchanged success-shaped result. |
| Clipping and warming levels where allowed | Coordinates and simulation identity remain correct for every variable. |
| Common-cell point selection | A NaN in one variable does not move the selected cell or trigger neighbor averaging. Reversing variable order selects the same cell; single-point and multi-point paths agree. Explicit spatial masks remain applicable, and measurement arrays are not computed to infer a mask. Single-variable behavior is unchanged. |
| NetCDF/Zarr/CSV exports | Every output field and required metadata represented; supported round trips pass. |
| CSV with independent percentile dimensions | Two temperature percentiles and three precipitation percentiles produce five long-format rows per shared coordinate combination, not six Cartesian-product rows. Include `variable`, `value`, `units`, and each variable's applicable coordinate columns. |
| CSV schema and missing values | Shared-dimension and independent-dimension multi-variable results use the same long schema. Retain rows containing missing measurements; leave inapplicable coordinate columns blank. String and one-element selections retain their existing CSV format, including when a processor produces multiple output fields. |
| Laziness | No measurement-data tasks execute during ordinary retrieval/assembly; arrays remain lazy. Export and existing explicitly eager operations are assessed separately. |
| Independent queries | No leaked variables/processors between calls; no mutable per-query state on the catalog singleton. |

## Validation for implementation

Follow the repository's test authoring workflow in
[testing instructions](../../.github/instructions/testing-elegant.instructions.md).
Start with the smallest affected test selector, then run relevant module suites.
Use existing source-mirroring test directories; extend user-interface, validation,
data-access, dataset/factory, processor, and export coverage where affected.
Target at least 80% coverage on new code.

Final checks:

```bash
uv run black --check climakitae/ tests/
uv run isort --check-only climakitae/ tests/
uv run python -m pytest -n auto -m "not advanced" --no-header -q
uv run python -m pytest -m "advanced"
```

The advanced run requires network access and the repository's branch/CI policy.
These full-suite commands were not run during scoping.

### Checks performed during this assessment

- The initial ordinary targeted pytest run produced no output for several
  minutes and was stopped. A bounded rerun with external plugin autoload disabled
  completed; the cause of the initial stall was not established.
- Before merging main, the scalar setter/alias selection produced two failures
  and one pass. After the merge removed the alias, the selected scalar setter
  test still failed (`['tasmax']` versus `'tasmax'`).
- During the earlier scope cleanup, bias-adjustment tests initially failed collection because branch changes
  removed `_get_airport_code_lookup_table`. After removing those branch changes,
  this command passed with **242 passed, 12 deselected**:

  ```bash
  timeout 150s env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync python -m pytest tests/new_core/param_validation/test_bias_adjust_model_to_station_data_param_validator.py tests/new_core/processors/test_bias_adjust_model_to_station.py tests/new_core/processors/test_processor_utils.py -m 'not advanced' --no-header -q
  ```

- This post-merge command failed with **1 failed, 84 deselected**; it remains
  evidence for the normalization/compatibility work, not a passing feature test:

  ```bash
  timeout 60s env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync python -m pytest tests/new_core/test_user_interface.py -k 'test_variable_id_valid or deprecated_aliases_removed' -q --no-header
  ```

- Tiny in-memory probes reproduced first-variable-only unit conversion and
  ignored targeted metrics. Xarray probes confirmed union padding for different
  percentile lists and Cartesian-product CSV expansion.
- The bias-adjustment test result above is historical cleanup evidence only;
  that processor is not included in the multi-variable implementation scope.

## Implementation completion

The implementation tickets and acceptance tests are complete. Current
evidence includes:

- Public selection accepts strings, lists, and tuples, with no fixed variable
  count cap; scalar and one-element behavior remains compatible.
- CADCAT WRF and LOCA2, renewable generation, Sup3rCC, and HDP paths have
  network-backed multi-variable smoke tests.
- Local intake tests cover joint member/station coverage, derived dependencies,
  timestamp alignment, laziness, and HDP field assembly. Processor and export
  tests cover common-cell clipping, targeted units and metrics, independent
  percentile dimensions, and long-format CSV.
- Targeted multi-variable suites passed with 291 tests; advanced catalog smoke
  tests passed with 5 tests. The documented basic suite completed with 2335
  passed, 26 failures, and 2 collection errors. The failures target removed
  bias-adjustment helpers; collection errors include an unrelated indentation
  error in `examples/test_virtual_cubed.py` and an obsolete bias-adjustment
  import.

The remaining technical question is production-scale retrieval cost for
per-asset multi-variable assembly. It has not been benchmarked; it does not
change the verified coverage or alignment contract.


Escalate any required public-contract change to the maintainer. Do not silently
relax coverage or alignment to make a fixture pass.
