# Processor: WarmingLevel

**Registry key:** `warming_level` &nbsp;|&nbsp; **Priority:** 10 &nbsp;|&nbsp; **Category:** Temporal Processing

Subset climate data by global warming level thresholds instead of calendar dates. Transform time-series data to a warming-level-centric approach for climate impact analysis aligned with IPCC warming scenarios.

## Algorithm

### Execution Flow

1. **Initialization** (lines 73–111): Validate dict, store `warming_levels`, `warming_level_window` (default 15), `warming_level_months` (default `UNSET`), `add_dummy_time` (default `False`). **Eagerly loads two CSV lookup tables** (`gwl_1850-1900ref.csv` at L91 and `gwl_1981-2010ref.csv` at L94). Sets `self.name = "warming_level_simple"` (note: this is the metadata key written to context).
2. **Defensive Reload** (lines 120–129): If `warming_level_times` is somehow `None`, attempt to reload from CSV; raise `RuntimeError` on failure.
3. **Member ID Reformatting** (line 132 → method at line 343): `reformat_member_ids` splits any `member_id` dimension into separate dict keys with `key.member_id` suffix.
4. **Time Domain Extension** (line 135): `extend_time_domain` (helper) splices historical onto SSP scenarios so the data range covers 1980/1850 – 2100.
5. **Center-Year Lookup** (line 150 → method at line 376): `get_center_years` returns `{key: [year_for_wl1, year_for_wl2, ...]}`. Lookup uses tuple `(activity_id, member_id, source_id)` parsed from each key.
6. **Per-Key, Per-WL Slicing** (lines 165–249, nested loops):
   - Skip key if no center years (line 168).
   - Skip individual WLs where `year` is `None`/`NaN` (lines 174–177).
   - Compute window: `start_year = center - window`, `end_year = center + window - 1` (lines 184–185).
   - Skip incomplete WLs via `_determine_is_complete_wl` (lines 188–198).
   - `data.sel(time=slice(start, end))` (line 201).
   - Drop Feb 29 leap days (lines 204–205).
   - Optional month filter via `warming_level_months` (lines 208–211).
   - Swap `time` dim for `time_delta` index (range `-L/2 .. L/2`) (lines 215–219).
   - `expand_dims({"warming_level": [wl]})` (line 222) + assign `simulation` coord (line 225).
   - Append to local `slices` list.
7. **Concatenation** (lines 251–253): `xr.concat(slices, dim="warming_level", join="outer", fill_value=np.nan)`.
8. **Optional Dummy Time** (line 256): If `add_dummy_time=True`, `add_dummy_time_to_wl(ret[key])` restores a synthetic `time` axis.
9. **Context Bookkeeping** (lines 267, 271): Store `_sim_centered_years` mapping in context for downstream `concatenate` reconstruction; `update_context` writes a description under `context["new_attrs"]["warming_level_simple"]`.

## Parameters

| Parameter | Type | Required | Default | Description | Constraints |
|-----------|------|----------|---------|-------------|-------------|
| `warming_levels` | list[float] | ✓ | — | Global warming levels (°C above pre-industrial) | [0.8, 1.5, 2.0, 2.5, 3.0] common; 1.5–4.0 typical |
| `warming_level_window` | int | | 15 | Years before/after central year to include | ≥1; 15 year window typical (30 year total) |
| `warming_level_months` | list[int] | | UNSET | Months to keep (1–12) | E.g., [6,7,8] for JJA; UNSET = all months |
| `add_dummy_time` | bool | | False | Replace offset-from-center dimension with dummy time | Useful for tools requiring time dimension |

## Code References

| Method | Lines | Purpose |
|--------|-------|---------|
| `__init__` | [73–111](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/warming_level.py#L73) | Validate config, store params, eagerly load GWL CSV lookup tables (L91, L94) |
| `execute` | [113–273](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/warming_level.py#L113) | Reformat member ids, extend time, look up center years, slice + reshape per GWL, concatenate |
| `update_context` | [308–329](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/warming_level.py#L308) | Write `new_attrs["warming_level_simple"]` description |
| `set_data_accessor` | [331–341](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/warming_level.py#L331) | Store catalog reference (currently unused) |
| `reformat_member_ids` | [343–374](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/warming_level.py#L343) | Split data with `member_id` dim into separate dict entries |
| `get_center_years` | [376–521](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/warming_level.py#L376) | Per-(key, wl) center-year lookup against `warming_level_times` and `warming_level_times_idx` |

## Examples

### Single Warming Level

```python
from climakitae.new_core.user_interface import ClimateData

# Extract data at 1.5°C warming
data = (ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .experiment_id("ssp245")
    .variable("t2max")
    .table_id("day")
    .grid_label("d03")
    .processes({
        "warming_level": {
            "warming_levels": [1.5]
        }
    })
    .get())
```

### Multiple Warming Levels

```python
# Compare 1.5°C, 2.0°C, and 3.0°C warming levels
data = (ClimateData()
    .catalog("cadcat")
    .activity_id("LOCA2")
    .experiment_id("ssp370")
    .variable("tasmax")
    .table_id("day")
    .grid_label("d02")
    .processes({
        "warming_level": {
            "warming_levels": [1.5, 2.0, 3.0]
        }
    })
    .get())

# data.warming_level is now a coordinate with 3 values
# Access with: data.sel(warming_level=1.5)
```

### Custom Window

```python
# Use 20-year windows (instead of default 15)
data = (ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .experiment_id("ssp585")
    .variable("pr")
    .table_id("mon")
    .grid_label("d03")
    .processes({
        "warming_level": {
            "warming_levels": [2.0, 2.5],
            "warming_level_window": 20
        }
    })
    .get())
```

### Seasonal Filter

```python
# Summer (JJA) only at 2°C warming
data = (ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .experiment_id("ssp245")
    .variable("t2max")
    .table_id("day")
    .grid_label("d03")
    .processes({
        "warming_level": {
            "warming_levels": [2.0],
            "warming_level_months": [6, 7, 8]  # June, July, August
        }
    })
    .get())
```

### Chained with Clipping

```python
# Full workflow: clip + warming level + export
data = (ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .experiment_id("ssp245")
    .variable("t2max")
    .table_id("day")
    .grid_label("d03")
    .processes({
        "clip": "San Francisco Bay",
        "warming_level": {
            "warming_levels": [1.5, 2.0, 3.0],
            "warming_level_window": 15
        },
        "export": {
            "filename": "sf_warming_levels",
            "file_format": "NetCDF"
        }
    })
    .get())
```

## Implementation Details

### Global Warming Level Lookup

GWL timing is pre-computed from climate model simulations and stored in CSV files shipped with `climakitae`:

- **`gwl_1850-1900ref.csv`**: Year/timestamp when each `(activity_id, member_id, source_id)` triple reaches each integer warming level (1850–1900 reference period). Used when the requested WL exists as a column in the table.
- **`gwl_1981-2010ref.csv`** (loaded as `warming_level_times_idx`): Time-indexed table of running warming-level estimates per simulation column. Used as a fallback when the requested WL is not a column in `gwl_1850-1900ref.csv` — the processor finds the **first** time the simulation column crosses the requested level.

Lookup keys parse the dict key as `key.split(".")` and use `(key_list[2], member_id, key_list[3])`, which corresponds to `(activity_id, member_id, source_id)` in catalog terms. Missing entries log a warning and append `np.nan` for that warming level (the slice is then skipped).

### Time Windows

The processor creates an asymmetric ([center-window, center+window-1]) window:

```python
start_year = center_year - self.warming_level_window
end_year = center_year + self.warming_level_window - 1
da_slice = data.sel(time=slice(f"{start_year}", f"{end_year}"))
```

With default `warming_level_window=15`, this is a 30-year span. Feb 29 is then dropped to keep slice lengths consistent across leap and non-leap years.

### `time_delta` Reindexing

After slicing, the `time` dimension is replaced with a centered integer offset:

```python
length = da_slice.sizes["time"]
time_delta = range(-length // 2, length // 2)
da_slice = da_slice.swap_dims({"time": "time_delta"}).drop_vars("time")
da_slice = da_slice.assign_coords(time_delta=time_delta)
```

This lets multiple warming levels (with different absolute years but matching window length) share a common dimension before `xr.concat`.

### Edge Cases

- **Model doesn't reach GWL**: `get_center_years` appends `np.nan`; the inner loop skips that WL with a warning.
- **Incomplete window**: `_determine_is_complete_wl` returns `False` when the simulation lacks data on either end; skipped with a warning.
- **No valid WLs for a key**: The key is removed from the result dict (lines 239–247).
- **Monthly filtering**: Applied after time slicing, so per-WL counts may vary if some months are dropped.

### Dummy Time (Optional)

Some downstream tools require a real `time` dimension. Setting `add_dummy_time=True` calls `add_dummy_time_to_wl(ds)` which adds a synthetic, monotonically increasing time coordinate back onto the result. Useful for visualization but not climatologically meaningful.

### Context Side Effects

- `context["_sim_centered_years"]`: dict mapping each key to the list of valid center years (consumed downstream by `concatenate`).
- `context["new_attrs"]["warming_level_simple"]`: description string (note the `_simple` suffix — this comes from `self.name`).

## Common Patterns

### Compare Scenarios at Same Warming Level

```python
# Historical, SSP2-4.5, SSP5-8.5 at 2°C warming
scenarios = ["historical", "ssp245", "ssp585"]
results = {}

for scenario in scenarios:
    results[scenario] = (ClimateData()
        .catalog("cadcat")
        .activity_id("WRF")
        .experiment_id(scenario)
        .variable("t2max")
        .table_id("day")
        .grid_label("d03")
        .processes({
            "warming_level": {"warming_levels": [2.0]}
        })
        .get())
```

### Model Uncertainty Across Warming Levels

```python
# Get all 5 WRF models at multiple warming levels
data = (ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .experiment_id("ssp370")
    .variable("pr")
    .table_id("mon")
    .grid_label("d03")
    .processes({
        "warming_level": {
            "warming_levels": [1.5, 2.0, 2.5, 3.0]
        }
    })
    .get())

# data.dims: (warming_level, sim, lat, lon)
# Compute multi-model mean across warming levels
multi_model_mean = data.mean(dim="sim")
```

## See Also

- [Processor Index](index.md)
- [Time Slice Processor](time_slice.md) — Alternative: calendar-based temporal subsetting
- [Architecture → Warming Levels Concept](../concepts.md#global-warming-levels)
- [How-To Guides → Warming Level Analysis](../howto/warming-levels.md)
- Cal-Adapt GWL Resources: [IPCC AR6 Warming Levels](https://www.ipcc.ch/)
