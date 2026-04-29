# Processor Reference

**What each processor does, with parameters and edge cases.** If you'd rather
start from a goal ("clip to a county", "export to Zarr"), use the
[How-To Guides](../howto.md) instead.

Processors transform climate data through a configurable pipeline. Each processor executes in priority order, handling a specific transformation task: temporal subsetting, spatial clipping, format conversion, aggregation, or metadata updates.

## Processor Registry

All available processors, listed in execution order (lowest priority first).

The **registry key** is the string you pass to `.processes({...})`. It is not always identical to the source filename — for example, the concatenation processor is registered under the key `"concat"`.

| Registry key | Priority | Page | Purpose |
|--------------|---------:|------|---------|
| `filter_unadjusted_models` | 0 | [filter_unadjusted_models](filter_unadjusted_models.md) | Filter models by a-priori bias-adjustment status (default: enabled for WRF) |
| `drop_leap_days` | 1 | [drop_leap_days](drop_leap_days.md) | Remove Feb 29 for calendar alignment |
| `convert_units` | 5 | [convert_units](convert_units.md) | Unit conversion for climate variables |
| `warming_level` | 10 | [warming_level](warming_level.md) | Subset around global warming-level thresholds |
| `concat` | 50 | [concatenate](concatenate.md) | Concatenate datasets along the `sim` (or named) dimension |
| `bias_adjust_model_to_station` | 60 | [bias_adjust_model_to_station](bias_adjust_model_to_station.md) | Quantile-delta-mapping bias correction using HadISD station observations |
| `clip` | 65 | [clip](clip.md) | Spatial subsetting by boundary, point(s), bbox, station, or shapefile |
| `convert_to_local_time` | 70 | [convert_to_local_time](convert_to_local_time.md) | Convert UTC to a local timezone |
| `time_slice` | 150 | [time_slice](time_slice.md) | Temporal subsetting by date range and optional season |
| `metric_calc` | 7500 | [metric_calc](metric_calc.md) | Derived metrics, threshold exceedance, and 1-in-X return-period analysis |
| `update_attributes` | 9998 | [update_attributes](update_attributes.md) | Update dataset/variable attributes |
| `export` | 9999 | [export](export.md) | Write to NetCDF, Zarr, CSV, or GeoTIFF |

## Execution Order

Processors execute in **priority order** (ascending — lowest priority runs first). Within each workflow, only registered processors with configured parameters execute.

```
Phase 1: Catalog refinement (priority 0–10)
  ├─ filter_unadjusted_models (0)    — Drop models without a-priori bias adjustment
  ├─ drop_leap_days (1)              — Calendar normalization
  ├─ convert_units (5)               — Unit standardization
  └─ warming_level (10)              — Global-warming-level window selection

Phase 2: Aggregation, correction, spatial (priority 50–70)
  ├─ concat (50)                     — Concatenate along sim (or other) dimension
  ├─ bias_adjust_model_to_station (60) — Station-based bias correction (WRF only)
  ├─ clip (65)                       — Geographic filtering
  └─ convert_to_local_time (70)      — Timezone conversion

Phase 3: Temporal subsetting & metrics (priority 150–7500)
  ├─ time_slice (150)                — Calendar date / season filtering
  └─ metric_calc (7500)              — Derived metrics & extreme-value analysis

Phase 4: Finalization (priority 9998–9999)
  ├─ update_attributes (9998)        — Metadata finalization
  └─ export (9999)                   — File writing & archival
```

!!! tip "Why does spatial clipping run before time slicing?"
    `clip` (priority 65) runs before `time_slice` (priority 150) so that spatial
    subsetting reduces the data volume that subsequent steps must read. Reducing
    spatial extent first is usually the dominant performance win for gridded
    queries.

## Architecture

Each processor implements three core methods:

- **`__init__(value: Dict)`** — Parse and validate parameters
- **`execute(data: Dataset, context: Dict) → Dataset`** — Apply transformation, handle branching (lists, dicts, single datasets)
- **`update_context(context: Dict) → None`** — Record operation in dataset attributes

The processor registry uses decorators:

```python
@register_processor(key="processor_name", priority=N)
class ProcessorName(DataProcessor):
    def execute(self, data, context):
        # Transform data
        return data
    
    def update_context(self, context):
        # Record metadata
        pass
```

## How to Read These Diagrams

Each processor page includes:

- **Algorithm flowchart** — Execution flow with decision points
- **Code links** — Click boxes to jump to source code
- **Parameter table** — Valid inputs and constraints
- **Usage example** — How to invoke via `ClimateData.processes()`

## Adding a New Processor

See [Architecture → Extension Guide → Add a Processor](../architecture.md#add-a-processor-4-step-guide) for step-by-step instructions. Summary:

1. Create `climakitae/new_core/processors/my_processor.py` inheriting from `DataProcessor`
2. Implement `execute()` and `update_context()`
3. Decorate with `@register_processor(key="my_processor", priority=N)` 
4. Import in `__init__.py` to register at import time
5. Add documentation page `docs-mkdocs/new-core/processors/my_processor.md`
