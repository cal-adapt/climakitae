# Processor Reference

Processors transform climate data through a configurable pipeline. Each processor executes in priority order, handling a specific transformation task: temporal subsetting, spatial clipping, format conversion, aggregation, or metadata updates.

## Processor Registry

All available processors with brief descriptions and execution order:

| Processor | Priority | Purpose | Input | Output |
|-----------|----------|---------|-------|--------|
| [`time_slice`](time_slice.md) | 150 | Temporal subsetting by date range or seasons | Dataset | Dataset (same schema) |
| [`warming_level`](warming_level.md) | 160 | Temporal subsetting by global warming level thresholds | Dataset | Dataset (same schema) |
| [`clip`](clip.md) | 200 | Spatial subsetting by geometry, points, or stations | Dataset | Dataset (reduced spatial extent) |
| [`convert_units`](convert_units.md) | 210 | Unit conversion for climate variables | Dataset | Dataset (converted units) |
| [`concatenate`](concatenate.md) | 220 | Dataset concatenation across specified dimensions | Dataset list | Dataset (merged) |
| [`metric_calc`](metric_calc.md) | 230 | Compute derived metrics and indices | Dataset | Dataset (new variables) |
| [`bias_adjust_model_to_station`](bias_adjust_model_to_station.md) | 240 | Bias correction using station observations | Dataset | Dataset (bias-adjusted) |
| [`filter_unadjusted_models`](filter_unadjusted_models.md) | 250 | Filter models by bias-adjustment status | Dataset | Dataset (filtered sim dimension) |
| [`drop_leap_days`](drop_leap_days.md) | 260 | Remove leap days for calendar alignment | Dataset | Dataset (Feb 29 removed) |
| [`convert_to_local_time`](convert_to_local_time.md) | 270 | Convert UTC to local time zones | Dataset | Dataset (local time) |
| [`update_attributes`](update_attributes.md) | 280 | Update dataset metadata attributes | Dataset | Dataset (updated attrs) |
| [`export`](export.md) | 9999 | Write data to storage (NetCDF, Zarr, CSV, GeoTIFF) | Dataset | Files (+ optional copy returned) |

## Execution Order

Processors execute in **priority order** (ascending). Within each workflow, only registered processors with configured parameters execute.

```
Phase 1: Temporal Processing (150-160)
  ├─ time_slice (150)       — Calendar date/season filtering
  └─ warming_level (160)    — Global warming level window selection

Phase 2: Spatial & Conversion (200-280)
  ├─ clip (200)                        — Geographic filtering
  ├─ convert_units (210)               — Unit standardization
  ├─ concatenate (220)                 — Dimension merging
  ├─ metric_calc (230)                 — Index computation
  ├─ bias_adjust_model_to_station (240) — Bias correction
  ├─ filter_unadjusted_models (250)    — Model selection
  ├─ drop_leap_days (260)              — Calendar normalization
  ├─ convert_to_local_time (270)       — Timezone conversion
  └─ update_attributes (280)           — Metadata finalization

Phase 3: Export (9999)
  └─ export (9999)          — File writing & archival
```

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
