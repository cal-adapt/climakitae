# Processor: FilterUnadjustedModels

**Priority:** 250 | **Category:** Data Selection

Filter climate models by bias-adjustment status. Select only models with a-priori bias adjustment (BA models) or exclude them for certain analyses.

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `filter_type` | str | "keep_ba" or "remove_ba" |

## Algorithm

Simple filter on the `sim` dimension based on model metadata.

## Examples

```python
from climakitae.new_core.user_interface import ClimateData

# Keep only bias-adjusted models
data = (ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .variable("t2max")
    .table_id("day")
    .grid_label("d03")
    .processes({
        "filter_unadjusted_models": {
            "filter_type": "keep_ba"
        }
    })
    .get())
```

## See Also

- [Processor Index](index.md)
