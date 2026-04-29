# Processor: UpdateAttributes

**Registry key:** `update_attributes` &nbsp;|&nbsp; **Priority:** 9998 &nbsp;|&nbsp; **Category:** Metadata Management

Update dataset and variable attributes with custom metadata. Add provenance information, units, descriptions, or other CF-compliant metadata.

## Algorithm

Apply attribute updates to dataset and/or individual variables.

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `attrs` | dict | Global dataset attributes |
| `var_attrs` | dict | Per-variable attributes (nested dict) |

## Examples

```python
from climakitae.new_core.user_interface import ClimateData

# Add custom metadata
data = (ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .variable("t2max")
    .table_id("day")
    .grid_label("d03")
    .processes({
        "update_attributes": {
            "attrs": {
                "title": "California WRF Maximum Temperature",
                "source": "UCLA WRF Downscaling",
                "institution": "Cal-Adapt Analytics Engine"
            },
            "var_attrs": {
                "t2max": {
                    "units": "K",
                    "long_name": "2-meter maximum air temperature"
                }
            }
        }
    })
    .get())
```

## See Also

- [Processor Index](index.md)
- [Architecture → Context Metadata](../architecture.md#context-metadata)
