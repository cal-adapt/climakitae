# Processor: ConvertToLocalTime

**Registry key:** `convert_to_local_time` &nbsp;|&nbsp; **Priority:** 70 &nbsp;|&nbsp; **Category:** Calendar Processing

Convert UTC time to local time zones. Adjust timestamps for regional analysis where local time interpretation is critical (e.g., daily peak demand at local midnight vs UTC).

## Algorithm

Reindex time coordinate from UTC to specified timezone(s).

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `timezone` | str or dict | Timezone (e.g., "US/Pacific", "US/Eastern") |
| `lon_based` | bool | Auto-determine timezone from longitude (optional) |

## Examples

```python
from climakitae.new_core.user_interface import ClimateData

# Convert to Pacific time
data = (ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .variable("t2max")
    .table_id("1hr")
    .grid_label("d03")
    .processes({
        "convert_to_local_time": {
            "timezone": "US/Pacific"
        }
    })
    .get())
```

## See Also

- [Processor Index](index.md)
