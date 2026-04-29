# Processor: DropLeapDays

**Registry key:** `drop_leap_days` &nbsp;|&nbsp; **Priority:** 1 &nbsp;|&nbsp; **Category:** Calendar Processing

Remove leap days (February 29th) from climate data. Align datasets with different calendar representations for consistent time-series analysis.

## Algorithm

Filter time dimension to exclude any dates matching February 29.

## Parameters

None — no configuration needed.

## Examples

```python
from climakitae.new_core.user_interface import ClimateData

# Remove leap days for 365-day year consistency
data = (ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .variable("t2max")
    .table_id("day")
    .grid_label("d03")
    .processes({
        "drop_leap_days": {}
    })
    .get())
```

## See Also

- [Processor Index](index.md)
