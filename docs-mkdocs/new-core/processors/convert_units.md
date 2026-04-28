# Processor: ConvertUnits

**Priority:** 210 | **Category:** Data Conversion

Convert climate variables to different units. Transform temperature (K ↔ °C ↔ °F), precipitation (mm ↔ inches), wind speed (m/s ↔ knots), and other meteorological quantities.

## Algorithm

```mermaid
flowchart TD
    Start([Input: Dataset with variables]) --> Parse["<a href='https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/convert_units.py#L60'>Parse source/target units</a>"]
    
    Parse --> Loop["<a href='https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/convert_units.py#L75'>For each variable</a>"]
    
    Loop --> Lookup["<a href='https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/convert_units.py#L85'>Look up conversion factor</a>"]
    
    Lookup --> ApplyConv["<a href='https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/convert_units.py#L95'>Apply conversion<br/>data * factor + offset</a>"]
    
    ApplyConv --> UpdateCtx["<a href='https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/convert_units.py#L105'>Update units in attrs</a>"]
    
    UpdateCtx --> End([Output: Dataset<br/>converted units])
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `source_unit` | str | Source unit (K, C, F, mm, in, m/s) |
| `target_unit` | str | Target unit (same options) |
| `variables` | list | Variables to convert (optional; default: all) |

## Supported Conversions

| Variable | Units |
|----------|-------|
| Temperature | K, °C, °F |
| Precipitation | mm/day, in/day |
| Wind Speed | m/s, knots |
| Pressure | Pa, hPa, mb |

## Examples

```python
from climakitae.new_core.user_interface import ClimateData

# Convert temperature K → °C
data = (ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .variable("t2max")
    .table_id("day")
    .grid_label("d03")
    .processes({
        "convert_units": {
            "source_unit": "K",
            "target_unit": "C"
        }
    })
    .get())
```

## See Also

- [Processor Index](index.md)
- [util/unit_conversions.py](https://github.com/cal-adapt/climakitae/blob/main/climakitae/util/unit_conversions.py)
