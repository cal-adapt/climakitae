# Processor: BiasAdjustModelToStation

**Priority:** 240 | **Category:** Data Refinement

Apply bias correction to WRF model data using quantile delta mapping (QDM) with weather station observations. Reduce systematic errors between model simulations and observed climate.

## Algorithm

```mermaid
flowchart TD
    Start([Input: WRF Dataset]) --> CheckActivity["<a href='https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/bias_adjust_model_to_station.py#L85'>Verify activity_id<br/>= WRF</a>"]
    
    CheckActivity -->|Not WRF| Warn["<a href='https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/bias_adjust_model_to_station.py#L95'>Log warning<br/>return unmodified</a>"]
    CheckActivity -->|WRF| LoadObs["<a href='https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/bias_adjust_model_to_station.py#L105'>Load station<br/>observations</a>"]
    
    Warn --> End1([Return original data])
    LoadObs --> CheckVar["<a href='https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/bias_adjust_model_to_station.py#L115'>Check variable<br/>support</a>"]
    
    CheckVar -->|Unsupported| Warn2["<a href='https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/bias_adjust_model_to_station.py#L120'>Log note<br/>return unmodified</a>"]
    CheckVar -->|Supported| QDM["<a href='https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/bias_adjust_model_to_station.py#L130'>Apply QDM<br/>quantile matching</a>"]
    
    Warn2 --> End2([Return original data])
    QDM --> UpdateCtx["<a href='https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/bias_adjust_model_to_station.py#L145'>Update context</a>"]
    
    UpdateCtx --> End3([Output: bias-adjusted<br/>Dataset])
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `stations` | list | Station codes (e.g., ["KSAC", "KSFO"]) |

## Requirements

- **Activity ID**: Must be "WRF" (dynamical downscaling)
- **Variables**: Currently supports `t2` (hourly temperature)
- **Data**: Must span sufficient historical period for QDM training

## Examples

```python
from climakitae.new_core.user_interface import ClimateData

# Bias-correct WRF data at Sacramento observation point
data = (ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .variable("t2")
    .table_id("1hr")
    .grid_label("d03")
    .processes({
        "bias_adjust_model_to_station": {
            "stations": ["KSAC"]
        }
    })
    .get())
```

## See Also

- [Processor Index](index.md)
- [How-To Guides → Bias Correction](../howto.md#bias-correction-station-localization)
- [Architecture → Bias Correction](../architecture.md#bias-correction)
