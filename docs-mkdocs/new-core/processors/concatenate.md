# Processor: Concatenate

**Priority:** 220 | **Category:** Data Assembly

Merge multiple climate datasets by concatenating along specified dimensions. Combine data from different models, scenarios, or time periods into unified arrays.

## Algorithm

```mermaid
flowchart TD
    Start([Input: list of Datasets]) --> CheckInput{Input<br/>valid?}
    
    CheckInput -->|Empty| LogWarn["<a href='https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/concatenate.py#L50'>Log warning<br/>no data</a>"]
    CheckInput -->|Valid| ParseDim["<a href='https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/concatenate.py#L65'>Extract dimensions<br/>from parameters</a>"]
    
    LogWarn --> End1([Return input])
    ParseDim --> Loop["<a href='https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/concatenate.py#L75'>For each dimension</a>"]
    
    Loop --> LoopConcat["<a href='https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/concatenate.py#L85'>xr.concat along dim</a>"]
    LoopConcat --> UpdateCtx["<a href='https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/concatenate.py#L95'>Update context</a>"]
    
    UpdateCtx --> End2([Output: merged Dataset])
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `dim` | str or list | ✓ | — | Dimension(s) to concatenate along |

## Examples

```python
from climakitae.new_core.user_interface import ClimateData

# Combine multiple models along 'sim' dimension
data1 = ClimateData().catalog("cadcat").activity_id("WRF")...get()
data2 = ClimateData().catalog("cadcat").activity_id("LOCA2")...get()

# Merge via concatenate processor
merged = (ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .variable("t2max")
    .table_id("day")
    .grid_label("d03")
    .processes({
        "concatenate": {"dim": "sim"}
    })
    .get())
```

## See Also

- [Processor Index](index.md)
- [How-To Guides → Multi-Model Analysis](../howto.md#multi-model-ensemble)
