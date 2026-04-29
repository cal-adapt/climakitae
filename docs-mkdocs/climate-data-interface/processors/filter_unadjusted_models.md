# Processor: FilterUnadjustedModels

**Registry key:** `filter_unadjusted_models` &nbsp;|&nbsp; **Priority:** 0 &nbsp;|&nbsp; **Category:** Data Selection

Filter climate models by a-priori bias-adjustment status. By default the new-core pipeline applies this processor to WRF queries, dropping `source_id` values that lack a-priori bias adjustment so that downstream analysis works on a homogeneous ensemble.

## Parameter shape

The processor takes a **single string** with one of two values:

```python
.processes({"filter_unadjusted_models": "yes"})  # default — drop unadjusted models
.processes({"filter_unadjusted_models": "no"})   # opt out — keep all models
```

| Field | Type | Allowed values | Default | Description |
|-------|------|----------------|---------|-------------|
| `value` | `str` | `"yes"`, `"no"` (case-insensitive) | `"yes"` | `"yes"` removes the models listed in `NON_WRF_BA_MODELS` (see [`climakitae/core/constants.py`](https://github.com/cal-adapt/climakitae/blob/main/climakitae/core/constants.py)). `"no"` is a no-op. |

!!! note "Default behavior"
    This processor is added automatically to the pipeline by `ClimateData` for
    bias-relevant WRF queries. If you want to *include* unadjusted models, pass
    `"filter_unadjusted_models": "no"` explicitly.

## Algorithm

Filters the `source_id` (or `sim`) dimension by membership in the curated WRF a-priori-adjusted model list.

## Examples

```python
from climakitae.new_core.user_interface import ClimateData

# Default: drop unadjusted models (this is what runs implicitly for WRF)
data = (ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .variable("t2max")
    .table_id("day")
    .grid_label("d03")
    .processes({"filter_unadjusted_models": "yes"})
    .get())

# Opt-out: keep all models
data_all = (ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .variable("t2max")
    .table_id("day")
    .grid_label("d03")
    .processes({"filter_unadjusted_models": "no"})
    .get())
```

## See also

- [Processor index](index.md)
- [`climakitae/new_core/processors/filter_unadjusted_models.py`](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/filter_unadjusted_models.py)
- Cal-Adapt: Analytics Engine — [About climate projections and models](https://analytics.cal-adapt.org/guidance/about_climate_projections_and_models)
