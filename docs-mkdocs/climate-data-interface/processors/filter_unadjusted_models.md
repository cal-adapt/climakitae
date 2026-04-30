# Processor: FilterUnadjustedModels

**Registry key:** `filter_unadjusted_models` &nbsp;|&nbsp; **Priority:** 0 &nbsp;|&nbsp; **Category:** Data Selection

Drop WRF model entries whose `(activity_id, source_id, member_id)` tuple is in the curated `NON_WRF_BA_MODELS` list (models without a-priori bias adjustment). Used by default in the new-core pipeline so downstream analysis runs against a homogeneous WRF ensemble.

## Algorithm

```mermaid
flowchart TD
    Start([execute]) --> ValueMatch{match self.value}

    ValueMatch -->|"yes"| ContainsCheck{_contains_unadjusted_models?}
    ContainsCheck -->|Yes| WarnRemoved[Log warning:<br/>models removed]
    ContainsCheck -->|No| ReturnAsIs[Return result unchanged]
    WarnRemoved --> Remove[_remove_unadjusted_models<br/>(per-entry filter; single Dataset → None)]
    Remove --> End

    ValueMatch -->|"no"| ContainsCheck2{_contains_unadjusted_models?}
    ContainsCheck2 -->|Yes| WarnKept[Log warning:<br/>proceed with caution]
    ContainsCheck2 -->|No| Pass[Return result unchanged]
    WarnKept --> Pass

    ValueMatch -->|other| Raise[Raise ValueError]

    ReturnAsIs --> End([Output])
    Pass --> End

    click Start "https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/filter_unadjusted_models.py#L65" "execute"
    click ContainsCheck "https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/filter_unadjusted_models.py#L129" "_contains_unadjusted_models"
    click ContainsCheck2 "https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/filter_unadjusted_models.py#L129" "_contains_unadjusted_models"
    click Remove "https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/filter_unadjusted_models.py#L173" "_remove_unadjusted_models"
```

### Membership test

`_contains_unadjusted_models` reads `intake_esm_attrs:activity_id`, `intake_esm_attrs:source_id`, and `intake_esm_attrs:member_id` from the dataset's `attrs`, joins them as `f"{activity}_{source}_{member}"`, and checks for membership in the `NON_WRF_BA_MODELS` constant in [`climakitae/core/constants.py`](https://github.com/cal-adapt/climakitae/blob/main/climakitae/core/constants.py).

For dict / list / tuple inputs the test recurses, returning `True` if any contained dataset is unadjusted. For a single matching `xr.Dataset`/`xr.DataArray`, `_remove_unadjusted_models` returns `None`; for a dict/list/tuple it removes only the matching entries and preserves the container type.

## Parameters

The processor takes a **single string** (case-insensitive):

| Field | Type | Allowed | Default | Description |
|-------|------|---------|---------|-------------|
| `value` | `str` | `"yes"` / `"no"` | `"yes"` | `"yes"` removes unadjusted entries; `"no"` keeps them but logs a warning. Anything else raises `ValueError`. |

!!! note "Default behavior"
    `ClimateData` inserts this processor automatically for bias-relevant WRF
    queries. To explicitly *include* unadjusted models, pass
    `"filter_unadjusted_models": "no"`.

## Examples

```python
from climakitae.new_core.user_interface import ClimateData

# Default: drop unadjusted models
data = (ClimateData()
    .catalog("cadcat").activity_id("WRF").institution_id("UCLA")
    .variable("t2max").table_id("day").grid_label("d03")
    .processes({"filter_unadjusted_models": "yes"})
    .get())

# Opt out: keep all models (warning logged)
data_all = (ClimateData()
    .catalog("cadcat").activity_id("WRF").institution_id("UCLA")
    .variable("t2max").table_id("day").grid_label("d03")
    .processes({"filter_unadjusted_models": "no"})
    .get())
```

## Code References

| Method | Lines | Purpose |
|--------|-------|---------|
| `__init__` | [51–63](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/filter_unadjusted_models.py#L51) | Lowercase + store value |
| `execute` | [65–127](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/filter_unadjusted_models.py#L65) | `match self.value`; warn + filter or warn + pass |
| `_contains_unadjusted_models` | [129–171](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/filter_unadjusted_models.py#L129) | Build model_id from `intake_esm_attrs:*`, check NON_WRF_BA_MODELS |
| `_remove_unadjusted_models` | [173–225](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/filter_unadjusted_models.py#L173) | Drop matching entries; preserve container type |
| `update_context` | [227–246](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/filter_unadjusted_models.py#L227) | (No-op for context attrs in this processor's flow) |
| `set_data_accessor` | [248–](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/filter_unadjusted_models.py#L248) | Unused stub |

## See also

- [Processor index](index.md)
- [`climakitae/new_core/processors/filter_unadjusted_models.py`](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/filter_unadjusted_models.py)
- [`NON_WRF_BA_MODELS` and `WRF_BA_MODELS` in `core/constants.py`](https://github.com/cal-adapt/climakitae/blob/main/climakitae/core/constants.py)
