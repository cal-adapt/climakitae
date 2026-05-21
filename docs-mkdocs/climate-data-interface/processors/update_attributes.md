# Processor: UpdateAttributes

**Registry key:** `update_attributes` &nbsp;|&nbsp; **Priority:** 9998 &nbsp;|&nbsp; **Category:** Metadata Management

Internal "finalizer" that merges every processing-step note that other processors stashed in `context["new_attrs"]` onto the result's `.attrs`, and fills in standard coordinate attributes from `common_attrs`. Runs near the end of the pipeline (priority 9998) and is added implicitly by `ClimateData`.

## Algorithm

```mermaid
flowchart TD
    Start([execute]) --> CtxCheck{self.name in context?}
    CtxCheck -->|No| EnsureCtx[update_context:<br/>add update_attributes marker<br/>into context.new_attrs]
    CtxCheck -->|Yes| ResultMatch
    EnsureCtx --> ResultMatch{match result}

    ResultMatch -->|dict| LoopDict[For each value:<br/>merge new_attrs into attrs;<br/>fill dim attrs from common_attrs]
    ResultMatch -->|Dataset / DataArray| Single[merge new_attrs into attrs;<br/>update dim attrs from common_attrs]
    ResultMatch -->|list / tuple| LoopList[For each item:<br/>merge new_attrs into attrs]
    ResultMatch -->|other| TypeErr[raise TypeError]

    LoopDict --> Done
    Single --> Done
    LoopList --> Done
    Done([Output: result with<br/>processing-history attrs])

    click Start "https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/update_attributes.py#L79" "execute"
    click EnsureCtx "https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/update_attributes.py#L123" "update_context"
```

### What gets merged

- `context["new_attrs"]` is a dict that earlier processors populate from inside their own `update_context` methods (e.g. `clip` records the boundary, `time_slice` records the start/end, `convert_units` records source and target units).
- The merge uses `attrs | context["new_attrs"]`, so `new_attrs` keys overwrite existing ones with the same name.
- For a single Dataset/DataArray, dim coords also get `common_attrs.get(dim, {})` applied (CF-compliant labels for `time`, `lat`, `lon`, etc.).

## Parameters

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `value` | `Any` | `UNSET` | Unused. The processor's behavior depends entirely on the pipeline `context`. Including this in `.processes({...})` manually has no extra effect. |

## Example (implicit)

You normally never reference this processor directly:

```python
from climakitae.new_core.user_interface import ClimateData

data = (ClimateData()
    .catalog("cadcat").activity_id("WRF").institution_id("UCLA")
    .variable("t2max").table_id("day").grid_label("d03")
    .processes({
        "time_slice": ("2030-01-01", "2060-12-31"),
        "clip": "Los Angeles",
    })
    .get())

# update_attributes runs automatically at priority 9998. The result's .attrs
# now includes summaries of the time_slice and clip operations as recorded
# by their respective update_context methods.
print(data.attrs)
```

## Code References

| Method | Lines | Purpose |
|--------|-------|---------|
| `__init__` | [67–77](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/update_attributes.py#L67) | Store unused `value`; set name |
| `execute` | [79–121](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/update_attributes.py#L79) | Ensure context entry; merge `new_attrs` per result type |
| `update_context` | [123–144](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/update_attributes.py#L123) | Add `"update_attributes"` marker into `context["new_attrs"]` |
| `set_data_accessor` | [146–](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/update_attributes.py#L146) | Unused stub |

## See also

- [Processor index](index.md)
- [`climakitae/new_core/processors/update_attributes.py`](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/update_attributes.py)
- `common_attrs` (CF coord defaults) used here lives alongside the processors module.
