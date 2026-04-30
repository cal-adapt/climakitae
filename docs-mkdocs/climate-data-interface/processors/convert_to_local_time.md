# Processor: ConvertToLocalTime

**Registry key:** `convert_to_local_time` &nbsp;|&nbsp; **Priority:** 70 &nbsp;|&nbsp; **Category:** Calendar Processing

Convert hourly time coordinates from UTC to the local time zone inferred from the data's central latitude/longitude. Optionally reindex the time axis to a regular hourly grid that handles daylight saving time gaps and duplicates.

## Algorithm

1. If `convert == "yes"`, branch by catalog:
   - `hdp` → `_convert_to_local_time_hdp` (use first station's lat/lon).
   - otherwise → `_convert_to_local_time_gridded` (compute central lat/lon; bail out with a warning if the dataset is not hourly).
2. `_find_timezone_and_convert` looks up the IANA zone for that point via `timezonefinder`, converts the time index from UTC to that zone with `pandas`, and tags the variable's `timezone` attribute.
3. If `reindex_time_axis == "yes"`, deduplicate timestamps that collide on DST fall-back and fill the missing spring-forward hour with NaN.

```mermaid
flowchart TD
    Start([execute]) --> ValueMatch{match self.value}
    ValueMatch -->|"no"| Passthrough[Return result unchanged]
    ValueMatch -->|other| Raise[Raise ValueError]
    ValueMatch -->|"yes"| CatalogCheck{context['catalog']<br/>== 'hdp'?}

    CatalogCheck -->|Yes| HDPFunc[func = _convert_to_local_time_hdp]
    CatalogCheck -->|No| GridFunc[func = _convert_to_local_time_gridded]

    HDPFunc --> ResultMatch{match result}
    GridFunc --> ResultMatch
    ResultMatch -->|dict| LoopDict[For each value: func]
    ResultMatch -->|Dataset / DataArray| Single[func]
    ResultMatch -->|list / tuple| LoopList[Per-item func, preserve container]
    ResultMatch -->|other| LogWarn[Log warning]

    LoopDict --> Convert
    Single --> Convert
    LoopList --> Convert

    Convert["_find_timezone_and_convert<br/>(timezonefinder + tz_convert)"]
    Convert --> Reindex{reindex_time_axis<br/>== 'yes'?}
    Reindex -->|Yes| DST[Dedupe DST fall-back duplicates;<br/>insert NaN for spring-forward gap]
    Reindex -->|No| TagAttr
    DST --> TagAttr[Assign 'timezone' attr per variable]
    TagAttr --> UpdateCtx[update_context]
    UpdateCtx --> End([Output: time-converted data])

    click Start "https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/convert_to_local_time.py#L88" "execute"
    click GridFunc "https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/convert_to_local_time.py#L212" "_convert_to_local_time_gridded"
    click HDPFunc "https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/convert_to_local_time.py#L269" "_convert_to_local_time_hdp"
    click Convert "https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/convert_to_local_time.py#L320" "_find_timezone_and_convert"
    click UpdateCtx "https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/convert_to_local_time.py#L186" "update_context"
```

## Parameters

The processor takes a **dict**:

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `convert` | `"yes"` / `"no"` | `"no"` | Master switch. If `"no"` (or omitted), the processor is a no-op. |
| `reindex_time_axis` | `"yes"` / `"no"` | `"no"` | After conversion, deduplicate DST collisions and insert NaN for the missing spring-forward hour. |

> Earlier docs described `timezone` and `lon_based` keys. Those are **not implemented** — the timezone is always inferred from the data's central lat/lon via `timezonefinder`.

### Frequency requirement

`_convert_to_local_time_gridded` only converts hourly data. If the dataset's `frequency` attribute (or inferred timestep) is not `"1hr"`, it logs a warning and returns the data unchanged.

## Example

```python
from climakitae.new_core.user_interface import ClimateData

data = (ClimateData()
    .catalog("cadcat")
    .activity_id("WRF")
    .institution_id("UCLA")
    .variable("t2")
    .table_id("1hr")
    .grid_label("d03")
    .processes({
        "clip": "Los Angeles",
        "convert_to_local_time": {"convert": "yes", "reindex_time_axis": "yes"},
    })
    .get())
```

## Code References

| Method | Lines | Purpose |
|--------|-------|---------|
| `__init__` | [66–86](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/convert_to_local_time.py#L66) | Read `convert` and `reindex_time_axis` from value dict |
| `execute` | [88–184](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/convert_to_local_time.py#L88) | Catalog-routed dispatch over result types |
| `update_context` | [186–206](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/convert_to_local_time.py#L186) | Record applied timezone in `new_attrs` |
| `_convert_to_local_time_gridded` | [212–267](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/convert_to_local_time.py#L212) | Central-pixel lat/lon → conversion (hourly only) |
| `_convert_to_local_time_hdp` | [269–293](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/convert_to_local_time.py#L269) | First-station lat/lon → conversion |
| `_contains_leap_days` | [295–318](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/convert_to_local_time.py#L295) | Detect Feb 29 entries |
| `_find_timezone_and_convert` | [320–](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/convert_to_local_time.py#L320) | `timezonefinder` lookup + `tz_convert`, optional reindex, set `timezone` attr |

## See also

- [Processor index](index.md)
- [`climakitae/new_core/processors/convert_to_local_time.py`](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/convert_to_local_time.py)
