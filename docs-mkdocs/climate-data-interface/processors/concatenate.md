# Processor: Concat

**Registry key:** `concat` &nbsp;|&nbsp; **Priority:** 50 &nbsp;|&nbsp; **Category:** Data Assembly

!!! note "Registry key vs filename"
    The processor's source file is `concatenate.py` and its class is `Concat`, but it
    is registered under the key **`"concat"`**. Always use `"concat"` (not
    `"concatenate"`) in `.processes({...})`.

Merge multiple climate datasets returned from a single catalog query by concatenating them along a new dimension (default: `sim`). This is the standard way to assemble a multi-model ensemble or to combine `historical` + `ssp*` time series into a single contiguous record.

The processor is invoked automatically when a query produces multiple datasets (e.g. multiple `source_id` values) and the user includes `"concat"` in `.processes({...})`. It dispatches to `_execute_gridded_concat` or `_execute_hdp_concat` based on the catalog type detected in the processing context.

## Algorithm

```mermaid
flowchart TD
    Start([Input: dict / list of Datasets]) --> CheckType{Single<br/>Dataset?}
    CheckType -->|Yes| Pass[Return unchanged]
    CheckType -->|No| Route{catalog == 'hdp'?}
    Route -->|Yes| HDP[_execute_hdp_concat<br/>concat along station]
    Route -->|No| Gridded[_execute_gridded_concat<br/>concat along sim with source_id labels]
    HDP --> Update[Update context attrs]
    Gridded --> Update
    Update --> End([Output: single Dataset])
```

## Parameter shape

The processor takes a **single string**: the name of the new dimension. The default is `"sim"`, which is what almost every multi-model workflow wants.

```python
.processes({"concat": "sim"})
```

| Field | Type | Description |
|-------|------|-------------|
| `value` | `str` | Name of the new dimension. Defaults to `"sim"` if a non-string is passed. |

## Examples

### Multi-model ensemble (gridded catalog)

```python
from climakitae.new_core.user_interface import ClimateData

ensemble = (ClimateData()
    .catalog("cadcat")
    .activity_id("LOCA2")
    .variable("tasmax")
    .experiment_id(["historical", "ssp370"])
    .table_id("day")
    .grid_label("d03")
    .processes({
        "time_slice": ("2000-01-01", "2050-12-31"),
        "clip": "Los Angeles",
        "concat": "sim",
    })
    .get())

# ensemble has a 'sim' dimension labeled by source_id
print(ensemble.sim.values)
```

### HDP station catalog

For the `hdp` catalog, `concat` produces a station-dimension stack:

```python
stations = (ClimateData()
    .catalog("hdp")
    .network_id("hadisd")
    .processes({"concat": "station"})
    .get())
```

## Behavior notes

- Input must be a collection (dict or iterable) of datasets/dataarrays. A single Dataset/DataArray is returned unchanged.
- For the gridded path, each input dataset is expected to carry a `source_id` attribute, which becomes its label along the new dimension.
- The HDP path uses a simpler concatenation that does not perform per-source attribute extraction.
- Runs at priority **50** — after early refinement (filter, leap days, units, warming level) and before spatial clipping and time slicing, so the merged ensemble flows through the rest of the pipeline as a single object.

## See also

- [Processor index](index.md)
- [`climakitae/new_core/processors/concatenate.py`](https://github.com/cal-adapt/climakitae/blob/main/climakitae/new_core/processors/concatenate.py)
