# Core Data Interface (Detailed)

The legacy data interface module providing function-based API for climate data access.

## Overview

`climakitae.core.data_interface` is the main entry point for the legacy interface. It provides:
- **DataParameters class** — Configuration object for data queries
- **get_data() function** — Execute data queries with validation

!!! warning
    This is the **legacy interface**. For new code, use `climakitae.new_core.user_interface.ClimateData` instead.

## DataParameters Class

::: climakitae.core.data_interface.DataParameters
    options:
      docstring_style: numpy
      show_source: true
      members_order: source
      merge_init_into_class: true

## Get Data Function

::: climakitae.core.data_interface.get_data
    options:
      docstring_style: numpy
      show_source: true

## Migration Note

For new code, use the modern `climakitae.new_core` interface. See the [migration guide](../migration/legacy-to-climate-data.md) for detailed upgrade instructions.

### Quick Example

**Legacy (old):**
```python
from climakitae.core.data_interface import get_data, DataParameters

params = DataParameters()
params.variable = "Maximum air temperature at 2m"
params.time_slice = (2015, 2045)            # year-range tuple
params.downscaling_method = "Statistical"    # \u2248 LOCA2
params.resolution = "3 km"                   # \u2248 grid_label d03
params.timescale = "monthly"                 # \u2248 table_id "mon"
data = get_data(params)
```

**Modern (new):**
```python
from climakitae.new_core.user_interface import ClimateData

data = (ClimateData()
    .variable("tasmax")
    .processes({"time_slice": (2015, 2045)})
    .get())
```
