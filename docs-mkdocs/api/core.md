# Legacy Core Module

The legacy `climakitae.core` module provides the original function-based interface for climate data access. This interface is maintained for backward compatibility; new code should use `climakitae.new_core.user_interface.ClimateData` instead.

## Overview

The legacy core module includes:
- **Data interface** — Main API for querying climate data
- **Boundaries** — Geographic boundary management
- **Data loading** — Data retrieval and parsing
- **Data export** — Save data to various formats
- **Constants** — Shared configuration constants
- **Paths** — File path configuration

## Main Data Interface

::: climakitae.core.data_interface.DataParameters
    options:
      docstring_style: numpy
      show_source: true

::: climakitae.core.data_interface.get_data
    options:
      docstring_style: numpy
      show_source: true

## Boundaries

::: climakitae.core.boundaries.Boundaries
    options:
      docstring_style: numpy
      show_source: true

## Data Loading

::: climakitae.core.data_load
    options:
      docstring_style: numpy
      show_source: true

## Data Export

::: climakitae.core.data_export
    options:
      docstring_style: numpy
      show_source: true

## Constants

::: climakitae.core.constants
    options:
      docstring_style: numpy
      show_source: true

## Paths

::: climakitae.core.paths
    options:
      docstring_style: numpy
      show_source: true

## Migration Note

For new code, use the modern `climakitae.new_core` interface:

```python
from climakitae.new_core.user_interface import ClimateData

data = (ClimateData()
    .catalog("cadcat")
    .variable("tasmax")
    .get())
```
