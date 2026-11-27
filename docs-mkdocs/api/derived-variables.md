# Derived Variables Module

Registry and utilities for climate-derived variable computation.

## Overview

The `climakitae.new_core.derived_variables` module provides:

- **Registry** — Centralized registry of available derived variables
- **Utilities** — Helper functions for variable transformation
- **Built-in derivations** — Pre-configured derived variables:  
    - Humidity: `relative_humidity_2m`, `dew_point_2m`, `specific_humidity_2m`  
    - Temperature: `heat_index`, `wind_chill`, `diurnal_temperature_range_loca`, `diurnal_temperature_range_wrf`, `HDD_loca`, `CDD_loca`, `HDD_wrf`, `CDD_wrf`  
    - Wind: `wind_speed_10m`, `wind_direction_10m`  
    - Climate indices: `fosberg_fire_weather_index`  

Use `ClimateData().show_derived_variables()` for the current list, including
user-registered variables.

## When derived variables are calculated

Validation replaces a requested derived variable with its catalog
dependencies, and retrieval opens only those fields. The internal
[`derive_variables`](../climate-data-interface/processors/derive_variables.md)
processor (priority 3) then calls the registered function after `clip`
(priority 2). Thus, each function receives only the clipped grid, in native
catalog units, before `warming_level` and `concat`.

## Registry

::: climakitae.new_core.derived_variables.registry
    options:
      docstring_style: numpy
      show_source: true

## Utilities

::: climakitae.new_core.derived_variables.utils
    options:
      docstring_style: numpy
      show_source: true

## Built-in Humidity Derivations

::: climakitae.new_core.derived_variables.builtin.humidity
    options:
      docstring_style: numpy
      show_source: true

## Built-in Temperature Derivations

::: climakitae.new_core.derived_variables.builtin.temperature
    options:
      docstring_style: numpy
      show_source: true

## Built-in Wind Derivations

::: climakitae.new_core.derived_variables.builtin.wind
    options:
      docstring_style: numpy
      show_source: true

## Built-in Climate Indices

::: climakitae.new_core.derived_variables.builtin.indices
    options:
      docstring_style: numpy
      show_source: true
