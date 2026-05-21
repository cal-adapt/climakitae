# Derived Variables Module

Registry and utilities for climate-derived variable computation.

## Overview

The `climakitae.new_core.derived_variables` module provides:

- **Registry** — Centralized registry of available derived variables
- **Utilities** — Helper functions for variable transformation
- **Built-in derivations** — Pre-configured derived variables:  
    - Humidity indices (relative humidity, dew point)  
    - Temperature indices (effective temperature, heat index)  
    - Wind computations (wind speed, direction)  
    - Climate indices (growing degree days, etc.)  

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
