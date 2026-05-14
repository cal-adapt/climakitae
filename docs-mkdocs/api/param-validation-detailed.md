# Parameter Validation (Detailed)

Catalog-specific parameter validators for climate data queries.

## Overview

The `climakitae.new_core.param_validation` module provides catalog-specific validators that ensure climate data query parameters are valid before execution. Each validator:  

- Validates required parameters for its catalog
- Checks parameter compatibility
- Suggests corrections for invalid parameters
- Handles edge cases (e.g., models that don't reach warming levels)

## Base Validator

All validators inherit from the abstract base class:

::: climakitae.new_core.param_validation.abc_param_validation
    options:
      docstring_style: numpy
      show_source: true

## Catalog Validators

::: climakitae.new_core.param_validation.cadcat_param_validator
    options:
      docstring_style: numpy
      show_source: true

::: climakitae.new_core.param_validation.time_slice_param_validator
    options:
      docstring_style: numpy
      show_source: true

::: climakitae.new_core.param_validation.warming_param_validator
    options:
      docstring_style: numpy
      show_source: true

::: climakitae.new_core.param_validation.clip_param_validator
    options:
      docstring_style: numpy
      show_source: true

::: climakitae.new_core.param_validation.export_param_validator
    options:
      docstring_style: numpy
      show_source: true

## Processor Validators

::: climakitae.new_core.param_validation.concat_param_validator
    options:
      docstring_style: numpy
      show_source: true

::: climakitae.new_core.param_validation.metric_calc_param_validator
    options:
      docstring_style: numpy
      show_source: true

::: climakitae.new_core.param_validation.convert_units_param_validator
    options:
      docstring_style: numpy
      show_source: true

::: climakitae.new_core.param_validation.bias_adjust_model_to_station_param_validator
    options:
      docstring_style: numpy
      show_source: true

::: climakitae.new_core.param_validation.filter_unadjusted_models_param_validator
    options:
      docstring_style: numpy
      show_source: true

::: climakitae.new_core.param_validation.drop_leap_days_param_validator
    options:
      docstring_style: numpy
      show_source: true

::: climakitae.new_core.param_validation.convert_to_local_time_param_validator
    options:
      docstring_style: numpy
      show_source: true

## Alternative Catalog Validators

::: climakitae.new_core.param_validation.renewables_param_validator
    options:
      docstring_style: numpy
      show_source: true

::: climakitae.new_core.param_validation.hdp_param_validator
    options:
      docstring_style: numpy
      show_source: true

## Utilities

::: climakitae.new_core.param_validation.param_validation_tools
    options:
      docstring_style: numpy
      show_source: true
