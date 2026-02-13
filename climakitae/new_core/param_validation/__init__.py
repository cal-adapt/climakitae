"""Parameter validation module for the new core functionality.

This subpackage provides a validation framework for climate data queries.
It includes:

- Abstract base classes for validators
- Catalog-specific validators (cadcat, renewables, etc.)
- Processor-specific parameter validators
- Utility functions for validation

The validation system uses a registry pattern, allowing validators to be
registered with decorators and discovered automatically by the DatasetFactory.

Key Components
--------------
- ``ParameterValidator``: Abstract base class for catalog validators
- ``DataValidator``: Validator for the main cadcat catalog
- ``RenewablesValidator``: Validator for renewable energy data

See Also
--------
climakitae.new_core.dataset_factory : Uses validators to create datasets
climakitae.new_core.processors : Processing steps with their validators
"""

from .bias_adjust_model_to_station_param_validator import (
    validate_bias_correction_station_data_param,
)
from .cadcat_param_validator import DataValidator
from .clip_param_validator import validate_clip_param
from .concat_param_validator import validate_concat_param
from .convert_to_local_time_validator import validate_convert_to_local_time_param
from .convert_units_param_validator import validate_convert_units_param
from .drop_leap_days_param_validator import validate_drop_leap_days_param
from .export_param_validator import validate_export_param
from .filter_unadjusted_models_param_validator import (
    validate_filter_unadjusted_models_param,
)
from .hdp_param_validator import HDPValidator
from .metric_calc_param_validator import validate_metric_calc_param
from .renewables_param_validator import RenewablesValidator
from .time_slice_param_validator import validate_time_slice_param
from .update_attributes_param_validator import validate_update_attributes_param
from .warming_param_validator import validate_warming_level_param

__all__ = [
    "DataValidator",
    "RenewablesValidator",
    "validate_bias_correction_station_data_param",
    "validate_clip_param",
    "validate_concat_param",
    "validate_convert_to_local_time_param",
    "validate_convert_units_param",
    "validate_drop_leap_days_param",
    "validate_export_param",
    "validate_filter_unadjusted_models_param",
    "validate_metric_calc_param",
    "validate_time_slice_param",
    "validate_update_attributes_param",
    "validate_warming_level_param",
]
