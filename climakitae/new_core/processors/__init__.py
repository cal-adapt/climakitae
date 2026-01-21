"""Initialize the processors, ensuring they get registered.

This module imports all DataProcessor implementations to trigger their
registration with the processor registry via the @register_processor decorator.
The registry enables dynamic processor discovery and priority-based execution
in the data processing pipeline.
"""

from .bias_adjust_model_to_station import BiasAdjustModelToStation
from .clip import Clip
from .concatenate import Concat
from .convert_units import ConvertUnits
from .drop_leap_days import DropLeapDays
from .export import Export
from .filter_unadjusted_models import FilterUnAdjustedModels
from .metric_calc import MetricCalc
from .time_slice import TimeSlice
from .update_attributes import UpdateAttributes
from .warming_level import WarmingLevel

__all__ = [
    "Concat",
    "ConvertUnits",
    "Clip",
    "DropLeapDays",
    "Export",
    "FilterUnAdjustedModels",
    "BiasAdjustModelToStation",
    "TimeSlice",
    "UpdateAttributes",
    "WarmingLevel",
    "MetricCalc",
]
