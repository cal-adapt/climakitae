"""
Initialize the processors, ensuring they get registered.
"""

from .clip import Clip
from .concatenate import Concat
from .convert_units import ConvertUnits
from .export import Export
from .filter_unbiased_models import FilterUnAdjustedModels
from .localize import Localize
from .metric_calc import MetricCalc
from .time_slice import TimeSlice
from .update_attributes import UpdateAttributes
from .warming_level import WarmingLevel

__all__ = [
    "Clip",
    "Concat",
    "ConvertUnits",
    "Export",
    "FilterUnAdjustedModels",
    "Localize",
    "MetricCalc",
    "TimeSlice",
    "UpdateAttributes",
    "WarmingLevel",
]
