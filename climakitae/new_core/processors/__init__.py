"""Initialize the processors, ensuring they get registered."""

from .bias_adjust_model_to_station import BiasAdjustModelToStation
from .clip import Clip
from .concatenate import Concat
from .convert_units import ConvertUnits
from .clip import Clip
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
    "Export",
    "FilterUnAdjustedModels",
    "BiasAdjustModelToStation",
    "TimeSlice",
    "UpdateAttributes",
    "WarmingLevel",
    "MetricCalc",
]
