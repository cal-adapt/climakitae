"""Initialize the processors, ensuring they get registered."""

from .bias_adjust_model_to_station import BiasCorrectStationData
from .concatenate import Concat
from .convert_units import ConvertUnits
from .clip import Clip
from .filter_unadjusted_models import FilterUnAdjustedModels
from .time_slice import TimeSlice
from .update_attributes import UpdateAttributes
from .warming_level import WarmingLevel

__all__ = [
    "Concat",
    "ConvertUnits",
    "Clip",
    "FilterUnAdjustedModels",
    "BiasCorrectStationData",
    "TimeSlice",
    "UpdateAttributes",
    "WarmingLevel",
]
