"""Initialize the processors, ensuring they get registered."""

from .concatenate import Concat
from .clip import Clip
from .filter_unadjusted_models import FilterUnAdjustedModels
from .time_slice import TimeSlice
from .update_attributes import UpdateAttributes
from .warming_level import WarmingLevel

__all__ = [
    "Concat",
    "Clip",
    "FilterUnAdjustedModels",
    "TimeSlice",
    "UpdateAttributes",
    "WarmingLevel",
]
