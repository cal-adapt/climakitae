"""
Initialize the processors, ensuring they get registered.
"""

from .clip import Clip
from .concatenate import Concat
from .convert_units import ConvertUnits
from .filter_unbiased_models import FilterUnbiasedModels
from .localize import Localize
from .time_slice import TimeSlice
from .update_attributes import UpdateAttributes
