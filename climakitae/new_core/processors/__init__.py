"""
Initialize the processors, ensuring they get registered.
"""

from .concatenate import Concat
from .filter_unadjusted_models import FilterUnAdjustedModels
from .update_attributes import UpdateAttributes

__all__ = [
    "Concat",
    "FilterUnAdjustedModels",
    "UpdateAttributes",
]
