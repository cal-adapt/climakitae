"""
Initialize the processors, ensuring they get registered.
"""

from .concatenate import Concat
from .filter_unbiased_models import FilterUnbiasedModels
from .update_attributes import UpdateAttributes

__all__ = [
    "Concat",
    "FilterUnbiasedModels",
    "UpdateAttributes",
]
