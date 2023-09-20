"""
Contains source code for the explore tools, used to access panel GUIs for exploring several climatological topics of interest: 
1. Average meteorological year
2. Thresholds 
3. Global warming levels 
"""

from climakitae.explore.thresholds import (
    ThresholdParameters,
    thresholds_visualize,
)
from climakitae.explore.warming import WarmingLevels
from climakitae.explore.amy import (
    AverageMetYearParameters,
    amy_visualize,
)


class AverageMetYear(AverageMetYearParameters):
    """Display AMY panel."""

    def show(self):
        return amy_visualize(self)


class Thresholds(ThresholdParameters):
    """Display Thresholds panel."""

    def show(self):
        return thresholds_visualize(
            self,
            option=self.option,
        )


def amy():
    return AverageMetYear()


def thresholds(option=1):
    return Thresholds(option=option)


def warming_levels():
    return WarmingLevels()
