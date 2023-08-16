"""
Contains source code for the Explore object, used to access panel GUIs for exploring several climatological topics of interest: 
1. Average meteorological year
2. Thresholds 
3. Global warming levels 
See the Explore object documentation for more information. 
"""

from climakitae.explore.thresholds import (
    ThresholdParameters,
    thresholds_visualize,
)
from climakitae.explore.warming import (
    WarmingLevelParameters,
    warming_levels_visualize,
)
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


class WarmingLevels(WarmingLevelParameters):
    """Display Warming Levels panel."""

    def show(self):
        return warming_levels_visualize(self)


def amy():
    return AverageMetYear()

def thresholds():
    return Thresholds()

def warming_levels():
    return WarmingLevels()
