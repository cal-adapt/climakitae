import abc

"""
Contains source code for the Explore object, used to access panel GUIs for exploring several climatological topics of interest: 
1. Average meteorological year
2. Thresholds 
3. Global warming levels 
See the Explore object documentation for more information. 
"""

from climakitae.explore.explore_thresholds import (
    ThresholdParameters,
    thresholds_visualize,
)
from climakitae.explore.explore_warming import (
    WarmingLevelParameters,
    warming_levels_visualize,
)
from climakitae.explore.explore_amy import (
    AverageMetYearParameters,
    amy_visualize,
)


class AbstractExplore(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def show(self):
        pass


class AverageMetYear(AbstractExplore):
    def __init__(self):
        self.selections = AverageMetYearParameters()

    """Display AMY panel."""

    def show(self):
        return amy_visualize(self.selections)


class Thresholds(AbstractExplore):
    def __init__(self, option=1):
        self.selections = ThresholdParameters()
        self.option = option

    """Display Thresholds panel."""

    def show(self):
        return thresholds_visualize(
            self.selections,
            option=self.option,
        )


class WarmingLevels(AbstractExplore):
    def __init__(self):
        self.selections = WarmingLevelParameters()

    """Display Warming Levels panel."""

    def show(self):
        return warming_levels_visualize(self.selections)


class Explore:
    """Explore the data using interactive GUIs.
    Only functional in a jupyter notebook environment.
    """

    def __repr__(self):
        """Print a string description of the available analysis method for this class."""
        return (
            "Choose one of these interactive panels to explore different aspects of the data:\n\n"
            "ck.explore.WarmingLevels(): Learn about global warming levels and explore regional responses.\n"
            "ck.explore.Thresholds(): Explore how frequencies of extreme events will change.\n"
            "ck.explore.AMY(): Produce an hourly time series for one year capturing mean climate conditions."
        )

    def amy():
        return AverageMetYear()

    def thresholds():
        return Thresholds()

    def warming_levels():
        return WarmingLevels()
