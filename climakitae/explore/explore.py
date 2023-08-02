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
    display_warming_levels,
)
from climakitae.explore.explore_amy import (
    AverageMetYearParameters,
    amy_visualize,
)


class Explore:
    """Explore the data using interactive GUIs.
    Only functional in a jupyter notebook environment.
    """

    # def __init__(self, selections, _cat, var_config):
    #     """Constructor

    #     Parameters
    #     ----------
    #     selections: _DataSelector
    #         Data settings (variable, unit, timescale, etc)
    #     _cat: intake_esm.core.esm_datastore
    #         AE data catalog
    #     var_config: pd.DataFrame
    #         Variable descriptions, units, etc in table format

    #     """
    #     self.selections = selections
    #     self._cat = _cat
    #     self.var_config = var_config

    def __repr__(self):
        """Print a string description of the available analysis method for this class."""
        return (
            "Choose one of these interactive panels to explore different aspects of the data:\n\n"
            "app.explore.warming_levels(): Learn about global warming levels and explore regional responses.\n"
            "app.explore.thresholds(): Explore how frequencies of extreme events will change.\n"
            "app.explore.amy(): Produce an hourly time series for one year capturing mean climate conditions."
        )

    class AMY:
        def __init__(self):
            self.data_parameters = AverageMetYearParameters()

        """Display AMY panel."""

        def show(self):
            return amy_visualize(self.data_parameters)

    class Thresholds(option=1):
        def __init__(self, option):
            self.data_parameters = ThresholdParameters()
            self.option = option

        """Display Thresholds panel."""

        def show(self):
            return thresholds_visualize(
                self.data_parameters,
                option=self.option,
            )

    class WarmingLevels:
        def __init__(self):
            self.data_parameters = WarmingLevelParameters()

        """Display Warming Levels panel."""

        def show(self):
            return display_warming_levels(self.data_parameters)
