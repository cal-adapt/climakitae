# A class for holding the app explore options

import panel as pn
import param

from .tmy import AverageMeteorologicalYear, _amy_visualize
from .threshold_panel import ThresholdDataParams, _thresholds_visualize
from .warming_levels import WarmingLevels, _display_warming_levels


class AppExplore(object):
    """
    A class for holding the following app explore options:
        app.explore.amy()
        app.explore.thresholds()
        app.explore.warming_levels()
    """

    def __init__(self, selections, location, _cat):
        self.selections = selections
        self.location = location
        self._cat = _cat

    def __repr__(self):
        """Print a string description of the available analysis method for this class. """
        return ("Choose one of these interactive panels to explore different aspects of the data:\n\n"
                "app.explore.warming_levels(): Learn about global warming levels and explore regional responses.\n"
                "app.explore.thresholds(): Explore how frequencies of extreme events will change.\n"
                "app.explore.amy(): Produce an hourly time series for one year capturing mean climate conditions."
        )

    def amy(self):
        """Display Average Meteorological Year panel. """
        tmy_ob = AverageMeteorologicalYear(
            selections = self.selections,
            location = self.location,
            catalog = self._cat
        )
        return _amy_visualize(
            tmy_ob = tmy_ob,
            selections = self.selections,
            location = self.location
        )

    def thresholds(self, option = 1):
        """Display Thresholds panel. """
        thresh_data = ThresholdDataParams(
            selections = self.selections,
            location = self.location,
            cat = self._cat
        )
        return _thresholds_visualize(
            thresh_data = thresh_data,
            selections = self.selections,
            location = self.location,
            option = option
        )

    def warming_levels(self):
        """Display Warming Levels panel. """
        warming_data = WarmingLevels(
            selections = self.selections,
            location = self.location,
            cat = self._cat
        )
        return _display_warming_levels(
            warming_data,
            self.selections,
            self.location
        )
