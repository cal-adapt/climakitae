"""
Contains source code for the AppExplore object, used to access panel GUIs for exploring several climatological topics of interest: 
1. Average meteorological year
2. Thresholds 
3. Global warming levels 
See the AppExplore object documentation for more information. 
"""

import panel as pn
import param
from .meteo_yr import _AverageMeteorologicalYear, _amy_visualize
from .threshold_panel import _ThresholdDataParams, _thresholds_visualize
from .warming_levels import _WarmingLevels, _display_warming_levels


class _AppExplore(object):
    """Explore the data using interactive GUIs.
    Only functional in a jupyter notebook environment.
    """

    def __init__(self, selections, _cat, var_config):
        """Constructor

        Parameters
        ----------
        selections: _DataSelector
            Data settings (variable, unit, timescale, etc)
        _cat: intake_esm.core.esm_datastore
            AE data catalog
        var_config: pd.DataFrame
            Variable descriptions, units, etc in table format

        """
        self.selections = selections
        self._cat = _cat
        self.var_config = var_config

    def __repr__(self):
        """Print a string description of the available analysis method for this class."""
        return (
            "Choose one of these interactive panels to explore different aspects of the data:\n\n"
            "app.explore.warming_levels(): Learn about global warming levels and explore regional responses.\n"
            "app.explore.thresholds(): Explore how frequencies of extreme events will change.\n"
            "app.explore.amy(): Produce an hourly time series for one year capturing mean climate conditions."
        )

    def amy(self):
        """Display Average Meteorological Year panel."""
        global tmy_ob
        tmy_ob = _AverageMeteorologicalYear(
            selections=self.selections,
            cat=self._cat,
            var_config=self.var_config,
        )
        return _amy_visualize(
            tmy_ob=tmy_ob,
            selections=self.selections,
        )

    def amy_selections(self):
        """Return the Average Meteorological Year panel object such that the user
        can pull information from the AMY panel or directly modify the values."""
        if (
            "tmy_ob" in globals()
        ):  # First, check if amy() has been run and the object has been created
            return tmy_ob
        else:
            print("Please first initialize the AMY panel by calling app.explore.amy()")
            return None

    def thresholds(self, option=1):
        """Display Thresholds panel."""
        thresh_data = _ThresholdDataParams(
            selections=self.selections, cat=self._cat
        )
        return _thresholds_visualize(
            thresh_data=thresh_data,
            selections=self.selections,
            option=option,
        )

    def warming_levels(self):
        """Display Warming Levels panel."""
        warming_data = _WarmingLevels(
            selections=self.selections,
            cat=self._cat,
            var_config=self.var_config,
        )
        return _display_warming_levels(
            warming_data,
            self.selections,
        )
