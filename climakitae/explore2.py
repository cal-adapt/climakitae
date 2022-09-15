# A class for holding the app explore options

import panel as pn

from .threshold_tools import ExceedanceParams, _exceedance_visualize

class AppExplore(object):
    """
    A class for holding the following app explore options:
        app.explore2.TMY()
        app.explore2.thresholds()
    """

    def __init__(self, selections, location, _cat):
        self.selections = selections
        self.location = location,
        self._cat = _cat

    def TMY(self):
        return pn.Card(title = "Typical Meteorological Year", collapsible = False)

    def thresholds(self, da, option=1):
        exc_choices = ExceedanceParams(da) # initialize an instance of the Param class for this dataarray
        return _exceedance_visualize(exc_choices, option) # display the holoviz panel