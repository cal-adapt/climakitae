# A class for holding the app explore options

import panel as pn
import param
from .tmy import AverageMeteorologicalYear, _amy_visualize

#-----------------------------------------------------------------------

class AppExplore(object):
    """
    A class for holding the following app explore options:
        app.explore2.amy()
        app.explore2.thresholds()
    """

    def __init__(self, selections, location, _cat):
        self.selections = selections
        self.location = location
        self._cat = _cat

    def amy(self):
        tmy_ob = AverageMeteorologicalYear(selections=self.selections, location=self.location, catalog=self._cat)
        return _amy_visualize(tmy_ob=tmy_ob, selections=self.selections, location=self.location)
