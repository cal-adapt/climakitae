# A class for holding the app explore options

import panel as pn
import param
from .tmy import TypicalMeteorologicalYear, _tmy_visualize

#-----------------------------------------------------------------------

class AppExplore(object):
    """
    A class for holding the following app explore options:
        app.explore2.tmy()
        app.explore2.thresholds()
    """

    def __init__(self, selections, location, _cat):
        self.selections = selections
        self.location = location,
        self._cat = _cat

    def tmy(selections, location):
        tmy_ob = TypicalMeteorologicalYear(selections=selections, location=location)
        return _tmy_visualize(tmy_ob.selections, tmy_ob.location)
