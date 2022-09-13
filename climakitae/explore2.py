# A class for holding the app explore options

import panel as pn

class AppExplore(object):
    """
    A class for holding the following app explore options:
        app.explore2.TMY()
        app.explore2.thresholds()
    """

    def __init__(self):
        pass

    def TMY(self):
        return pn.Card(title = "Typical Meteorological Year", collapsible = False)

    def thresholds(self):
        return pn.Card(title = "Explore extreme events", collapsible = False)