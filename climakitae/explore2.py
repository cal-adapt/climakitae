# A class for holding the app explore options

import panel as pn

from .threshold_tools import ThresholdDataParams, ExceedanceParams, _exceedance_visualize

class AppExplore(object):
    """
    A class for holding the following app explore options:
        app.explore2.TMY()
        app.explore2.thresholds()
    """

    def __init__(self, selections, location, _cat):
        self.selections2 = selections
        self.location2 = location,
        self._cat2 = _cat

    def TMY(self):
        return pn.Card(title = "Typical Meteorological Year", collapsible = False)

    # def thresholds(self, da, option=1):
    def thresholds(self, option=1):
        thresh_data = ThresholdDataParams(selections=self.selections2, location=self.location2)

        data_options_card = pn.Card(
            pn.Row(
                pn.Column(
                    pn.widgets.Select.from_param(thresh_data.param.variable2, name="Data variable"),
                    pn.widgets.StaticText.from_param(selections.param.variable_description),
                    width = 230),
                pn.Column(
                    pn.widgets.Select.from_param(thresh_data.param.area_subset2, name="Area subset"),
                    pn.widgets.Select.from_param(thresh_data.param.cached_area2, name="Cached area"),
                    location.view,
                    width = 230)
                )
        , title="Data Options", collapsible=False, width=460, height=515
        )

        description_box = pn.Card(
            "Description text",
            title = "About this tool", collapsible = False
        )

        # exc_choices = ExceedanceParams(da) # initialize an instance of the Param class for this dataarray
        # plot_panel = _exceedance_visualize(exc_choices, option) # display the holoviz panel

        return pn.Column(
            pn.Row(
                data_options_card, description_box
            ),
            # plot_panel
        )