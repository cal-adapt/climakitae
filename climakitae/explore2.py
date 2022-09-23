# A class for holding the app explore options

import panel as pn
import param

from .threshold_tools import ThresholdDataParams, _exceedance_visualize

_thresholds_tool_description = "Select a variable of interest, variable units, and region of interest to the left. Then, use the dropdowns below to define the extreme event you are interested in. After clicking 'Reload Plot', the plot on the lower right will show event frequencies across different simulations. You can also explore how these frequencies change under different global emissions scenarios. Save a plot to come back to later by putting your cursor over the lower right and clicking the save icon."


class AppExplore(object):
    """
    A class for holding the following app explore options:
        app.explore2.TMY()
        app.explore2.thresholds()
    """
    
    def __init__(self, selections, location, _cat):
        self.selections = selections
        self.location = location
        self._cat = _cat

    def TMY(self):
        return pn.Card(title = "Typical Meteorological Year", collapsible = False)

    def thresholds(self, option=1):
        thresh_data = ThresholdDataParams(selections=self.selections, location=self.location, _cat = self._cat)

        _first_row_height = 300

        data_options_card = pn.Card(
            pn.Row(
                pn.Column(
                    pn.widgets.Select.from_param(thresh_data.param.variable2, name="Data variable"),
                    pn.widgets.RadioButtonGroup.from_param(thresh_data.param.units2),
                    pn.widgets.StaticText.from_param(self.selections.param.variable_description),
                    width = 230
                    ),
                pn.Column(
                    pn.widgets.Select.from_param(thresh_data.param.area_subset2, name="Area subset"),
                    pn.widgets.Select.from_param(thresh_data.param.cached_area2, name="Cached area"),
                    pn.widgets.Button.from_param(thresh_data.param.reload_data, button_type="primary", width=150, height=30),
                    width = 230
                    ),
                pn.Column(
                    self.location.view,
                    width = 180
                    ),
                ),
        title="Data Options", collapsible=False, width=700, height=_first_row_height
        )

        description_box = pn.Card(
            _thresholds_tool_description,
            title = "About this tool", collapsible = False,
            # width=400, 
            height = _first_row_height
        )

        plot_panel = _exceedance_visualize(thresh_data, option) # display the holoviz panel

        return pn.Column(
            pn.Row(
                data_options_card, description_box
            ),
            plot_panel
        )