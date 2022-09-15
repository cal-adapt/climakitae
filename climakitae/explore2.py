# A class for holding the app explore options

import panel as pn
import param
from .tmy import _tmy_visualize


class TMYParams(param.Parameterized):
    """
    An object that holds the "Data Options" parameters for the
    explore.tmy panel.
    """
    def __init__(self, *args, **params):
        super().__init__(*args, **params)

        # Selectors defaults
        self.selections.append_historical = False
        self.selections.area_average = True
        self.selections.resolution = "45 km"
        self.selections.scenario = ["SSP 3-7.0 -- Business as Usual"]
        self.selections.time_slice = (1980,2100)
        self.selections.timescale = "hourly"
        self.selections.variable = "Air Temperature at 2m"

        # Location defaults
        self.location.area_subset = 'states'
        self.location.cached_area = 'CA'

    variable2 = param.ObjectSelector(default="Air Temperature at 2m",
        objects=["Air Temperature at 2m", "Surface Pressure"]
    )

    cached_area2 = param.ObjectSelector(default="CA",
        objects=["CA"]
    )

    area_subset2 = param.ObjectSelector(
        default="states",
        objects=["states", "CA counties"],
    )

    @param.depends("variable2", watch=True)
    def _update_variable(self):
        """Update variable in selections object to reflect variable chosen in panel"""
        self.selections.variable = self.variable2

    @param.depends("area_subset2", watch=True)
    def _update_cached_area(self):
        """
        Makes the dropdown options for 'cached area' reflect the type of area subsetting
        selected in 'area_subset' (currently state, county, or watershed boundaries).
        """
        if self.area_subset2 == "CA counties":
            # setting this to the dict works for initializing, but not updating an objects list:
            self.param["cached_area2"].objects = ["Santa Clara County", "Los Angeles County"]
            self.cached_area2 = "Santa Clara County"
        elif self.area_subset2 == "states":
            self.param["cached_area2"].objects = ["CA"]
            self.cached_area2 = "CA"

    @param.depends("area_subset2","cached_area2",watch=True)
    def _updated_location(self):
        """Update locations object to reflect location chosen in panel"""
        self.location.area_subset = self.area_subset2
        self.location.cached_area = self.cached_area2


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


    def tmy():
        # tmy_data = TMYParams(selections=self.selections, location=self.location)
        #
        # data_options = pn.Card(
        #     pn.Row(
        #         pn.Column(
        #             pn.widgets.Select.from_param(tmy_data.param.variable2, name="Data variable"),
        #             # selections.param.variable,
        #             pn.widgets.StaticText.from_param(selections.param.variable_description),
        #             pn.widgets.StaticText(name="", value="Variable Units"),
        #             pn.widgets.RadioButtonGroup.from_param(selections.param.units),
        #             width=230),
        #     )
        # , title="Data Options - Absolute TMY", collapsible=False, width=460, height=515
        # )
        #
        # mthd_box = pn.Card(
        #     "Here is where I describe the methods of the TMY.",
        #     title="Methodology", collapsible=False
        # )
        #
        # return pn.Row(
        #     pn.Column(
        #         (data_options),
        #         mthd_box
        #     )
        # )

        return _tmy_visualize()
