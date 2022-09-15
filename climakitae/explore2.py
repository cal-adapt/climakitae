# A class for holding the app explore options

import panel as pn
import param

#from .threshold_tools import ThresholdDataParams, ExceedanceParams, _exceedance_visualize

class ThresholdDataParams(param.Parameterized):
    """
    An object that holds the "Data Options" parameters for the 
    explore.thresholds panel. 
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

    # def thresholds(self, da, option=1):
    def thresholds(self, option=1):
        thresh_data = ThresholdDataParams(selections=self.selections, location=self.location)

        data_options_card = pn.Card(
            pn.Row(
                pn.Column(
                    pn.widgets.Select.from_param(thresh_data.param.variable2, name="Data variable"),
                    pn.widgets.StaticText.from_param(self.selections.param.variable_description),
                    width = 230),
                pn.Column(
                    pn.widgets.Select.from_param(thresh_data.param.area_subset2, name="Area subset"),
                    pn.widgets.Select.from_param(thresh_data.param.cached_area2, name="Cached area"),
                    self.location.view,
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