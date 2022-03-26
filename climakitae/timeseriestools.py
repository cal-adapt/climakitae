import xarray as xr
import param
import panel as pn
import panel.widgets as pnw
from matplotlib.figure import Figure


class TimeSeriesParams(param.Parameterized):
    """
    An object to hold time-series parameters, which depends only on the 'param' library.
    Currently used in '_timeseries_visualize', which uses 'panel' to draw the gui, but another
    UI could in principle be used to update these parameters instead.
    """
    def __init__(self,dataset,**params):
        super().__init__(**params)
        self.data = dataset
   
    reference_range = param.CalendarDateRange(default=(dt.datetime(1980,1,1),dt.datetime(2012,12,31)),
                                              bounds=(dt.datetime(1980,1,1),dt.datetime(2021,12,31)))
    remove_seasonal_cycle = param.Boolean(default=False)
    anomaly = param.Boolean(default=True) 
    separate_seasons = param.Boolean(default=False)
    smoothing = param.ObjectSelector(default="None", objects=["None", "running mean"])
    _time_scales = dict(
        [("H", "hours"), ("D", "days"), ("M", "months"), ("A", "years")]
    )
    num_timesteps = param.Integer(default=0,bounds=(0,240))
        
    @param.depends("anomaly", "reference_range", "separate_seasons", "smoothing", "num_timesteps", 
                   "remove_seasonal_cycle", watch=False)
    def view(self):
        """
        Does the main work of timeseries.explore(). Updating a plot in real-time
        to enable the user to preview the results of any timeseries transforms.
        """
        
        if self.remove_seasonal_cycle:
            to_plot = self.data.groupby("time.month") - self.data.groupby("time.month").mean("time")
        else:
            to_plot = self.data
        
        def _getAnom(y):
            """
            Returns the difference with respect to the average across a historical range.
            """
            return y - y.sel(time=slice(
                self.reference_range[0],
                self.reference_range[1])).mean("time")
        
        if self.anomaly and not self.separate_seasons:
            to_plot = _getAnom(to_plot)
        elif self.anomaly and self.separate_seasons:
            to_plot = to_plot.groupby("time.season").apply(_getAnom)
        else:
            to_plot = to_plot
        
        
        if self.smoothing == "running mean":
            if self.num_timesteps > 0:
                to_plot = to_plot.rolling(time=self.num_timesteps,center=True).mean("time")
                to_plot.name = str(self.num_timesteps)+" timesteps running mean"
            
        if self.separate_seasons:
            obj = to_plot.hvplot.line(x="time",widget_location="bottom",
                                      by="simulation", groupby=["scenario","time.season"]
                                     ) 
        else:
            obj = to_plot.hvplot.line(x="time",by="simulation",
                                      widget_location="bottom")
        return obj 

def _timeseries_visualize(choices):
    """
    Uses holoviz 'panel' library to display the parameters and view defined in an instance
    of TimeSeriesParams.
    """
    return pn.Column(choices.param,choices.view)

class Timeseries():
    """
    Holds the instance of TimeSeriesParams that is used both to display a panel that
    previews various time-series transforms (explore), and to save the transform represented by 
    the current state of that preview into a new variable (output_current).
    """
    def __init__(self,data):
        assert "xarray" in str(
            type(data)
            ), "Please pass an xarray dataset (e.g. as output by generate)."
        assert (
            "lat" not in data.dims
            ), "Please pass a timeseries (area average or individual station)."

        self.choices = TimeSeriesParams(data)
        
    def explore(self):
        return _timeseries_visualize(self.choices)

    def output_current(self):
        pass
