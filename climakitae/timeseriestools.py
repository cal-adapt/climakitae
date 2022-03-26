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
   
    anomaly = param.Boolean(default=True,label="Difference from a historical mean")
    reference_range = param.CalendarDateRange(default=(dt.datetime(1980,1,1),dt.datetime(2012,12,31)),
                                              bounds=(dt.datetime(1980,1,1),dt.datetime(2021,12,31)))
    remove_seasonal_cycle = param.Boolean(default=False)
    smoothing = param.ObjectSelector(default="None", objects=["None", "running mean"])
    _time_scales = dict(
        [("hours", "H"), ("days", "D"), ("months", "M"), ("years", "A")]
    )
    num_timesteps = param.Integer(default=0,bounds=(0,240))
    separate_seasons = param.Boolean(default=False,label="Disaggregate into four seasons")
    
    running_extremes = param.ObjectSelector(default="None", objects=["None","min","max","percentile"])
    resample_window = param.Integer(default=1,bounds=(1,30))
    resample_period = param.ObjectSelector(default="A",objects=_time_scales)
    percentile = param.Number(default=0,bounds=(0,1),
                              doc="Relevant if 'running extremes' is 'percentile.")
    overlay = param.Boolean(default=False,
                            doc="Overlay running extremes on existing view.")
    
    @param.depends("anomaly",watch=True)
    def update_seasonal_cycle(self):
        if not self.anomaly:
            self.remove_seasonal_cycle = False
    
    @param.depends("remove_seasonal_cycle",watch=True)
    def update_anom(self):
        if self.remove_seasonal_cycle:
            self.anomaly = True
            
    @param.depends("running_extremes",watch=True)
    def update_overlay(self):
        if self.running_extremes != "None":
            self.overlay = True
    
    def transform_data(self):
        """
        Returns a dataset that has been transformed in the ways that the params indicate,
        ready to plot in the preview window ("view" method of this class), or be saved out.
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
        
        def _running_mean(y):
            return y.rolling(time=self.num_timesteps,center=True).mean("time")
        
        if self.anomaly and not self.separate_seasons:
            to_plot = _getAnom(to_plot)
        elif self.anomaly and self.separate_seasons:
            to_plot = to_plot.groupby("time.season").apply(_getAnom)
        else:
            to_plot = to_plot
                   
        if self.smoothing == "running mean" and self.num_timesteps >0:
            if self.separate_seasons:
                to_plot = to_plot.groupby("time.season").apply(_running_mean)
            else:
                to_plot = _running_mean(to_plot)
            to_plot.name = str(self.num_timesteps)+" timesteps running mean"
            
        if self.running_extremes != "None":
            if self.running_extremes == "max":
                extremes = to_plot.resample(time=str(self.resample_window)+self.resample_period).max("time")
            elif self.running_extremes == "min":
                extremes = to_plot.resample(time=str(self.resample_window)+self.resample_period).min("time")
            elif self.running_extremes == "percentile":
                extremes = to_plot.resample(time=str(self.resample_window)+self.resample_period).quantile(q=self.percentile)
            new_name = str(self.resample_window)+self.resample_period+" "+self.running_extremes
            extremes['simulation'] = [str(k.values) +" -- "+ new_name for k in extremes.simulation]

            
            if self.overlay:
                to_plot = xr.merge([to_plot,extremes])
            else:
                to_plot = extremes

        return to_plot
    
    @param.depends("anomaly", "reference_range", "separate_seasons", "smoothing",  
                   "num_timesteps", "remove_seasonal_cycle", "running_extremes",
                   "resample_window","resample_period","percentile","overlay",watch=False)
    def view(self):
        """
        Does the main work of timeseries.explore(). Updating a plot in real-time
        to enable the user to preview the results of any timeseries transforms.
        """
        to_plot = self.transform_data()
                
        if self.separate_seasons: 
            obj = to_plot.hvplot.line(x="time",widget_location="bottom",
                                      by="simulation", groupby=["scenario","time.season"]
                                     ) 
        else:
            obj = to_plot.hvplot.line(x="time",by="simulation",
                                      widget_location="bottom"
                                     )
        return obj  

def _timeseries_visualize(choices):
    """
    Uses holoviz 'panel' library to display the parameters and view defined in an instance
    of TimeSeriesParams. Arranges the widgets in the panel in rows and columns.
    """
    return pn.Column(pn.Row(pn.Column(choices.param.anomaly, choices.param.reference_range, 
                            choices.param.remove_seasonal_cycle, choices.param.smoothing,
                            choices.param.num_timesteps, choices.param.separate_seasons),
                            pn.Spacer(width=50),
                            pn.Column(pn.Row(choices.param.running_extremes,
                                             choices.param.overlay,width=320),
                                      pn.Row(choices.param.resample_window,
                                             choices.param.resample_period,width=320),
                                      choices.param.percentile)),
                     choices.view)

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
        assert (
            "Historical + " in data.scenario.values.any()
            ), "Please append the historical period in your data retrieval."

        self.choices = TimeSeriesParams(data)
        
    def explore(self):
        return _timeseries_visualize(self.choices)

    def output_current(self):
        return self.choices.transform_data()
