from functools import wraps
from dask import delayed, compute
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

def explore(my_data, do_smoothing=False):
    """
    A placeholder for now, which will be generalized to display the most useful interactive hvplot
    for timeseries, probably in an xarray Dataset instead of a pandas DataFrame as currently written.
    As an initial skeleton, this code should be replaced, and is just meant as an example.
    """
    scenario_means = my_data.T.groupby(level="scenario").mean().T
    anom = scenario_means - scenario_means["1850":"1980"].mean()
    if do_smoothing == True:
        anom = anom.rolling(120, center=True).mean()[
            "2000":
        ]  # defaults to window-size for min periods, and closed=right

    dfPlot = anom.hvplot(
        label="Temperature"
    )  # need to get the dates displaying along the x-axis

    warming = [2, 3, 4]  # DEGREES
    linewidths = [0.3, 0.6, 1.1]
    scenario_colors = ["b", "r", "orange", "g"]  # the color order they plot in...
    temp_list = [dfPlot]
    for j, degrees in enumerate(warming):
        line_horiz = hv.Curve((anom.index, np.zeros(len(anom.index)) + degrees))
        line_horiz = line_horiz.opts(color="black", line_width=linewidths[j])
        temp_list.append(line_horiz)
        for i, scenario in enumerate(anom):
            level = get_global_warming_levels(anom, degrees)
            temp = hv.Curve(
                ([level[scenario] for k in np.arange(10)], np.linspace(0, degrees, 10))
            )
            temp = temp.opts(line_width=linewidths[j], color=scenario_colors[i])
            temp_list.append(temp)
    linePlot = hv.Overlay(temp_list)
    return linePlot


def progress_bar(func):
    @wraps(func)
    def pbar_wrapper(*args, **kwargs):
        """
        Generic decorator that shows a progress bar for any
        function using dask delayed and compute functionality.
        To use, put the following two lines above the 
        function definition (in this order):
        @progress_bar
        @delayed        
        """          
        with pbar:
            
            print(f"Request in progress. This may take a while. "+
                 "Thanks for your patience!")

            the_request = func(*args, **kwargs).compute()
            
        return(the_request)
        
    return(pbar_wrapper)

