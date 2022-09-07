import xarray as xr 
import hvplot.xarray
import holoviews as hv
from holoviews import opts

def _visualize(data): 
    """Make single map"""
    data_i = data.isel(time=0,scenario=0,simulation=0)
    _plot = data_i.hvplot.image(
        x="x", y="y", 
        grid=True, 
        title="Generic map"
    )
    return _plot