import xarray as xr 
import hvplot.xarray

def _visualize(data): 
    """Create a generic plot or map of the data"""
    
    # Workflow if data contains spatial coordinates 
    if set(["x","y"]).issubset(set(data.dims)):

        # Plotting inputs
        non_spatial_dims = [dim for dim in data.dims if dim not in ["x","y"]]
        try: 
            clabel = data.name + " ("+data.attrs["units"]+")"
        except: # Try except just in case units attribute is missing from data 
            clabel = data.name

        # Create map 
        _plot = data.hvplot.image(
            x="x", y="y", 
            grid=True, 
            groupby=non_spatial_dims, 
            clabel=clabel
        )

    # Workflow if data contains only time dimension
    elif "time" in data.dims: 

        # Create lineplot
        _plot = data.hvplot.line(x="time")
        
    return _plot