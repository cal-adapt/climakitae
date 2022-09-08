import xarray as xr 
import numpy as np
import hvplot.xarray
import warnings 
from .utils import _reproject_data

def _visualize(data, lat_lon=True, width=None, height=None): 
    """Create a generic visualization of the data

    Args: 
        data (xr.DataArray)
        lat_lon (boolean): reproject to lat/lon coords? (default to True) 
        width (int): width of plot (default to hvplot.image default) 
        height (int): hight of plot (default to hvplot.image default) 

    Returns: 
        hvplot.image()

    """
    
    # Warn user about speed if passing a zarr to the function
    if data.chunks is not None: 
        warnings.warn("This function may be quite slow unless you call .compute() on your data before passing it to app.view()")
        
    # Raise warning if width or height not provided as a pair 
    if (width is None and height is not None) or (height is None and width is not None): 
        warnings.warn("You must pass both a width and a height. Setting to plotting defaults.")
        
    # Workflow if data contains spatial coordinates 
    if set(["x","y"]).issubset(set(data.dims)):

        # Define colorbar label using variable and units 
        try: 
            clabel = data.name + " ("+data.attrs["units"]+")"
        except: # Try except just in case units attribute is missing from data 
            clabel = data.name
        
        # Reproject data to lat/lon
        if lat_lon == True:
            data = _reproject_data(
                xr_da = data, 
                proj="EPSG:4326", 
                fill_value=np.nan
            ) 
        
        # Create map with width/height arguments 
        if width is not None and height is not None: 
            _plot = data.hvplot.image(
                x="x", y="y", 
                grid=True, 
                clabel=clabel, 
                width=width, height=height
            )
        
        # Create plot without width/height arguments
        else: 
            _plot = data.hvplot.image(
                x="x", y="y", 
                grid=True, 
                clabel=clabel
            )

    # Workflow if data contains only time dimension
    elif "time" in data.dims: 

        # Create lineplot with width/height arguments 
        if width is not None and height is not None: 
            _plot = data.hvplot.line(x="time", width=width, height=height)
        
        # Create plot without width/height arguments
        else: 
            _plot = data.hvplot.line(x="time")
    
    # Error raised if data does not contain [x,y] or time dimensions 
    else: 
        raise ValueError("Input data must contain valid spatial and/or time dimensions")
   
    return _plot