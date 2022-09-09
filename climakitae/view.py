import xarray as xr 
import numpy as np
import hvplot.xarray
import warnings 
from .utils import _reproject_data

def _visualize(data, lat_lon=True, width=None, height=None, cmap="inferno_r"): 
    """Create a generic visualization of the data

    Args: 
        data (xr.DataArray)
        lat_lon (boolean): reproject to lat/lon coords? (default to True) 
        width (int): width of plot (default to hvplot.image default) 
        height (int): hight of plot (default to hvplot.image default) 
        cmap (str): colormap to apply to data (default to "viridis"); applies only to mapped data 

    Returns: 
        hvplot.image()

    """
    
    # Warn user about speed if passing a zarr to the function
    if data.chunks is not None: 
        warnings.warn("This function may be quite slow unless you call .compute() on your data before passing it to app.view()")
          
    # Workflow if data contains spatial coordinates 
    if set(["x","y"]).issubset(set(data.dims)):
        
        # Define colorbar label using variable and units 
        try: 
            clabel = data.name + " ("+data.attrs["units"]+")"
        except: # Try except just in case units attribute is missing from data 
            clabel = data.name
            
        # Set default width & height 
        if width is None: 
            width = 550
        if height is None: 
            height = 450
        
        # Reproject data to lat/lon
        if lat_lon == True:
            try: 
                data = _reproject_data(
                    xr_da = data, 
                    proj="EPSG:4326", 
                    fill_value=np.nan
                ) 
            except: # Reprojection can fail if the data doesn't have a crs element. If that happens, just carry on without projection (i.e. don't raise an error) 
                pass 
        
        # Create map 
        _plot = data.hvplot.image(
            x="x", y="y", 
            grid=True, 
            clabel=clabel, 
            width=width, 
            height=height, 
            cmap=cmap
        )
        
    # Workflow if data contains only time dimension
    elif "time" in data.dims: 
        
        # Set default width & height 
        if width is None: 
            width = 600 
        if height is None: 
            height = 300

        # Create lineplot
        _plot = data.hvplot.line(x="time", width=width, height=height)

    # Error raised if data does not contain [x,y] or time dimensions 
    else: 
        raise ValueError("Input data must contain valid spatial dimensions (x,y) and/or time dimensions")
   
    return _plot