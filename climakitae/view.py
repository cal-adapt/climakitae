import xarray as xr 
import numpy as np
import hvplot.xarray
import matplotlib.pyplot as plt 
import warnings
import pkg_resources
from .utils import _reproject_data, _read_ae_colormap, _read_var_csv

# Variable descriptions csv with colormap info 
var_descrip_pkg = pkg_resources.resource_filename('climakitae', 'data/variable_descriptions.csv')
var_descrip = _read_var_csv(var_descrip_pkg, index_col="description")

def _visualize(data, lat_lon=True, width=None, height=None, cmap=None): 
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
        
        # Set default cmap if no user input
        if cmap is None: 
            try: 
                cmap_name = var_descrip[data.name]["default_cmap"]
            except: # If variable not found, set to ae_orange without raising error 
                cmap_name = "ae_orange"
            
        # Must have more than one grid cell to generate a map 
        if (len(data["x"]) <= 1) and (len(data["y"]) <= 1):  
            print("Your data contains only one grid cell. A plot will be created using a default method that may or may not have spatial coordinates as the x and y axes.") # Warn user that plot may be weird 
            
            # Set default cmap if no user input
            # Different if using matplotlib (no "hex") 
            if cmap is None: 
                cmap = _read_ae_colormap(cmap=cmap_name)
            
            with warnings.catch_warnings():
                
                # Silence annoying matplotlib deprecation error 
                warnings.simplefilter("ignore")
                
                # Use generic static xarray plot
                _matplotlib_plot = data.plot(cmap=cmap) 
                _plot = plt.gcf() # Add plot to figure 
                plt.close() # Close to prevent annoying matplotlib collections object line from showing in notebook 
        
         # If there's more than one grid cell, generate a pretty map
        elif (len(data["x"]) > 1) and (len(data["y"]) > 1):  
        
            # Define colorbar label using variable and units 
            try: 
                clabel = data.name + " ("+data.attrs["units"]+")"
            except: # Try except just in case units attribute is missing from data 
                clabel = data.name
                
            # Set default cmap if no user input
            # Different if using hvplot (we need "hex") 
            if cmap is None: 
                cmap = _read_ae_colormap(cmap=cmap_name+"_hex")

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
                    warnings.warn("Data reprojection to lat/lon failed. Using native x,y grid.")
                    pass 

            # Create map 
            _plot = data.hvplot.image(
                x="x", y="y", 
                grid=True, 
                clabel=clabel, 
                cmap=cmap,
                width=width, 
                height=height
            )
        
        else: 
            raise ValueError("You've encountered a bug in the code. Check the view.py module to troubleshoot")
        
        
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