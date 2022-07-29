from functools import wraps
from dask import delayed, compute
import hvplot.xarray
import cartopy.crs as ccrs
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

def _explore(data_to_use):
    da_plt = data_to_use.hvplot.quadmesh(
        'lon','lat', groupby=['time','scenario','simulation'], 
         crs=ccrs.PlateCarree(),projection=ccrs.Orthographic(-118, 40),
         project=True,rasterize=True,
         coastline=True, features=['borders'])
    
    return da_plt
    
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

