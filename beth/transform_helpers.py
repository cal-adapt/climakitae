import xarray as xr
import inspect
import sys
import dask
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from functools import wraps
pbar = ProgressBar()
pbar.register()


def progress_bar(func):
    @wraps(func)
    def pbar_wrapper(ds,*args, **kwargs):
        """
        defines a decorator that shows a progress bar for any 
        transform. To use, import transform_helpers.progress_bar
        as progress_bar, then put the following two lines 
        above the transform definition:
        @delayed
        @progress_bar
        """        
        with pbar:
            print(f"Applying transform. This may take a while. "+
                 "Thanks for your patience!")
            transformed_ds = func(ds,*args, **kwargs).compute()
        return(transformed_ds)
    return(pbar_wrapper)

# will try to put in an append_metadata decorator here