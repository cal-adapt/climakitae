import xarray as xr
xr.set_options(keep_attrs=True)
from metadata_update import transform_details

@transform_details
def temporal_mean(ds,*args,**kwargs):
    
    """
    returns a full-record temporal mean of a dataset (ds).
    args are not doing anything right now but changing metadata,
    but one kwarg does something just to show that options
    can be updated flexibly as metadata.
    """
    
    # simple example: a temporal mean
    # let's throw in an option for time slicing:    
    if ('time_bounds'):
        time_list = kwargs.get("time_bounds")
        t0 = time_list[0]
        t1 = time_list[1]
        
    else: # if operating over the entire record
        t0 = ds.time.isel(time=0).values
        t1 = ds.time.isel(time=1).values
            
    ds_transformed = ds.sel(time=slice(t0,t1)).mean(dim='time')     
    
    return(ds_transformed)