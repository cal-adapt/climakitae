import xarray as xr
import pandas as pd
import shutil
import dask
from dask import delayed, compute
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()


def _export_to_user(user_export_format,data_to_export,file_name):
    """
    The data export method, called by core.Application.export, it saves
    a dataset (which can be quite large) to the current working directory
    in the output format requested by the user (which is
    stored in 'export_format').
    """
    
    
    
    if ("CSV" in user_export_format.output_file_format):
        print("csv")
        
    elif ("NetCDF" in user_export_format.output_file_format):
        print("nc")
        
        
    return(print("you will get your data "+str(data_to_export)+
                 " as a "+user_export_format.output_file_format+" called"+
                str(file_name)))
