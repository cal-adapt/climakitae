import xarray as xr
import pandas as pd
import shutil
import dask
import warnings
import datetime
import numpy as np
xr.set_options(keep_attrs=True)
import os

def _export_to_user(user_export_format,data_to_export,
                    file_name,**kwargs):
    """
    The data export method, called by core.Application.export_dataset. Saves
    a dataset to the current working directory in the output format requested by the user 
    (which is stored in 'user_export_format').
    
    user_export_format: pulled from dropdown called by app.export_as()
    data_to_export: xarray ds or da to export
    variable_name: string corresponding to variable to export
    file_name: string corresponding to desired output file name
    kwargs: variable, scenario, and simulation (as needed)
    
    NOTE: requires the following at the top of the notebook:
    !pip install pytest-shutil
    """
    
    excel_row_limit = 1048576
    path_and_name = './'+file_name
    req_format = user_export_format.output_file_format
    ds_attrs = data_to_export.attrs
    ct = datetime.datetime.now()
    ct_str = ct.strftime("%d-%b-%Y (%H:%M:%S)")    
    ck_attrs = {'data_retrieved_by' : 'Cal-Adapt Analytics Engine v 0.0.1',
               'data_export_timestamp' : ct_str}   
    ds_attrs.update(ck_attrs)
        
    assert "xarray" in str(
            type(data_to_export)
            ), "Please pass an xarray dataset or data array (e.g., as output by generate)."
    
    assert type(file_name) is str,("Please pass a string "+
    "(any characters surrounded by quotation marks)"+
        " for your file name.")
    
    if "Dataset" in str(type(data_to_export)):
        if ("variable" not in kwargs):
            raise ValueError("Please pass variable = 'var'"+
                             " anywhere after file_name in the export_dataset() call."+
                            " E.g., 'app.export_dataset(data,file_name='example',"+
                            "variable='T2')")
        else:
            assert type(kwargs['variable']) is str,("Please pass variable name "+
            "in double or single quotation marks.")
            var = kwargs['variable']
            data_to_export = data_to_export[var]
            
    # we have to ensure that each non-NetCDF has only one simulation
    # and scenario per file. 
    if ("NetCDF" not in req_format):
        if ('simulation' in data_to_export.coords): # if simulation coord exists
            if ('simulation' not in kwargs): # and one simulation is not supplied by user
                if (np.size(data_to_export.coords['simulation'].values) > 1): # and there is > 1
                    raise ValueError("File format does not allow for data from"+
                            " more than one simulation. Please pass simulation = 'sim'"+
                             " anywhere after file_name in the export_dataset() call."+
                             " E.g., 'app.export_dataset(data,file_name='example',"+
                            "simulation='cesm2')")
                else: # automatically pull the simulation name if there is 1
                    simu = data_to_export.coords['simulation'].values                    
            else: # it is a kwarg
                assert type(kwargs['simulation']) is str,("Please pass simulation name "+
                "in double or single quotation marks.")
                simu = kwargs['simulation']
                    
            sim_dict = {'simulation_model' : simu}
            # ds_attrs.update(sim_dict)
            data_to_export['simulation'].attrs = {'simulation_model' : simu}
            data_to_export = data_to_export.sel(simulation=simu) # subset        
    
    # same as above, but for scenario instead of simulation
        if ('scenario' in data_to_export.coords): 
            if ('scenario' not in kwargs): 
                if (np.size(data_to_export.coords['scenario'].values) > 1):
                    raise ValueError("File format does not allow for data from"+
                            " more than one scenario. Please pass scenario = 'scenario'"+
                             " anywhere after file_name in the export_dataset() call."+
                             " E.g., 'app.export_dataset(data,file_name='example',"+
                            "scenario='historical')")
                else: 
                    scen = data_to_export.coords['scenario'].values                    
            else: 
                assert type(kwargs['scenario']) is str,("Please pass scenario name "+
                "in double or single quotation marks.")
                scen = kwargs['scenario']
                    
            scen_dict = {'climate_scenario' : scen}
            # ds_attrs.update(scen_dict)
            data_to_export['scenario'].attrs = scen_dict
            data_to_export = data_to_export.sel(scenario=scen) # subset                    
   
    data_to_export.attrs = ds_attrs
    data_to_export = data_to_export.squeeze(drop=True)
    file_size_threshold = 5 # in GB
    bytes_per_gigabyte = 1024*1024*1024    
    disk_space = shutil.disk_usage('./')[2]/bytes_per_gigabyte
    data_size = data_to_export.nbytes/bytes_per_gigabyte
    
    if (disk_space < data_size):
        raise ValueError("Not enough disk space to export data!"+
                        " You need at least "+str(data_size)+ " GB free"+
                        " in the hub directory, which has 10 GB total space."+
                         " Try smaller subsets of space, time,"+
                        " scenario, and/or simulation, pick a coarser"+
                        " spatial or temporal scale, or clean any exported datasets"+
                        " which you have already downloaded or do not want.")
    
    if (data_size > file_size_threshold):
        print("WARNING: xarray dataset size = "+str(data_size)+
              " GB. This might take a while!")              
              
            
    if ("NetCDF" in req_format):
        print("Alright, exporting specified data to NetCDF. This might take a while - "+
             "hang tight!")
        data_to_export.to_netcdf(path_and_name+".nc")
        
    else:        
        data_to_export = data_to_export.squeeze()
        print("NOTE: Metadata will be saved to a separate file called "+
              file_name+"_metadata.txt. Be sure to download it"+
              " when you download your data!")       
        metadata_to_file(data_to_export,file_name,req_format)# make metadata text file
    
        if ("CSV" in req_format):
            print("WARNING: Exporting to CSV can take a long time,"+
                         " and is not recommended for data with more than 2 coordinates."+
                          " Please note that the data will be compressed via gzip. Even so,"+
                 " inherent inefficiencies may result in a file which is too large to save here.")
            print("Converting data...")
            to_save = data_to_export.to_dataframe()        
            csv_nrows = len(to_save.index)        
            if (csv_nrows > excel_row_limit):
                print("WARNING: Dataset exceeds Excel limit of "+
                              str(excel_row_limit)+" rows.")        
            print("Compressing data... This can take a bit...")
            to_save.to_csv(path_and_name+".gz",compression='gzip')

        elif ("GeoTIFF" in req_format):
            if ('time' in data_to_export.dims):
                print("NOTE: Saving time series as multiband raster in which "+
                     "each band corresponds to a time step.")
                print("See metadata file for more information.")
            print("Saving as GeoTIFF...")
            data_to_export.rio.to_raster(path_and_name+".tif")
                   
    return(print("Saved! You can find your file(s) in the panel to the left "+
                "and download to your local machine from there." ))


def metadata_to_file(ds,output_name,req_format):
    """
    Writes NetCDF metadata to a txt file so users can still access it 
    after exporting to a CSV or GeoTIFF.
    Does not appear to work for GeoTIFF.
    """
    if os.path.exists(output_name+"_metadata.txt"):
        os.remove(output_name+"_metadata.txt")
        
    with open(output_name+"_metadata.txt", 'w') as f:
        f.write('======== Metadata for '+req_format+' file '+output_name+' ========')
        f.write('\n')
        f.write('\n')
        f.write('\n')
        f.write('===== Global file attributes =====')
        f.write('\n')
        f.write('\n')
        for att_keys,att_values in list(zip(ds.attrs.keys(),ds.attrs.values())):    
            f.write(str(att_keys)+" : "+str(att_values))
            f.write('\n')                       

# only used if exporting datasets
#         f.write('\n')
#         f.write('\n')
#         f.write('===== Variable attributes =====')
#         f.write('\n')

#         for var in ds.data_vars:
#             f.write('\n')
#             f.write("== "+str(var)+" ==")
#             f.write('\n')
#             f.write("Variable dimensions: "+str(ds[var].dims))
#             f.write('\n')
#             for att_keys,att_values in list(zip(ds[var].attrs.keys(),ds[var].attrs.values())):    
#                 f.write(str(att_keys)+" : "+str(att_values))
#                 f.write('\n')

        f.write('\n')
        f.write('\n')
        f.write('===== Coordinate attributes =====')
        f.write('\n')
        
        if ("GeoTIFF" in req_format):
            f.write("==== Note: All coordinates come from the original NetCDF,"+
                   " and may not exist in this raster. ====")
            f.write('\n')
            
            if ('time' in ds.dims):
                first_ts = ds['time'].values[0]
                f.write("==== Note: Time series was saved as a multiband raster,"+
                   " and each band corresponds to a time step. ====")
                f.write('\n')
                f.write("First time slice is: "+str(first_ts)+". Use this and 'units'"+
                        " attribute to determine time stamp for each band." ) 
                f.write('\n')

        for coord in ds.coords:
            f.write('\n')
            f.write("== "+str(coord)+" ==")
            f.write('\n')
            for att_keys,att_values in list(zip(ds[coord].attrs.keys(),ds[coord].attrs.values())):    
                f.write(str(att_keys)+" : "+str(att_values))
                f.write('\n')

            