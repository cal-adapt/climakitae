import xarray as xr
import pandas as pd
import shutil
import dask
import warnings
import datetime
import numpy as np
xr.set_options(keep_attrs=True)
import os

def export_to_netcdf(data_to_export,save_name,**kwargs):
    print("Alright, exporting specified data to NetCDF. This might take a while - "+
         "hang tight!")
    data_to_export.to_netcdf(save_name)
    
def export_to_csv(data_to_export,save_name,**kwargs):
    print("WARNING: Exporting to CSV can take a long time,"+
                 " and is not recommended for data with more than 2 dimensions."+
                  " Please note that the data will be compressed via gzip. Even so,"+
         " inherent inefficiencies may result in a file which is too large to save here.")
    print("Converting data...")
    excel_row_limit = 1048576
    to_save = data_to_export.to_dataframe()        
    csv_nrows = len(to_save.index)        
    if (csv_nrows > excel_row_limit):
        print("WARNING: Dataset exceeds Excel limit of "+
                      str(excel_row_limit)+" rows.")        
    print("Compressing data... This can take a bit...")
    to_save.to_csv(save_name,compression='gzip')

def export_to_geotiff(data_to_export,save_name,**kwargs):
    if ('time' in data_to_export.dims):
        print("NOTE: Saving as multiband raster in which "+
             "each band corresponds to a time step.")
        print("See metadata file for more information.")
    print("Saving as GeoTIFF...")
    data_to_export.rio.to_raster(save_name)

def _export_to_user(user_export_format,data_to_export,
                    file_name,**kwargs):
    """
    The data export method, called by core.Application.export_dataset. Saves
    a dataset to the current working directory in the output format requested by the user 
    (which is stored in 'user_export_format').
    
    user_export_format: pulled from dropdown called by app.export_as()
    data_to_export: xarray ds or da to export
    file_name: string corresponding to desired output file name
    kwargs: variable, scenario, and simulation (as needed)
    """
    
    file_name = file_name.split('.')[0]
    
    assert type(file_name) is str,("Please pass a string "+
    "(any characters surrounded by quotation marks)"+
        " for your file name.")
    
    path_and_name = './'+file_name
    req_format = user_export_format.output_file_format
    
    extension_dict = {'NetCDF' : '.nc',
                      'CSV' : '.gz',
                      'GeoTIFF' : '.tif'}
    
    f_extension = extension_dict[req_format]
    save_name = path_and_name+f_extension
    
    if os.path.exists(save_name):
        raise AssertionError("File " + save_name+ " exists," +
                             " please either delete that file" +
                             " from the work space or specify" +
                             " a new file name here.")
   
    ds_attrs = data_to_export.attrs
    ct = datetime.datetime.now()
    ct_str = ct.strftime("%d-%b-%Y (%H:%M:%S)")    
    ck_attrs = {'data_exported_from' : 'Cal-Adapt Analytics Engine v 0.0.1',
               'data_export_timestamp' : ct_str}       
    
    if req_format is None:
        raise AssertionError("Please select a file format from the dropdown menu.")
        
    assert "xarray" in str(
            type(data_to_export)
            ), "Please pass an xarray dataset or data array (e.g., as output by generate)."
    
    
    # metadata stuff
    ds_attrs.update(ck_attrs)
    data_to_export.attrs = ds_attrs
    
    # squeeze out singleton dims
    data_to_export = data_to_export.squeeze(drop=True)
    
    # now check file size and avail workspace disk space
    # raise error for not enough space
    # and warning for large file
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
        
    # now here is where exporting actually begins
    # we will have different functions for each file type
    # to keep things clean-ish    
    if ("NetCDF" in req_format):
        export_to_netcdf(data_to_export,save_name,**kwargs) 
    else:
        if "Dataset" in str(type(data_to_export)):
            raise ValueError("We are only converting"+
                             " data arrays to this format at this time,"+
                             " please pass a data array"+
                             " (not a dataset).") 
        else:
            if ("CSV" in req_format):
                export_to_csv(data_to_export,save_name,**kwargs)
            elif ("GeoTIFF" in req_format):
                export_to_geotiff(data_to_export,save_name,**kwargs)

        
    return(print("All done!"))
    
    # if "Dataset" in str(type(data_to_export)):
    #     if ("variable" not in kwargs):
    #         raise ValueError("Please pass variable = 'var'"+
    #                          " anywhere after file_name in the export_dataset() call."+
    #                         " E.g., 'app.export_dataset(data,file_name='example',"+
    #                         "variable='T2')")
    #     else:
    #         assert type(kwargs['variable']) is str,("Please pass variable name "+
    #         "in double or single quotation marks.")
    #         var = kwargs['variable']
    #         var_attrs = data_to_export[var].attrs
    #         data_to_export = data_to_export[var]
    #         ds_attrs.update(var_attrs)

            
    # we have to ensure that each non-NetCDF has only one simulation
    # and scenario per file. 
#     if ("NetCDF" not in req_format):
#         if ('simulation' in data_to_export.coords): # if simulation coord exists
#             if ('simulation' not in kwargs): # and one simulation is not supplied by user
#                 if (np.size(data_to_export.coords['simulation'].values) > 1): # and there is > 1
#                     raise ValueError("File format does not allow for data from"+
#                             " more than one simulation. Please pass simulation = 'sim'"+
#                              " anywhere after file_name in the export_dataset() call."+
#                              " E.g., 'app.export_dataset(data,file_name='example',"+
#                             "simulation='cesm2')")
#                 else: # automatically pull the simulation name if there is 1
#                     simu = data_to_export.coords['simulation'].values                    
#             else: # it is a kwarg
#                 assert type(kwargs['simulation']) is str,("Please pass simulation name "+
#                 "in double or single quotation marks.")
#                 simu = kwargs['simulation']
                    
#             sim_dict = {'simulation_model' : simu}
#             # ds_attrs.update(sim_dict)
#             data_to_export['simulation'].attrs = {'simulation_model' : simu}
#             data_to_export = data_to_export.sel(simulation=simu) # subset        
    
#     # same as above, but for scenario instead of simulation
#         if ('scenario' in data_to_export.coords): 
#             if ('scenario' not in kwargs): 
#                 if (np.size(data_to_export.coords['scenario'].values) > 1):
#                     raise ValueError("File format does not allow for data from"+
#                             " more than one scenario. Please pass scenario = 'scenario'"+
#                              " anywhere after file_name in the export_dataset() call."+
#                              " E.g., 'app.export_dataset(data,file_name='example',"+
#                             "scenario='historical')")
#                 else: 
#                     scen = data_to_export.coords['scenario'].values                    
#             else: 
#                 assert type(kwargs['scenario']) is str,("Please pass scenario name "+
#                 "in double or single quotation marks.")
#                 scen = kwargs['scenario']
                    
#             scen_dict = {'climate_scenario' : scen}
#             ds_attrs.update(scen_dict)
#             data_to_export['scenario'].attrs = scen_dict
#             data_to_export = data_to_export.sel(scenario=scen) # subset                    
              
            
#     if ("NetCDF" in req_format):
#         print("Alright, exporting specified data to NetCDF. This might take a while - "+
#              "hang tight!")
#         # data_to_export.attrs = var_atts 
        
#         data_to_export.attrs = ds_attrs
#         data_to_export.to_netcdf(save_name)
        
#     else:        
#         data_to_export = data_to_export.squeeze()
#         print("NOTE: Metadata will be saved to a separate file called "+
#               file_name+"_metadata.txt. Be sure to download it"+
#               " when you download your data!")       
#         metadata_to_file(data_to_export,file_name,req_format)# make metadata text file


#         elif ("GeoTIFF" in req_format):
#             if ('time' in data_to_export.dims):
#                 print("NOTE: Saving time series as multiband raster in which "+
#                      "each band corresponds to a time step.")
#                 print("See metadata file for more information.")
#             print("Saving as GeoTIFF...")
#             data_to_export.rio.to_raster(save_name)
                   
    return(print("Saved! You can find your file(s) in the panel to the left "+
                "and download to your local machine from there." ))


def metadata_to_file(ds,output_name,req_format):
    """
    Writes NetCDF metadata to a txt file so users can still access it 
    after exporting to a CSV or GeoTIFF.
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
            f.write("==== Note: Conversion from NetCDF to GeoTIFF is experimental,"+
                    " and may result in loss of coordinate data. We have tried to "+
                    " reproduce the necessary reference information"+
                    " in this metadata file, but some may be missing. ====")
            f.write('\n')
            f.write("==== All coordinates come from the original NetCDF,"+
                   " and may not exist in this raster. ====")
            f.write('\n')
            
            if ('time' in ds.dims):
                first_ts = ds['time'].values[0]
                f.write("==== Note: Time series was saved as a multiband raster,"+
                   " and each band corresponds to a time step. ====")
                f.write('\n')
                f.write("First time slice is: "+str(first_ts)+". Use this and 'frequency'"+
                        " attribute to determine time stamp for each band." ) 
                f.write('\n')

        for coord in ds.coords:
            f.write('\n')
            f.write("== "+str(coord)+" ==")
            f.write('\n')
            for att_keys,att_values in list(zip(ds[coord].attrs.keys(),
                                            ds[coord].attrs.values())):    
                f.write(str(att_keys)+" : "+str(att_values))
                f.write('\n')
                

            