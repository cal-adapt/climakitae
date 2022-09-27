import xarray as xr
import pandas as pd
import shutil
import dask
import warnings
import datetime
import numpy as np
import rasterio
import os
xr.set_options(keep_attrs=True)


def export_to_netcdf(data_to_export,save_name,**kwargs):
    '''
    exports user-selected data to netCDF format.
    this function is called from the _export_to_user
    function if the user selected netCDF output.
    
    data_to_export: xarray dataset or array to export
    save_name: string corresponding to desired output file name + file extension
    kwargs: reserved for future use
    '''    
    print("Alright, exporting specified data to NetCDF. This might take a while - "+
         "hang tight!")
    data_to_export.to_netcdf(save_name)
    
    
def export_to_csv(data_to_export,save_name,**kwargs):
    '''
    exports user-selected data to CSV format.
    this function is called from the _export_to_user
    function if the user selected CSV output.
    
    data_to_export: xarray dataset or array to export
    save_name: string corresponding to desired output file name + file extension
    kwargs: reserved for future use
    '''
    
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
    '''
    exports user-selected data to geoTIFF format.
    this function is called from the _export_to_user
    function if the user selected geoTIFF output.
    
    data_to_export: xarray dataset or array to export
    save_name: string corresponding to desired output file name + file extension
    kwargs: reserved for future use
    '''
    ds_attrs = data_to_export.attrs
        
    # if x and/or y exist as coordinates
    # but have been squeezed out as dimensions
    # (eg we have point data), add them back in as dimensions.
    # rasters require both x and y dimensions
    if ('x' not in data_to_export.dims):
        if ('x' in data_to_export.coords):
            the_sel = data_to_export.expand_dims('x')
        else:
            raise ValueError("No x dimension or coordinate exists;"+
                             " cannot export to GeoTIFF. Please provide"+
                            " a data array with both x and y"+
                             " spatial coordinates.")
    if ('y' not in data_to_export.dims):
        if ('y' in data_to_export.coords):
            data_to_export = data_to_export.expand_dims('y')
        else:
            raise ValueError("No y dimension or coordinate exists;"+
                             " cannot export to GeoTIFF. Please provide"+
                            " a data array with both x and y"+
                             " spatial coordinates.")

    # squeeze singleton dimensions as long as they are 
    # simulation and/or scenario dimensions;
    # retain simulation and/or scenario metadata
    if (('scenario' in data_to_export.coords) and 
        ('scenario' not in data_to_export.dims)):
        scen_attrs = {'scenario' : str(data_to_export.coords['scenario'].values)}
        ds_attrs = dict(ds_attrs, **scen_attrs)    
    if (('scenario' in data_to_export.dims) and 
          (len(data_to_export.scenario)==1)):
        scen_attr = {'scenario' : str(data_to_export.scenario.values[0])}
        ds_attrs = dict(ds_attrs, **scen_attr)
        data_to_export = data_to_export.squeeze(dim='scenario')
    elif (('scenario' not in data_to_export.dims) and 
          ('scenario' not in data_to_export.coords)):
        warnings.warn("'scenario' not in data array as"+
                      " dimension or coordinate; this information"+
                      " will be lost on export to raster."+
                      " Either provide a data array"+
                      " which contains a single scenario"+
                      " as a dimension and/or coordinate,"+
                      " or record the scenario sampled"+
                      " for your records.")

    if (('simulation' in data_to_export.coords) and 
        ('simulation' not in data_to_export.dims)):
        sim_attrs = {'simulation' : str(data_to_export.coords['simulation'].values)}
        ds_attrs = dict(ds_attrs, **sim_attrs)
    if ((str('simulation') in data_to_export.dims) and 
    (len(data_to_export.simulation)==1)):
        sim_attrs = {'simulation' : str(data_to_export.simulation.values)}
        ds_attrs = dict(ds_attrs, **sim_attrs)
        data_to_export = data_to_export.squeeze(dim='simulation')
    elif (('simulation' not in data_to_export.dims) and 
          ('simulation' not in data_to_export.coords)):
        warnings.warn("'simulation' not in data array as"+
                      " dimension or coordinate; this information"+
                      " will be lost on export to raster."+
                      " Either provide a data array"+
                      " which contains a single simulation"+
                      " as a dimension and/or coordinate,"+
                      " or record the simulation sampled"+
                      " for your records.")
        
    ndim = len(data_to_export.dims)
    if (ndim == 3):
        if ('time' in data_to_export.dims):
            data_to_export = data_to_export.transpose('time', 'y', 'x')
            if (len(data_to_export.time) > 1):
                print("Saving as multiband raster in which"+
                 " each band corresponds to a time step.")
        elif ('simulation' in data_to_export.dims):
            data_to_export = data_to_export.transpose('simulation', 'y', 'x')
            if (len(data_to_export.simulation) > 1):
                print("Saving as multiband raster in which"+
                 " each band corresponds to a simulation.")
        elif ('scenario' in data_to_export.dims):
            data_to_export = data_to_export.transpose('scenario', 'y', 'x')
            if (len(data_to_export.scenario) > 1):
                print("Saving as multiband raster in which"+
                 " each band corresponds to a climate scenario.")
                    
    print("Saving as GeoTIFF...")
    data_to_export.rio.to_raster(save_name)
    meta_data_dict = ds_attrs
    
    with rasterio.open(save_name, 'r+') as raster:
        raster.update_tags(**meta_data_dict)
        raster.close()
    
def _export_to_user(user_export_format,data_to_export,
                    file_name,**kwargs):
    """
    The data export method, called by core.Application.export_dataset. Saves
    a dataset to the current working directory in the output 
    format requested by the user (which is stored in 'user_export_format').
    
    user_export_format: pulled from dropdown called by app.export_as()
    data_to_export: xarray ds or da to export
    file_name: string corresponding to desired output file name
    kwargs: variable, scenario, and simulation (as needed)
    """
    ndims = len(data_to_export.dims)
    file_name = file_name.split('.')[0]
    
    assert type(file_name) is str,("Please pass a string "+
    "(any characters surrounded by quotation marks)"+
        " for your file name.")
    
    path_and_name = './'+file_name
    req_format = user_export_format.output_file_format
    if req_format is None:
        raise AssertionError("Please select a file format from the dropdown menu.")
    
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
        
    assert "xarray" in str(
            type(data_to_export)
            ),("Please pass an xarray dataset (NetCDF only)"+
               " or data array (any format).")
       
    # metadata stuff
    ds_attrs.update(ck_attrs)
    data_to_export.attrs = ds_attrs
    
    # now check file size and avail workspace disk space
    # raise error for not enough space
    # and warning for large file
    file_size_threshold = 5 # in GB
    bytes_per_gigabyte = 1024*1024*1024    
    disk_space = shutil.disk_usage('./')[2]/bytes_per_gigabyte
    data_size = data_to_export.nbytes/bytes_per_gigabyte
    
    if (disk_space <= data_size):
        raise ValueError("Not enough disk space to export data!"+
                        " You need at least "+str(data_size)+ " GB free"+
                        " in the hub directory, which has 10 GB total space."+
                         " Try smaller subsets of space, time,"+
                        " scenario, and/or simulation; pick a coarser"+
                        " spatial or temporal scale; or clean any exported datasets"+
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
        assert "DataArray" in str(type(data_to_export)),("We are only"+
                             " converting data arrays to this format"+
                             " at this time, please pass a data array"+
                             " (not a dataset). HINT:" ) 
        if ("CSV" in req_format):
            export_to_csv(data_to_export,save_name,**kwargs)
        elif ("GeoTIFF" in req_format):
            dim_check = data_to_export.isel(x=0,y=0).squeeze().shape
            shape_str = str(dim_check).strip('(').strip(')').replace(", "," x ")
            if sum([int(dim>1) for dim in dim_check]) > 1:
                raise AssertionError("Too many non-spatial dimensions"+
                                     " with length > 1 -- cannot convert"+
                                     " to GeoTIFF. Current data shape"+
                                     " excluding x and y coordinates is: "+
                                     shape_str+". Please subset your"+
                                     " selection accordingly.")
                
            export_to_geotiff(data_to_export,save_name,**kwargs)    
                  
    return(print("Saved! You can find your file(s) in the panel to the left "+
                "and download to your local machine from there." ))
                

            