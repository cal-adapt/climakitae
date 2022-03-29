import xarray as xr
import pandas as pd
import shutil
import dask
import warnings
import datetime
xr.set_options(keep_attrs=True)
import os

def _export_to_user(user_export_format,data_to_export,
                    variable_name,file_name):
    """
    The data export method, called by core.Application.export_dataset. Saves
    a dataset to the current working directory in the output format requested by the user 
    (which is stored in 'user_export_format').
    
    user_export_format: pulled from dropdown called by app.export_as()
    data_to_export: xarray ds or da to export
    variable_name: string corresponding to variable to export
    file_name: string corresponding to desired output file name
    
    NOTE: requires the following at the top of the notebook:
    !pip install pytest-shutil
    """
    
    excel_row_limit = 1048576
    var = variable_name
    path_and_name = './'+file_name
    req_format = user_export_format.output_file_format
    
    assert "xarray" in str(
            type(data_to_export)
            ), "Please pass an xarray dataset (e.g., as output by generate)."
    
    assert type(variable_name) is str,("Please pass the name of a variable "+
    "surrounded by quotation marks.")
    
    assert type(file_name) is str,("Please pass a string "+
    "(any characters surrounded by quotation marks)"+
        " for your file name.")
    
    data_to_export = data_to_export.squeeze()
    
    file_size_threshold = .5 # in GB
    bytes_per_gigabyte = 1024*1024*1024    
    disk_space = shutil.disk_usage('./')[2]/bytes_per_gigabyte
    data_size = data_to_export.nbytes/bytes_per_gigabyte
    
    if (data_size > file_size_threshold):
        print("WARNING: xarray dataset size = "+str(data_size)+
              " GB. This might take a while!")              
    if (disk_space < data_size):
        raise ValueError("Not enough disk space to export data!"+
                        "You need at least "+str(data_size)+ " GB free"+
                        " in the hub directory.")       
        
    ct = datetime.datetime.now()
    ds_attrs = data_to_export.attrs
    ts_attr = {'data_export_timestamp' : ct}
    ds_attrs.update(ts_attr)
    data_to_export.attrs = ds_attrs
        
    if ("NetCDF" in req_format):
        print("Alright, exporting specified data to NetCDF.")
        data_to_export.to_netcdf(path_and_name+".nc")
        
    else:
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
            to_save = data_to_export[var].to_dataframe()        
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
            data_to_export[var].rio.to_raster(path_and_name+".tif")
                   
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

        f.write('\n')
        f.write('\n')
        f.write('===== Variable attributes =====')
        f.write('\n')

        for var in ds.data_vars:
            f.write('\n')
            f.write("== "+str(var)+" ==")
            f.write('\n')
            f.write("Variable dimensions: "+str(ds[var].dims))
            f.write('\n')
            for att_keys,att_values in list(zip(ds[var].attrs.keys(),ds[var].attrs.values())):    
                f.write(str(att_keys)+" : "+str(att_values))
                f.write('\n')

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

            