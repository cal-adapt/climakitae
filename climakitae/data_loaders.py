import xarray as xr
from shapely.geometry import Polygon, box, Point
import regionmask
import intake
import numpy as np

# support methods for core.Application.generate

def _experiment_id(scenario):
    ''' 
    Returns the SSP designation in CMIP 'experiment_id' format based on the 
    contents of the human-readable scenario designation stored in 'selections'.
    '''
    split = scenario.split(' ')
    for i, one in enumerate(split):
        if 'SSP' in one:
            numbers = split[i+1].split('-')
            ssp = numbers[0]
            rcp = numbers[1].split('.')
            rcp = rcp[0]+rcp[1]
            return 'ssp'+ssp+rcp
        elif one == 'Historical Climate':
            return 'historical'
        elif one == 'Historical Reconstruction':
            return ''
    return None

def _get_file_list(cat,selections,scenario):
    '''
    Returns a list of simulation names for all of the simulations present in the catalog 
    for a given scenario, contingent on other user-supplied constraints in 'selections'.
    '''
    file_list = []
    for item in list(cat):
        if cat[item].metadata['nominal_resolution'] == selections.resolution:
            if cat[item].metadata['experiment_id'] == _experiment_id(scenario):
                file_list.append(cat[item].name)
    return file_list

def _get_var_name(cat,var_description):
    _ds = cat[list(cat)[0]].to_dask()
    for i in _ds.data_vars:
        if _ds[i].attrs['description'] == var_description:
            return i
    return 'T2' #a default that might not be what the user asked for...
    # add some handling for if it's not there for some reason,
    # to provide a more useful error message instead of returning whatever default

def _open_and_concat(cat,file_list,selections,geom):
    '''
    Open multiple zarr files, and add them to one big xarray Dataset. Coarsens in time, and/or 
    subsets in space if selections so-indicates. Won't work unless the file_list supplied
    contains files of only one nominal resolution (_get_file_list ensures this).
    '''
    all_files = xr.Dataset()
    for one_file in file_list:
        with cat[one_file].to_dask() as data:
            source_id = data.attrs['source_id']
            if selections.variable == 'precipitation (total)':
                pass
            elif selections.variable == 'wind 10m magnitude':
                pass
            else:
                var_name = _get_var_name(cat,selections.variable)
                data = data[var_name]
            #coarsen in time if 'selections' so-indicates:
            if selections.timescale == 'daily':
                data = data.resample(time='1D').mean()
            if geom:
                #subset data spatially:
                ds_region = regionmask.Regions([geom],
                               abbrevs=['lat/lon box'],name='test mask')
                mask=ds_region.mask(data.lon,data.lat,wrap_lon=False)
                data = data.where(np.isnan(mask)==False).dropna('x',how='all').dropna('y',how='all')
            #add data to larger Dataset being built
            all_files.assign(source_id = data)
    return all_files.to_array('simulation') 

def _get_as_shapely(location):
    '''
    Takes the location data in the 'location' parameter, and turns it into a shapely object.
    Just doing polygons for now. Later other point/station data will be available too.
    '''
    #shapely.geometry.box(minx, miny, maxx, maxy):
    return box(location.longitude[0],location.latitude[0],location.longitude[1],location.latitude[1])

def _read_from_catalog(selections,location):
    '''
    The primary and first data loading method, called by core.Application.generate, it returns 
    a dataset (which can be quite large) containing everything requested by the user (which is 
    stored in 'selections' and 'location').
    '''
    cat = intake.open_catalog('s3://cdcat/cae.yaml')
    if (location.subset_by_lat_lon == True):
        geom = _get_as_shapely(location)
    else:
        geom = False #for now... later a cached polygon will be an elseif option too
        
    all_files = xr.Dataset()
    for one_scenario in selections.scenario:
        files_by_scenario = _get_file_list(cat,selections,one_scenario)
        temp = _open_and_concat(cat,files_by_scenario,selections,geom)
        all_files.assign(one_scenario=temp)
        #if selections.append_historical:
        #    files_historical = get_file_list(selections,'historical')
        #    all_files = xr.concat([files_historical,all_files],dim='time')
    return all_files
