import param
import panel as pn
import intake
from shapely.geometry import Point, box, Polygon
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#constants: instead will be read from database of some kind:
_cached_stations = ['','BAKERSFIELD MEADOWS FIELD','BLYTHE ASOS','BURBANK-GLENDALE-PASADENA AIRPORT','LOS ANGELES DOWNTOWN USC CAMPUS','NEEDLES AIRPORT','FRESNO YOSEMITE INTERNATIONAL AIRPORT','IMPERIAL COUNTY AIRPORT','LAS VEGAS MCCARRAN INTERNATIONAL AP','LOS ANGELES INTERNATIONAL AIRPORT','LONG BEACH DAUGHERTY FIELD','MERCED MUNICIPAL AIRPORT','MODESTO CITY-COUNTY AIRPORT','SAN DIEGO MIRAMAR WSCMO','OAKLAND METRO INTERNATIONAL AIRPORT','OXNARD VENTURA COUNTY AIRPORT','PALM SPRINGS REGIONAL AIRPORT','RIVERSIDE MUNICIPAL AIRPORT','RED BLUFF MUNICIPAL AIRPORT','SACRAMENTO EXECUTIVE AIRPORT','SAN DIEGO LINDBERGH FIELD','SANTA BARBARA MUNICIPAL AIRPORT','SAN LUIS OBISPO AIRPORT','GILLESPIE FIELD','SAN FRANCISCO INTERNATIONAL AIRPORT','SAN JOSE INTERNATIONAL AIRPORT','SANTA ANA JOHN WAYNE AIRPORT','THERMAL / PALM SPRINGS','UKIAH MUNICIPAL AIRPORT','LANCASTER WILLIAM J FOX FIELD']

#=== Select ===================================
class LocSelectorArea(param.Parameterized):
    '''
    Used to produce a panel of widgets for entering one of the types of location information used to 
    define a timeseries from an average over an area. Will update 'location' with whatever reflects the 
    last change made to any one of the options. User can 
    1. update the latitude or longitude of a bounding box
    2. select a predefined area, from a set of shapefiles we pre-select
    [future: 3. upload their own shapefile with the outline of a natural or administrative geographic area]
    '''
    subset_by_lat_lon = param.Boolean()
    #would be nice if these lat/lon sliders were greyed-out when subset option is not selected
    latitude = param.Range(default=(41, 42), bounds=(10, 67)) 
    longitude = param.Range(default=(-125,-115), bounds=(-156.82317,-84.18701)) 
    #shapefile = param.FileSelector(path='../../*/*.shp*', precedence=0.5) #not for March 2022
    #cached_area = param.ObjectSelector(objects=['CA','Sierra','LA County']) #this is a placeholder list
    
    #@param.depends('cached_area',watch=True)
    #def _update_loc_cached(self):
    #    '''Updates the 'location' object to be the Polygon associated with the selected geographic area.'''
    #    location = _areas_database[self.cached_area] #need this database to exist...
    
    # not for soft launch:
    #@param.depends('shapefile',watch=True)
    #def _update_loc_shp(self):
    #    '''Updates the 'location' object to be the polygon in the uploaded shapefile.
    #    Dealing with user-uploaded data of any kind might not be in the soft launch.'''
    #    # probably need to do checking for valid input (also for security reasons)
    #    user_location = gpd.read_file("shapefile.shp")
    #    assert user_location.geom_type in ['Point','Polygon'], "Please upload a valid shapefile."
    #    # maybe also offer interactive selecting if there's more than one
    #    # polygon in the shapefile!
    #    location = user_location
        
    #doesn't display yet for some reason:
    @param.depends('latitude','longitude',watch=True)
    def view(self):
        geometry = box(self.longitude[0],self.latitude[0],self.longitude[1],self.latitude[1])
        fig0 = plt.figure()
        ax = plt.subplot(111,projection=ccrs.Orthographic(-115,40))
        ax.set_extent([-160,-84,8,68],crs=ccrs.PlateCarree())
        ax.add_geometries([geometry], crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        return pn.pane.Matplotlib(fig0,dpi=144)
        
class LocSelectorPoint(param.Parameterized):
    '''
    If the user wants a timeseries that pertains to a point, they may choose from among a set of 
    pre-calculated station locations. Later this class can be extended to accomodate user-specified
    station data. Produces a panel from which to select from pre-calculated stations, and updates
    the 'location' object accordingly.
    '''
    cached_station = param.Selector(objects=_cached_stations) 
    
    @param.depends('cached_station',watch=True)
    def _update_location(self):
        '''Updates the 'location' object to be the point associated with the selected station.'''
        location = _stations_database[self.cached_station]

class CatalogContents():
    def __init__(self):
        # get the list of data variables from one of the zarr files:
        cat = intake.open_catalog('s3://cdcat/cae.yaml') 
        _ds = cat[list(cat)[0]].to_dask()
        _variable_choices_hourly_wrf = [_ds[i].attrs['description'] for i in _ds.data_vars] 
        _variable_choices_hourly_wrf = _variable_choices_hourly_wrf #+['precipitation (total)', 'wind 10m magnitude'] #which we'll derive from what's there
        #expand this dictionary to also be dependent on LOCA vs WRF:
        _variable_choices_daily_loca = ['Temperature','Maximum Relative Humidity','Minimum Relative Humidity','Solar Radiation','Wind Speed','Wind Direction','Precipitation']
        _variable_choices_hourly_loca = ['Temperature','Precipitation']

        _variable_choices_hourly = {'Dynamical': _variable_choices_hourly_wrf,
                                    'Statistical': _variable_choices_hourly_loca}
        _variable_choices_daily = {'Dynamical': _variable_choices_hourly_wrf,
                                   'Statistical': _variable_choices_daily_loca}

        self._variable_choices = {'hourly': _variable_choices_hourly,
                                          'daily': _variable_choices_daily} 

                                          # hard-coded options:
        self._scenario_choices = ['Historical Climate','Historical Reconstruction','SSP 2-4.5 -- Middle of the Road','SSP 3-7.0 -- Business as Usual', 'SSP 5-8.5 -- Burn it All']
        resolutions = []
        for name in cat:
            resolutions.append(cat[name].metadata['nominal_resolution'])
        self._resolutions = list(set(resolutions))
        
class DataSelector(param.Parameterized):
    '''
    An object to hold data parameters, which depends only on the 'param' library. 
    Currently used in '_display_select', which uses 'panel' to draw the gui, but another 
    UI could in principle be used to update these parameters instead.
    '''
    choices = CatalogContents()
    variable = param.ObjectSelector(objects=choices._variable_choices['hourly']['Dynamical'])
    timescale = param.ObjectSelector(default='hourly',objects=['hourly','daily']) #for WRF, will just coarsen data to start

    #not needed yet until we have LOCA data:
    #dyn_stat = param.ListSelector(objects=['Dynamical','Statistical'])
    
    #@param.depends('timescale','dyn_stat', watch=True)
    #def _update_variables(self):
    #    '''
    #    Updates variable choices, which are different between dynamical/statistical, and 
    #    for statistical are also different for hourly/daily.
    #    '''
    #    variables = choices._variable_choices[self.timescale][self.dyn_stat]
    #    self.param['variable'].objects = variables
    #    self.variable = variables[0]
    scenario = param.ListSelector(objects=choices._scenario_choices)
    resolution = param.ObjectSelector(default='45 km',objects=choices._resolutions)
    @param.depends('resolution',watch=True)
    def _update_scenarios(self):
        pass #add this, so that options depend on resolution selected
    #append_historical = param.Boolean()    #need to add this as well

def _display_select(selections,location,location_type='area average'):
    '''
    Called by 'select' at the beginning of the workflow, to capture user selections. Displays panel of widgets 
    from which to make selections. Location selection widgets depend on location_type argument, which can 
    have only one of two values: 'area average' or 'station'. Modifies 'selections' object, which is used 
    by generate() to build an appropriate xarray Dataset. 
    Currently, core.Application.select does not pass an argument for location_type -- 'area average' is 
    the only choice, until station-based data are available.
    '''
    assert location_type in ['area average','station'], "Please enter either 'area average' or 'station'."
    
    #_which_loc_input = {'area average': LocSelectorArea, 'station': LocSelectorPoint}
    location_chooser = pn.Row(location.param)
        
    # add in when we have LOCA data too:
    # pn.widgets.RadioButtonGroup.from_param(selections.param.dyn_stat), 
    first_col = pn.Column(selections.param.timescale,selections.param.variable, \
                              pn.widgets.CheckBoxGroup.from_param(selections.param.scenario), \
                              pn.widgets.RadioButtonGroup.from_param(selections.param.resolution))
    return pn.Row(first_col,location_chooser)
    
